from __future__ import annotations

from collections.abc import Mapping
from http import HTTPStatus
from typing import Any, ClassVar, cast

from .backends import GenerationBackend, GenerationError, GenerationParams
from .chat_templates import ChatTemplate
from .messages import AssistantMessage, Message, PromptMessage, ToolSchema
from .rollout import FinishReason, Generation, Prompt, Rollout
from .session import Session


class GatewayError(Exception):
    """A request the gateway refuses; the subclass says with which status."""

    status_code: ClassVar[HTTPStatus] = HTTPStatus.INTERNAL_SERVER_ERROR


class SessionNotFoundError(GatewayError):
    status_code = HTTPStatus.NOT_FOUND


class DuplicateSessionError(GatewayError):
    status_code = HTTPStatus.CONFLICT


class PromptTooLongError(GatewayError):
    status_code = HTTPStatus.BAD_REQUEST


class HistoryConflictError(GatewayError):
    """The request does not extend what the session served, so its prompt would not be the model's context."""

    status_code = HTTPStatus.CONFLICT


class GenerationAbortedError(GatewayError):
    """The engine cut the generation short; nothing was recorded, so the harness may send the turn again."""

    status_code = HTTPStatus.SERVICE_UNAVAILABLE


class Gateway:
    """tokin's core as a plain object; the HTTP layer maps routes onto its methods.

    Free of any web framework, so a trainer can drive it in-process and tests
    exercise it without a client. What it refuses, it refuses with a
    `GatewayError` whose subclass names the HTTP status, and the HTTP layer
    renders that once instead of choosing a status route by route.

    Sessions live in memory, keyed by id, from `create` until `delete`. There is
    no expiry: only the harness knows when a run is over, and the trainer reads
    the rollouts back after the last turn, so an idle session is not a stale one.
    """

    def __init__(
        self,
        template: ChatTemplate,
        backend: GenerationBackend,
        *,
        context_length: int,
        return_logprobs: bool = False,
        routed_experts: bool = False,
    ) -> None:
        self.template = template
        self.backend = backend
        self.context_length = context_length
        self.return_logprobs = return_logprobs
        self.routed_experts = routed_experts
        self.sessions: dict[str, Session] = {}

    def create(self, id: str | None = None) -> Session:
        session = Session(id) if id else Session()
        if session.id in self.sessions:
            raise DuplicateSessionError(f"session {session.id!r} already exists")
        self.sessions[session.id] = session
        return session

    def get(self, id: str) -> Session:
        try:
            return self.sessions[id]
        except KeyError:
            raise SessionNotFoundError(f"no session {id!r}") from None

    def delete(self, id: str) -> Session:
        return self.sessions.pop(self.get(id).id)

    def prompt(self, rollout: Rollout, messages: list[Message], tools: list[ToolSchema] | None) -> str:
        """The text `messages` add to `rollout`: all of them on the first turn, those past what was served after."""
        if not rollout.segments:
            return self.template.apply(messages, tools)
        served, tail = messages[: len(rollout.messages)], messages[len(rollout.messages) :]
        # The harness wrote every message but the assistant's, which its SDK may reshape; the rest come back as served.
        if not tail or any(
            new["role"] != old["role"] or (old["role"] != "assistant" and new != old)
            for old, new in zip(rollout.messages, served, strict=True)
        ):
            raise HistoryConflictError("the messages do not extend what this session served")
        if any(m["role"] == "assistant" for m in tail):
            raise HistoryConflictError("the new messages hold an assistant message tokin did not generate")
        last = rollout.segments[-1]
        if not (isinstance(last, Generation) and last.finish_reason is FinishReason.STOP):
            raise HistoryConflictError("the last generation did not end on a stop token, so it cannot be extended")
        answered = cast(AssistantMessage, rollout.messages[-1]).get("tool_calls")
        return self.template.apply_increment(cast(list[PromptMessage], tail), tools, tool_calls=answered)

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        params: Mapping[str, Any] | None = None,
        session_id: str | None = None,
    ) -> tuple[AssistantMessage, Generation, int]:
        """One turn: the reply parsed, the generation it came from, and how many tokens the engine was given.

        `params` is what the harness asked of the engine under `GenerationParams`' names; its
        `max_tokens` is capped to the context left. Without `session_id` the turn runs on a session
        of its own, which nothing keeps.
        """
        session = self.get(session_id) if session_id else Session()
        async with session.lock:
            rollout = session.current
            text = self.prompt(rollout, messages, tools)
            prompt = Prompt(self.template.encode(text))
            room = self.context_length - len(rollout) - len(prompt)
            if room < 1:
                raise PromptTooLongError(
                    f"{len(rollout) + len(prompt)} prompt tokens leave no room in {self.context_length}"
                )
            params = dict(params or {})
            params["max_tokens"] = min(params.get("max_tokens") or room, room)
            params["stop_ids"] = self.template.stop_ids
            if self.return_logprobs:
                params["return_logprobs"] = True
            if self.routed_experts:
                params["routed_experts_start"] = max(0, len(rollout) - 1)
            generation = await self.backend.generate(
                rollout.token_ids + prompt.token_ids, cast(GenerationParams, params)
            )
            if generation.finish_reason is FinishReason.ABORT:
                raise GenerationAbortedError("the engine aborted the generation")
            stopped = generation.finish_reason is FinishReason.STOP
            # The next increment drops its opening stop token on the promise that the model sampled one.
            if stopped and generation.token_ids[-1:] not in ([i] for i in self.template.stop_ids):
                raise GenerationError(
                    f"the engine reported a stop ending on {generation.token_ids[-1:]}, not a stop id"
                )
            rollout.append(prompt)
            rollout.append(generation)
            sampled = generation.token_ids[:-1] if stopped else generation.token_ids
            reasoning_open = text.rfind(self.template.reasoning_start) > text.rfind(self.template.reasoning_end)
            message = self.template.parse(
                cast(str, self.template.tokenizer.decode(sampled)), tools, reasoning_open=reasoning_open
            )
            rollout.messages.extend(messages[len(rollout.messages) :])
            rollout.messages.append(message)
            return message, generation, len(rollout) - len(generation)
