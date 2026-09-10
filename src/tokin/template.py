from __future__ import annotations

import copy
import itertools
import json
from collections.abc import Iterable
from typing import Any, ClassVar, cast

from transformers import PreTrainedTokenizerBase

from .types import AssistantMessage, Message, PromptMessage, SystemMessage, ToolCall, ToolSchema, UserMessage


class TemplateError(RuntimeError):
    """The template cannot continue a served prompt with these messages; start over from `apply`."""


class ChatTemplate:
    """One model family's chat format, bound to a tokenizer.

    A subclass states the family's facts as class attributes and inherits the
    behaviour; a family without a jinja template overrides `apply` as well.

    Attributes:
        name (str): Registry key, such as `"qwen"`.
        turn_end (str): What the template writes after assistant content, such as
            `"<|im_end|>\n"`. Empty when a turn ends by the next one starting.
        stop (tuple[str, ...]): Every token the model may stop on; the server gets them as stop ids.
        kwargs (tuple[str, ...]): Keyword arguments the template reads, such as `enable_thinking`;
            HF passes them as `**kwargs`, the wire as `chat_template_kwargs`, and any other name
            is a mistake the template would swallow.
        models (tuple[str, ...]): Checkpoints the family was verified against; the tokenizer
            tests run over them.
    """

    name: ClassVar[str] = ""
    turn_end: ClassVar[str] = ""
    stop: ClassVar[tuple[str, ...]] = ()
    kwargs: ClassVar[tuple[str, ...]] = ()
    models: ClassVar[tuple[str, ...]] = ()

    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs: Any) -> None:
        self.tokenizer = tokenizer
        if unknown := set(kwargs) - set(self.kwargs):
            raise ValueError(f"{self.name or type(self).__name__} takes no kwarg named {sorted(unknown)}")
        self._kwargs = kwargs
        self.stop_ids = frozenset(self.token_id(s) for s in self.stop)
        self.turn_end_ids = self.encode(self.turn_end)

    def encode(self, text: str) -> list[int]:
        """Never adds special tokens: the template already wrote the ones it wants."""
        return cast(list[int], self.tokenizer.encode(text, add_special_tokens=False))

    def token_id(self, token: str) -> int:
        """Input `token` has to be exactly one token here, or the family and the tokenizer do not match."""
        ids = self.encode(token)
        if len(ids) != 1:
            raise ValueError(f"{token!r} is {len(ids)} tokens for this tokenizer, not one")
        return ids[0]

    def conform(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Copy `messages` with each tool call's `arguments` parsed into a dict, the shape HF templates read."""
        copies = cast(list[dict[str, Any]], copy.deepcopy(messages))
        for call in (c for m in copies for c in m.get("tool_calls") or []):
            if isinstance(call["function"]["arguments"], str):
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
        return copies

    def apply(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply the tokenizer's built-in chat template with this family's argument shape and kwargs baked in."""
        text = self.tokenizer.apply_chat_template(
            self.conform(messages),
            tools=cast(Any, tools) or None,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **self._kwargs,
        )
        return cast(str, text)

    def stub(self, messages: list[PromptMessage], tool_calls: Iterable[ToolCall] | None) -> list[Message]:
        """A stub history to put before `messages` so the chat template will render them."""
        stub: list[Message] = [
            SystemMessage(role="system", content="tokin-stub-system"),
            UserMessage(role="user", content="tokin-stub-user"),
        ]
        if messages[0]["role"] == "tool":
            if not tool_calls:
                raise TemplateError("tool results need the tool_calls they answer")
            # A non-empty reasoning keeps the think block in place whether or not this turn is the last one.
            stub.append(
                AssistantMessage(role="assistant", content="", reasoning_content=" ", tool_calls=list(tool_calls))
            )
        return stub

    def apply_after_stub(
        self,
        messages: list[PromptMessage],
        tools: list[ToolSchema] | None = None,
        *,
        tool_calls: Iterable[ToolCall] | None = None,
        add_generation_prompt: bool,
    ) -> str:
        """Apply the chat template to `messages` alone by rendering them after a stub history and subtracting it."""
        stub = self.stub(messages, tool_calls)
        roles = [m["role"] for m in messages]
        try:
            before = self.apply(stub, tools, add_generation_prompt=False)
            after = self.apply([*stub, *messages], tools, add_generation_prompt=add_generation_prompt)
        except Exception as e:
            raise TemplateError(f"template refuses {roles}: {e}") from e
        if not after.startswith(before):
            raise TemplateError(f"template rewrites earlier turns when appending {roles}")
        grown = after[len(before) :]
        if any(isinstance(c, str) and c and c not in grown for c in (m.get("content") for m in messages)):
            raise TemplateError(f"template drops {roles}")
        return grown

    def apply_increment(
        self,
        messages: list[PromptMessage],
        tools: list[ToolSchema] | None = None,
        *,
        tool_calls: Iterable[ToolCall] | None = None,
    ) -> str:
        """Apply the chat template to `messages` as the next increment of a multi-turn rollout."""
        results = list(itertools.takewhile(lambda m: m["role"] == "tool", messages))
        rest = messages[len(results) :]
        if any(m["role"] == "tool" for m in rest):
            raise TemplateError("tool results come before any other message")
        # The model stops on the turn end's first token and never writes the rest, so the increment starts with it.
        text = cast(str, self.tokenizer.decode(self.turn_end_ids[1:]))
        if results:
            text += self.apply_after_stub(results, tools, tool_calls=tool_calls, add_generation_prompt=not rest)
        if rest:
            text += self.apply_after_stub(rest, tools, add_generation_prompt=True)
        return text
