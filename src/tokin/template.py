from __future__ import annotations

import copy
import itertools
import json
from collections.abc import Iterable
from typing import Any, ClassVar, Literal, cast, overload

from transformers import PreTrainedTokenizerBase

from .types import Message, ToolCall, ToolSchema

STUB: tuple[Message, ...] = (
    {"role": "system", "content": "tokin-stub-system"},
    {"role": "user", "content": "tokin-stub-user"},
)


class RenderError(RuntimeError):
    """The template cannot extend the served prefix; the caller has to fork."""


class ChatTemplate:
    """One model family's chat format, bound to a tokenizer.

    A subclass states the family's facts as class attributes and inherits the
    behaviour; a family without a jinja template overrides `apply` as well.

    Attributes:
        name (str): Registry key, such as `"qwen"`.
        turn_end (str): What the template writes after assistant content, such as
            `"<|im_end|>\n"`. Empty when a turn ends by the next one starting.
        stop (tuple[str, ...]): Every token the model may stop on; the server gets them as stop ids.
        arguments (Literal["dict", "str"]): Whether the template wants tool-call arguments as a mapping
            or as JSON text.
        kwargs (tuple[str, ...]): Keyword arguments the template reads, such as `enable_thinking`;
            HF passes them as `**kwargs`, the wire as `chat_template_kwargs`, and any other name
            is a mistake the template would swallow.
        models (tuple[str, ...]): Checkpoints the family was verified against; the tokenizer
            tests run over them.
    """

    name: ClassVar[str] = ""
    turn_end: ClassVar[str] = ""
    stop: ClassVar[tuple[str, ...]] = ()
    arguments: ClassVar[Literal["dict", "str"]] = "dict"
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
        """Copy with tool-call arguments in the shape this template accepts.

        OpenAI clients send `arguments` as a JSON string. Most templates index it
        as a mapping and break on the string; DeepSeek's paste it as text and
        would print a mapping's repr. Only the direction differs by family.
        """
        copies = cast(list[dict[str, Any]], copy.deepcopy(messages))
        for call in (c for m in copies for c in m.get("tool_calls") or []):
            function = call["function"]
            if self.arguments == "dict" and isinstance(function["arguments"], str):
                function["arguments"] = json.loads(function["arguments"])
            elif self.arguments == "str" and not isinstance(function["arguments"], str):
                function["arguments"] = json.dumps(function["arguments"])
        return copies

    @overload
    def apply(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        add_generation_prompt: bool = True,
        tokenize: Literal[True] = True,
    ) -> list[int]: ...

    @overload
    def apply(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        add_generation_prompt: bool = True,
        tokenize: Literal[False],
    ) -> str: ...

    def apply(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
    ) -> list[int] | str:
        """`apply_chat_template` with this family's argument shape and kwargs baked in."""
        text = self.tokenizer.apply_chat_template(
            self.conform(messages),
            tools=cast(Any, tools) or None,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **self._kwargs,
        )
        return self.encode(cast(str, text)) if tokenize else cast(str, text)

    def delta(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        tool_calls: Iterable[ToolCall] | None = None,
    ) -> list[int]:
        """Ids for `messages`, the ones after the last assistant turn, ready to follow it.

        Each message is rendered against a two-message stub instead of the real
        history, so the template never gets to rewrite earlier turns. Tool results
        go behind the `tool_calls` they answer: several templates refuse a bare
        tool message, and one renders the call id.
        """
        ids: list[int] = []
        groups = [list(g) for _, g in itertools.groupby(messages, key=lambda m: m["role"] == "tool")]
        for i, group in enumerate(groups):
            stub: list[Message] = [*STUB]
            if group[0]["role"] == "tool":
                if not tool_calls:
                    raise RenderError("tool results need the tool_calls they answer")
                # A non-empty reasoning keeps the think block in place whether or not this turn is the last.
                stub.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": " ",
                        "tool_calls": list(tool_calls),
                    }
                )
            roles = [m["role"] for m in group]
            try:
                before = self.apply(stub, tools, add_generation_prompt=False, tokenize=False)
                after = self.apply(stub + group, tools, add_generation_prompt=i == len(groups) - 1, tokenize=False)
            except Exception as e:
                raise RenderError(f"template refuses {roles}: {e}") from e
            if not after.startswith(before):
                raise RenderError(f"template rewrites earlier turns when appending {roles}")
            grown = after[len(before) :]
            # A template that skips a role renders nothing for it, and the harness would never learn.
            contents = [m.get("content") for m in group]
            if any(isinstance(c, str) and c and c not in grown for c in contents):
                raise RenderError(f"template drops {roles}")
            ids += self.encode(grown)
        return ids

    def extend(
        self,
        prompt_ids: list[int],
        completion_ids: list[int],
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        *,
        tool_calls: Iterable[ToolCall] | None = None,
    ) -> list[int]:
        """The next prompt: the served ids, the turn end the model left unfinished, then `messages`.

        Sampled ids are never altered, and the result is what the template itself
        would have written. A stop token that contradicts what follows, such as a
        secondary eos or the opener of a different role, is a `RenderError`: the
        conversation has to restart from a full render.
        """
        delta = self.delta(messages, tools, tool_calls=tool_calls)
        end = self.turn_end_ids
        k = max(k for k in range(len(end) + 1) if completion_ids[len(completion_ids) - k :] == end[:k])
        stopped = bool(completion_ids) and completion_ids[-1] in self.stop_ids
        if stopped and k == 0:
            if not end and delta and completion_ids[-1] == delta[0]:
                delta = delta[1:]
            else:
                raise RenderError(f"model stopped on id {completion_ids[-1]}, which the template cannot continue from")
        return prompt_ids + completion_ids + end[k:] + delta
