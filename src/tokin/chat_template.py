from __future__ import annotations

import copy
import itertools
import json
from collections.abc import Iterable
from typing import Any, ClassVar, cast

from transformers import PreTrainedTokenizerBase

from .tool_parser import ToolParser
from .types import (
    AssistantMessage,
    Message,
    PromptMessage,
    SystemMessage,
    ToolCall,
    ToolSchema,
    UserMessage,
)


class ChatTemplateError(RuntimeError):
    """The template cannot continue a served prompt with these messages; start over from `apply`."""


class ChatTemplate:
    """One model family's chat format, bound to a tokenizer.

    A subclass states the family's facts as class attributes and inherits the
    behaviour; a family without a jinja template overrides `apply` as well.

    Attributes:
        name (str): Registry key, such as `"qwen"`.
        turn_end (str): What the template writes after assistant content, such as
            `"<|im_end|>\n"`. Empty when a turn ends by the next one starting.
        reasoning_start, reasoning_end (str): The tags around reasoning, such as `"<think>"` and
            `"</think>"`. Empty when the family has no reasoning block.
        tool_parser (ToolParser | None): The grammar the family writes tool calls in; `None` when
            it has none.
        stop (tuple[str, ...]): Every token the model may stop on; the server gets them as stop ids.
        kwargs (tuple[str, ...]): Keyword arguments the template reads, such as `enable_thinking`;
            HF passes them as `**kwargs`, the wire as `chat_template_kwargs`, and any other name
            is a mistake the template would swallow.
        models (tuple[str, ...]): Checkpoints the family was verified against; the tokenizer
            tests run over them.
    """

    name: ClassVar[str] = ""
    turn_end: ClassVar[str] = ""
    reasoning_start: ClassVar[str] = ""
    reasoning_end: ClassVar[str] = ""
    tool_parser: ClassVar[ToolParser | None] = None
    stop: ClassVar[tuple[str, ...]] = ()
    kwargs: ClassVar[tuple[str, ...]] = ()
    models: ClassVar[tuple[str, ...]] = ()

    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs: Any) -> None:
        self.tokenizer = tokenizer
        if unknown := set(kwargs) - set(self.kwargs):
            raise ValueError(f"{self.name or type(self).__name__} takes no kwarg named {sorted(unknown)}")
        self._kwargs = kwargs
        self.stop_ids = frozenset(self.token_id(s) for s in self.stop)

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
                raise ChatTemplateError("tool results need the tool_calls they answer")
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
            raise ChatTemplateError(f"template refuses {roles}: {e}") from e
        if not after.startswith(before):
            raise ChatTemplateError(f"template rewrites earlier turns when appending {roles}")
        return after[len(before) :]

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
            raise ChatTemplateError("tool results come before any other message")
        text = self.turn_end
        if results:
            text += self.apply_after_stub(results, tools, tool_calls=tool_calls, add_generation_prompt=not rest)
        if rest:
            text += self.apply_after_stub(rest, tools, add_generation_prompt=True)
        # The model stopped on the token that opens the increment, so the prompt already has it.
        for s in self.stop:
            if text.startswith(s):
                return text[len(s) :]
        raise ChatTemplateError(f"the increment opens with {text[:16]!r}, which the model never stops on")

    def parse(
        self, response: str, tools: list[ToolSchema] | None = None, *, reasoning_open: bool = False
    ) -> AssistantMessage:
        """Read back the assistant message from what the model sampled.

        Args:
            response (str): The sampled text without its stop token.
            tools (list[ToolSchema] | None): The schemas the calls may name; XML grammars type
                their argument values by them.
            reasoning_open (bool): The prompt wrote `reasoning_start` and not `reasoning_end`, so
                `response` begins inside the reasoning block.
        """
        text = response.lstrip()
        message: AssistantMessage = {"role": "assistant", "content": None}
        if self.reasoning_end and (reasoning_open or text.startswith(self.reasoning_start)):
            head, _, text = text.partition(self.reasoning_end)
            if reasoning := head.removeprefix(self.reasoning_start).strip():
                message["reasoning_content"] = reasoning
        if self.tool_parser:
            text, calls = self.tool_parser.parse(text, tools)
            if calls:
                message["tool_calls"] = calls
        if content := text.strip():
            message["content"] = content
        return message
