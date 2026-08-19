from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Protocol

Messages = list[dict[str, Any]]
Tools = list[dict[str, Any]] | None

# The prefix new turns are rendered against. Its only job is to give the template
# a syntactically complete conversation to append to, so the content is inert.
_FAKE_PREFIX: Messages = [
    {"role": "system", "content": "F"},
    {"role": "user", "content": "F"},
]


class Tokenizer(Protocol):
    def apply_chat_template(self, conversation: Messages, /, **kwargs: Any) -> Any:
        """Render a conversation to text or ids."""

    def encode(self, text: str, /, **kwargs: Any) -> list[int]:
        """Turn text into ids."""

    def decode(self, token_ids: list[int], /, **kwargs: Any) -> str:
        """Turn ids back into text."""


class RenderError(RuntimeError):
    """The template would not produce a usable delta."""


@dataclass(frozen=True)
class Hooks:
    """Per-family overrides. Both default to behaviour that holds across families.

    `seam` fixes up the join between what the model emitted and the next delta.
    The default inserts the separator probed from the template, which is what
    every family tested needs; an override is for the case a render cannot see —
    a model that emits its own stop token where the delta already carries one.

    `delta` replaces incremental rendering wholesale, for a tokenizer with no
    Jinja template to diff against.
    """

    seam: Callable[[list[int], list[int]], list[int]] | None = None
    delta: Callable[[Messages, Tools], list[int]] | None = None


class Renderer:
    """Renders new turns to the token ids that extend a rollout.

    A conversation's ids are built once and then only appended to, so the ids a
    model emitted are the ids it sees next turn. Getting there needs the tokens
    for just the appended messages, which is not the same as encoding them on
    their own — the template frames every message with role headers and
    separators that only appear in context.

    So the delta is taken by difference against an inert two-message prefix. That
    the prefix is fake rather than the real history is what makes this work on
    templates that rewrite history when new turns arrive: with no prior assistant
    turn in the list, there is nothing for the template to rewrite.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        template_kwargs: dict[str, Any] | None = None,
        hooks: Hooks | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.template_kwargs = dict(template_kwargs or {})
        self.hooks = hooks or Hooks()

    def render(self, messages: Messages, tools: Tools = None) -> list[int]:
        """Render a whole conversation, ready for the model to continue it."""
        return self.encode(self._apply(messages, tools, generation=True))

    def delta(self, appended: Messages, tools: Tools = None) -> list[int]:
        """The ids for `appended`, framed as a continuation of a conversation."""
        if not appended:
            return []
        if self.hooks.delta is not None:
            return self.hooks.delta(appended, tools)

        without = self._apply(_FAKE_PREFIX, tools, generation=False)
        with_ = self._apply(_FAKE_PREFIX + appended, tools, generation=True)
        if not with_.startswith(without):
            roles = [m.get("role") for m in appended]
            raise RenderError(f"appending {roles} did not extend the rendered prefix")
        return self.encode(with_[len(without) :])

    def extend(self, prefix: list[int], appended: Messages, tools: Tools = None) -> list[int]:
        """The full prompt: a rollout's ids, then the new turns."""
        delta = self.delta(appended, tools)
        if not prefix:
            return delta
        if self.hooks.seam is not None:
            return self.hooks.seam(list(prefix), delta)
        return list(prefix) + self.separator_ids + delta

    def encode(self, text: str) -> list[int]:
        """Encode text the template already framed, so no special tokens are added."""
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        """Decode generated ids, keeping the markers a harness parses tool calls from."""
        if not token_ids:
            return ""
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=False))

    @cached_property
    def separator_ids(self) -> list[int]:
        """What the template puts between a closed assistant turn and the next message.

        A model stops at its stop token without emitting whatever follows it, so
        that text has to be supplied when joining. Probed rather than configured:
        render a terminal assistant turn, take what trails the marker, and drop
        the leading stop token.
        """
        probe = self._apply(
            [
                {"role": "user", "content": "U"},
                {"role": "assistant", "content": "__TOKIN__"},
            ],
            None,
            generation=False,
        )
        _, _, tail = probe.partition("__TOKIN__")
        return self.encode(tail)[1:]

    def _apply(self, messages: Messages, tools: Tools, *, generation: bool) -> str:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": generation,
            **self.template_kwargs,
        }
        if tools:
            kwargs["tools"] = tools
        return str(self.tokenizer.apply_chat_template(messages, **kwargs))
