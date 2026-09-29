from __future__ import annotations

from collections.abc import Collection
from typing import Annotated, Any, ClassVar, Literal, cast

from openai.types.shared_params import ResponseFormatText
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from ..backends import GenerationParams
from ..messages import Message, ToolSchema


class ChatRequest(BaseModel, extra="forbid"):
    """The harness's `/v1/chat/completions` body, as much of OpenAI's as `tokin` serves.

    A key not declared is refused by name rather than dropped, since dropping it would change
    what the harness gets back without telling it; serving one more is declaring it.
    """

    model: str
    messages: list[Message]
    tools: list[ToolSchema] | None = None
    stream: bool = False

    max_tokens: int | None = Field(None, validation_alias=AliasChoices("max_completion_tokens", "max_tokens"))
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    # Engine extensions, which a harness sends through `extra_body`.
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None

    # What `tokin` cannot honour, narrowed to the values that ask for nothing, since SDKs send those unasked.
    n: Literal[1] = 1
    stop: Annotated[list[str], Field(max_length=0)] | None = None
    tool_choice: Literal["auto"] | None = None
    parallel_tool_calls: Literal[True] | None = None
    logprobs: Literal[False] | None = None
    response_format: ResponseFormatText | None = None

    # Bookkeeping OpenAI accepts and `tokin` has no use for.
    user: str | None = None
    safety_identifier: str | None = None
    metadata: dict[str, str] | None = None
    store: bool | None = None
    stream_options: dict[str, Any] | None = None
    service_tier: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_options: dict[str, Any] | None = None
    prompt_cache_retention: str | None = None

    # SDKs decorate messages with keys no template reads, so messages drop what the body refuses.
    messages_adapter: ClassVar[TypeAdapter[list[Message]]] = TypeAdapter(
        list[Annotated[Message, Field(discriminator="role")]], config=ConfigDict(extra="ignore")
    )

    @field_validator("messages", mode="plain")
    @classmethod
    def lenient(cls, value: Any) -> list[Message]:
        """Validate `messages` by the adapter, reading OpenAI's `Iterable` fields into lists so a bad item fails here as a 400."""
        messages = cls.messages_adapter.validate_python(value)
        for m in (cast(dict[str, Any], m) for m in messages):
            for key in ("content", "tool_calls"):
                if not isinstance(value := m.get(key), str | None):
                    m[key] = list(value)
        return messages

    def params(self, max_tokens: int, stop_ids: Collection[int]) -> GenerationParams:
        """What the harness asked, for the engine; `max_tokens` is the caller's so it can be capped to the context left."""
        # A field named as a `GenerationParams` key means the same to the engine; the caller's two keys go last to win.
        asked = self.model_dump(include=set(GenerationParams.__optional_keys__), exclude_none=True)
        return cast(GenerationParams, {**asked, "max_tokens": max_tokens, "stop_ids": stop_ids})
