from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Required, TypedDict


class GenerationError(RuntimeError):
    """The inference server answered outside what the protocol allows."""


class GenerationParams(TypedDict, total=False):
    """Parameters for a single generation request; a key left out takes the model's own default."""

    # Identity: names the request to the inference engine.
    request_id: str

    # Length and stopping.
    max_tokens: Required[int]
    stop_ids: Required[Collection[int]]

    # Distribution: how the next token is drawn.
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    seed: int

    # Penalties on tokens already in the sequence.
    frequency_penalty: float
    presence_penalty: float
    repetition_penalty: float

    # Constrained decoding / Structured output.
    response_schema: dict[str, Any]

    # Reporting: what comes back besides the ids; none of it changes them.
    return_logprobs: bool


@dataclass(frozen=True, slots=True)
class Generation:
    """What the model sampled for one prompt, as ids, including the stop id it ended on."""

    token_ids: list[int]
    finish_reason: Literal["stop", "length", "abort"]
    logprobs: list[float] | None = None


class GenerationBackend(ABC):
    """An inference server spoken to in token ids only."""

    # Where the engine spells a `GenerationParams` key differently; a key not listed keeps its name.
    param_map: ClassVar[dict[str, str]] = {}
    # How a value is reshaped for the engine's wire format; a key not listed goes as it is.
    param_convert: ClassVar[dict[str, Callable[[Any], Any]]] = {}

    def translate(self, params: GenerationParams) -> dict[str, Any]:
        """`params` under the engine's names and shapes; two keys landing on one name is refused."""
        out: dict[str, Any] = {}
        for key, value in params.items():
            name = self.param_map.get(key, key)
            if name in out:
                raise ValueError(f"{key!r} lands on {name!r}, which another key already set")
            out[name] = self.param_convert[key](value) if key in self.param_convert else value
        return out

    @abstractmethod
    async def generate(
        self,
        token_ids: list[int],
        params: GenerationParams,
    ) -> Generation:
        raise NotImplementedError
