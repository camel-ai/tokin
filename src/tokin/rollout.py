from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Prompt:
    """Ids the harness's messages rendered to, as the engine received them."""

    token_ids: list[int]

    def __len__(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True, slots=True)
class Generation:
    """What the model sampled for one prompt, as ids, including the stop id it ended on.

    `routed_experts` is the engine's buffer as it came: int32, C order, shape `(positions, layers, top_k)`
    flattened, positions running from `routed_experts_start` up to the last sampled token, which predicts
    nothing. The trainer reshapes it with its model's layer count and top-k; MoE training replays the routing.
    """

    token_ids: list[int]
    finish_reason: Literal["stop", "length", "abort"]
    logprobs: list[float] | None = None
    routed_experts: bytes | None = None

    def __post_init__(self) -> None:
        # Whether logprobs are wanted is the trainer's call; alignment is not.
        if self.logprobs is not None and len(self.logprobs) != len(self.token_ids):
            raise ValueError(f"logprobs has {len(self.logprobs)} entries but token_ids has {len(self.token_ids)}")

    def __len__(self) -> int:
        return len(self.token_ids)


@dataclass
class Rollout:
    """The prompts and generations of one conversation, in order."""

    segments: list[Prompt | Generation] = field(default_factory=list)

    @property
    def token_ids(self) -> list[int]:
        """Every segment's ids concatenated — the prefix the next prompt extends."""
        return [i for s in self.segments for i in s.token_ids]

    def append(self, segment: Prompt | Generation) -> None:
        """Add a segment; a rollout opens with a prompt, and an empty segment adds nothing."""
        if not segment.token_ids:
            return
        if not self.segments and isinstance(segment, Generation):
            raise RuntimeError("a rollout cannot open with a generation")
        self.segments.append(segment)

    def routed_experts(self, layers: int, top_k: int) -> NDArray[np.int32]:
        """Every position's expert routing so far, shape `(len(self) - 1, layers, top_k)`, joined from the per-call slices.

        Each generation's slice must run from where the previous one ended to its own second-to-last
        token; the engine sometimes appends one extra row for the final token, which is dropped.
        """
        covered, tokens, chunks = 0, 0, []
        for i, segment in enumerate(self.segments):
            tokens += len(segment)
            if not isinstance(segment, Generation):
                continue
            if segment.routed_experts is None:
                raise ValueError(f"segment {i} was generated without its expert routing")
            # A token's routing is produced when it is fed in, so every token so far has a row but the newest.
            positions = tokens - 1
            x = np.frombuffer(segment.routed_experts, dtype=np.int32)
            rows, rest = divmod(x.size, layers * top_k)
            # +1 is sglang having fed the final token once more, as after an abort; the next turn owns it.
            if rest or rows not in (positions - covered, positions - covered + 1):
                raise ValueError(f"segment {i} carries {rows} routing rows for {positions - covered} positions")
            chunks.append(x.reshape(rows, layers, top_k)[: positions - covered])
            covered = positions
        return np.concatenate(chunks) if chunks else np.empty((0, layers, top_k), dtype=np.int32)

    def __len__(self) -> int:
        return sum(len(s) for s in self.segments)

    def __repr__(self) -> str:
        generated = sum(len(s) for s in self.segments if isinstance(s, Generation))
        return f"Rollout(segments={len(self.segments)}, tokens={len(self)}, generated={generated})"
