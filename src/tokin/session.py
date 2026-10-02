from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

from .rollout import Rollout


@dataclass
class Session:
    """One agent run, as the rollouts it produced; turns append to `current` in place.

    A list although a run has one rollout for now: a run whose history is rewritten
    continues in a fresh one, and every rollout of a run shares its outcome.
    """

    id: str = field(default_factory=lambda: uuid4().hex)
    rollouts: list[Rollout] = field(default_factory=lambda: [Rollout()])
    # One turn at a time; the gateway holds it from prompt to parse.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, compare=False)

    @property
    def current(self) -> Rollout:
        """The rollout new turns append to."""
        return self.rollouts[-1]

    def __len__(self) -> int:
        return sum(len(rollout) for rollout in self.rollouts)

    def __repr__(self) -> str:
        return f"Session(id={self.id!r}, rollouts={len(self.rollouts)}, tokens={len(self)})"
