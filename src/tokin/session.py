from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import uuid4

from .rollout import Rollout


@dataclass
class Session:
    """One agent run, as the rollouts it produced.

    Usually one: turns append to it in place. A harness that rewrites history —
    context compaction, a re-rendered prior turn — breaks the prefix the next
    prompt would extend, and the run continues in a fresh rollout via `fork`.
    Every rollout here shares an outcome, which is what lets a trainer weight the
    run once rather than once per fork.
    """

    id: str = field(default_factory=lambda: uuid4().hex)
    rollouts: list[Rollout] = field(default_factory=lambda: [Rollout()])
    # One turn at a time; the gateway holds it from prompt to parse.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, compare=False)

    @property
    def current(self) -> Rollout:
        """The rollout new turns append to."""
        return self.rollouts[-1]

    def fork(self) -> Rollout:
        """Start a fresh rollout, leaving the previous one closed but trainable."""
        self.rollouts.append(Rollout())
        return self.rollouts[-1]

    def __len__(self) -> int:
        return sum(len(rollout) for rollout in self.rollouts)

    def __repr__(self) -> str:
        return f"Session(id={self.id!r}, rollouts={len(self.rollouts)}, tokens={len(self)})"
