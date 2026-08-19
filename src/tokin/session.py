from __future__ import annotations

from dataclasses import dataclass, field

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

    rollouts: list[Rollout] = field(default_factory=lambda: [Rollout()])

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
        return f"Session(rollouts={len(self.rollouts)}, tokens={len(self)})"
