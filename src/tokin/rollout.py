from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    """A contiguous run of token ids, either from the prompt or sampled by the model."""

    token_ids: list[int]
    generated: bool
    logprobs: list[float] | None = None

    def __post_init__(self) -> None:
        # Whether logprobs are required is the trainer's call; alignment is not.
        if self.logprobs is not None and len(self.logprobs) != len(self.token_ids):
            raise ValueError(f"logprobs has {len(self.logprobs)} entries but token_ids has {len(self.token_ids)}")

    def __len__(self) -> int:
        return len(self.token_ids)


@dataclass
class Rollout:
    """The turns of one conversation, in order."""

    turns: list[Turn] = field(default_factory=list)

    def add_prompt(self, token_ids: list[int]) -> None:
        """Add prompt (input) tokens."""
        # A zero-length turn adds no tokens but still reads as a segment boundary.
        if token_ids:
            self.turns.append(Turn(token_ids=list(token_ids), generated=False))

    def add_response(self, token_ids: list[int], logprobs: list[float] | None = None) -> None:
        """Add model-generated (output) tokens."""
        if not token_ids:
            return
        if not self.turns:
            raise RuntimeError("a rollout cannot open with a generated turn; add_prompt first")
        self.turns.append(
            Turn(
                token_ids=list(token_ids),
                generated=True,
                logprobs=None if logprobs is None else list(logprobs),
            )
        )

    @property
    def token_ids(self) -> list[int]:
        """Every turn's ids concatenated — the prefix the next prompt extends."""
        return [i for turn in self.turns for i in turn.token_ids]

    def __len__(self) -> int:
        return sum(len(turn) for turn in self.turns)

    def __repr__(self) -> str:
        generated = sum(len(t) for t in self.turns if t.generated)
        return f"Rollout(turns={len(self.turns)}, tokens={len(self)}, generated={generated})"
