# tokin

A gateway that sits between an agent harness and an inference server: the harness
speaks text over `/v1/chat/completions`, the server only ever receives token ids.

## Commands

```bash
uv sync                       # install, including dev tooling
uv run pytest tests -q        # tests
uv run ruff check .           # lint
uv run ruff format .          # format
uv run mypy src               # types
uv run pre-commit run --all-files
```

`pre-commit` runs mypy and the fast tests through the project venv, so `uv sync`
has to have run first.

## Style

Formatting is `ruff`'s job. What a linter can't check:

### Files

No license or copyright headers — the licence lives at the repo root.

No module or package docstrings. A file starts at its first import or definition;
`D100` and `D104` are unselected for that reason.

### Docstrings

Document what the signature can't say. Types are in the signature already.

Three tiers, most code being the first:

- **None.** The name and types are enough. `def token_ids(self) -> list[int]`
  needs nothing.
- **One line.** There is a constraint or consequence a caller must know:
  `"""Append tokens the model sampled; must follow at least one prompt turn."""`
- **A short paragraph**, only when a design choice would otherwise look arbitrary
  and someone might reasonably undo it.

No `Args:` / `Returns:` sections — the signature carries the types and we don't
generate API docs from docstrings. Don't restate the function name.

For calibration, the share of functions with *no* docstring in projects we
follow: hatch 79%, httpx 52%, openai-agents 49%, pydantic 48%. Full coverage is
not the goal; `D1` stays unselected.

### Comments

One sentence. Longer only when the reasoning genuinely doesn't compress.

Comment the *why*, on the line it explains. Design rationale that needs
paragraphs belongs in the commit message, attached to the change rather than to
the code forever.

### Prose

Single backticks around identifiers and endpoints: `input_ids`, not
``input_ids``. Applies to docstrings, comments, and Markdown.

## Tests

Default to module-level test functions, no class grouping — the name carries the
context. pydantic and httpx are ~99% module-level.

Group into a class only when one function has three or more scenarios worth
covering. Then the class is named for the function under test
(`TestAddResponse`), and the methods name only the scenario — don't repeat the
function name in both. hatch is the reference here: 82% grouped, and its method
names average 3.8 words because the class already said what is being tested.

No docstrings on tests; the name is the documentation. If a case needs
justification, a one-line comment above the assertion.

## Commits

Conventional Commits, enforced by a `commit-msg` hook: `feat`, `fix`, `docs`,
`style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

`main` takes pull requests only, squash-merged, with `ci-ok` green.
