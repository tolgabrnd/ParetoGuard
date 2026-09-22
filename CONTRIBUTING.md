# Contributing to ParetoGuard

Thanks for considering a contribution. This project prioritizes engineering rigor and
honest evaluation over surface polish — please read this before opening a PR.

## Getting started

```bash
git clone <your-fork-url>
cd paretoguard
uv sync
uv run pytest tests/unit -v
```

No API keys are required to develop, test, or run the offline demo.

## Development workflow

1. Open an issue or comment on an existing one before large changes, so design
   direction can be agreed on first.
2. Create a branch off `main`.
3. Make focused, atomic commits using [Conventional Commits](https://www.conventionalcommits.org/).
4. Before opening a PR, run:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy src/paretoguard
   uv run pytest tests/unit tests/integration tests/property tests/regression -v
   ```
5. Update documentation in the same PR whenever you change behavior or a module
   boundary (see `docs/ARCHITECTURE.md`).

## Rules that will get a PR sent back for changes

- **No fabricated numbers.** Any benchmark result, cost figure, or reliability
  percentage added to docs/README must come from a script/test that reproduces it, or
  be explicitly labeled as an example/mock value.
- **No tests that call paid provider APIs.** Use `MockProvider` or mark the test
  `@pytest.mark.live` (excluded from CI by default).
- **No business logic in `cli/` or `api/`.** Those layers parse input and format
  output; the logic belongs in the corresponding library module.
- **No secrets committed**, including in test fixtures or notebooks.

## Code style

- Full type hints on public functions/classes; `mypy --strict` must pass.
- Docstrings on public APIs explain *why*, not restate the signature.
- No comments that just narrate what the next line does.
- Prefer small, cohesive modules over large multi-purpose files.

## Reporting bugs / requesting features

Use the issue templates in `.github/ISSUE_TEMPLATE/`. For security issues, see
`SECURITY.md` — do not open a public issue.
