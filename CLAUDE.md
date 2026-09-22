# CLAUDE.md

Instructions for any Claude Code session (or other AI coding agent) working in this
repository. Read `docs/BUILD_PLAN.md` and `docs/ARCHITECTURE.md` first.

## Engineering principles

- Never fabricate benchmark results, capabilities, costs, or reliability numbers.
  Every number in docs/README comes from a real, reproducible run or is explicitly
  labeled as an example/mock value.
- Never call paid provider APIs from tests. `tests/` must pass with zero network
  access and zero API keys; live-provider tests are marked `@pytest.mark.live` and
  are opt-in only.
- Never commit secrets. `.env` is gitignored; only `.env.example` (names, no values)
  is committed.
- Prefer deterministic evaluation. LLM-as-judge is a separate, clearly labeled path —
  never silently mixed into primary/deterministic metrics.
- Maintain strict module boundaries (see the table in `docs/ARCHITECTURE.md`).
  `routing` never calls a provider directly; `providers` never knows about routing or
  evaluation; `cli` and `api` contain no business logic, only parsing/formatting over
  library calls.
- Public functions/classes in `src/paretoguard` are fully typed. `mypy --strict`
  must pass.
- Run lint, type checks, and the test suite before every commit.
- Keep commits atomic and use Conventional Commits (`feat(scope): ...`,
  `fix(scope): ...`, `docs: ...`, `chore: ...`, `test: ...`, `ci: ...`, `perf: ...`).
- Update the relevant doc (`ARCHITECTURE.md`, `EVALUATION.md`, `ROUTING.md`, etc.)
  in the same commit whenever behavior or a module boundary changes.
- Preserve backwards compatibility for public APIs unless a break is intentional and
  documented in `CHANGELOG.md`.
- Do not hard-code assumptions about which current model is "best" — model
  capability is configuration (`configs/models.example.yaml`), not a code constant.
- No unbounded agent loops (`agents` enforces max steps/tool calls/cost/wall time).
- No unbounded API spend: runtime and benchmark code must respect
  `PARETOGUARD_MAX_RUN_USD`, `PARETOGUARD_MAX_CALLS`, `PARETOGUARD_MAX_CONCURRENCY`.

## Repository commands

```bash
uv sync                          # install base dependencies
uv sync --all-extras             # install with optional extras (currently: torch, for the future PyTorch router)
uv run pytest tests/unit -v      # unit tests (no network, no keys)
uv run ruff check .              # lint
uv run ruff format .             # format
uv run mypy src/paretoguard      # type check
uv build                         # build package
```

As later phases land: `uv run paretoguard benchmark --provider mock --suite <name>`,
`uv run paretoguard chaos ...`, `uv run paretoguard report <run-id>` (documented fully
once implemented — see `docs/BUILD_PLAN.md` for status).

## Git policy

Never:
- rewrite history without explicit instruction from the user;
- force push;
- backdate commits or create empty commits to inflate activity;
- commit failing code, secrets, generated caches, or local DuckDB files;
- push to any remote unless explicitly instructed.

Before every commit:
1. inspect `git status`;
2. inspect `git diff`;
3. run the quality gates relevant to the change (lint, typecheck, tests);
4. verify no secrets are staged;
5. verify the commit scope is cohesive (split if it isn't);
6. write the message using Conventional Commits.

## Working in phases

This project is built in phases defined in `docs/BUILD_PLAN.md`. At the start of a
session: check `git log`, check `docs/BUILD_PLAN.md`'s status table, and run the test
suite before writing new code. Implement only the assigned milestone group, then stop
and report rather than continuing into the next phase unprompted.
