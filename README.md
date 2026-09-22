# ParetoGuard

Adaptive, reliability-aware routing and chaos evaluation for production LLM and agent systems.

> Benchmark models. Route under constraints. Inject failures. Measure recovery.

**Status: early development.** Core domain models and storage are in place; routing,
evaluation suites, chaos engineering, and the CLI/API/dashboard are not yet
implemented. See [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md) for the phased roadmap and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the system design.

No benchmark numbers are published yet — none will be until they come from an actual
committed, reproducible experiment. See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for
what is and isn't claimed at any given point in development.

## Why

Production LLM applications rarely benefit from sending every request to one fixed
model. Tasks differ in required reasoning quality, structured-output reliability,
tool-use reliability, cost, latency, and context length — and providers themselves
fail: rate limits, 5xx errors, timeouts, malformed output, degraded availability.
ParetoGuard is a platform for measuring those trade-offs and failure modes, and for
building routing/recovery logic that responds to them, rather than assuming a fixed
"best" model.

## Development

```bash
uv sync
uv run pytest tests/unit -v
uv run ruff check .
uv run mypy src/paretoguard
```

Full quickstart, offline demo, and provider setup instructions will be published once
the corresponding subsystems exist (tracked in `docs/BUILD_PLAN.md`).

## License

Apache-2.0. See [LICENSE](LICENSE).
