# ParetoGuard Build Plan

This document is the authoritative roadmap for building ParetoGuard. It is executed in
phases; each phase stops for review before the next begins.

## Principles

- Never fabricate benchmark results, capabilities, or numbers.
- Every phase must leave the repository runnable and tested.
- Commits are atomic and use Conventional Commits.
- Business logic never lives in CLI or API handlers.
- CI never requires real provider credentials.

## Phase status

| Phase | Commits | Scope | Status |
|-------|---------|-------|--------|
| A | 01-05 | Toolchain, docs, contributor workflow, domain models, storage | Done |
| B | 06-11 | Provider protocol, MockProvider, runtime, OpenAI/Anthropic/Gemini adapters, telemetry | Done |
| C | 12-17 | Eval schema, graders, benchmark runner, extraction/numeric/context/tool-use suites, consistency metrics | Not started |
| D | 18-24 | Static/rule/Pareto/reliability routers, learned routing (dataset, calibrated model, optional Torch), confidence/abstention | Not started |
| E | 25-29 | Agent simulator, chaos fault injection, recovery policies (retry/fallback/circuit-breaker/escalation), resilience suites | Not started |
| F | 30-32 | Statistics (CIs, comparisons), regression detection, report generation | Not started |
| G | 33-35 | CLI, FastAPI service, dashboard | Not started |
| H | 36-41 | Test hardening, CI, performance, reproducibility docs, README, v0.1.0 release | Not started |

## Commit sequence

01. `chore: bootstrap Python project and development toolchain`
02. `docs: define architecture, engineering principles, and build plan`
03. `chore: add Claude Code project instructions and contributor workflow`
04. `feat(core): add typed domain models and configuration system`
05. `feat(storage): add versioned DuckDB experiment store`
06. `feat(providers): define provider protocol and deterministic mock provider`
07. `feat(runtime): add async execution, timeout, retry, and budget controls`
08. `feat(providers): add OpenAI provider adapter`
09. `feat(providers): add Anthropic provider adapter`
10. `feat(providers): add Gemini provider adapter`
11. `feat(telemetry): add normalized tracing and model health metrics`
12. `feat(evals): add versioned evaluation suite and task schemas`
13. `feat(evals): implement deterministic graders`
14. `feat(evals): add async benchmark runner and repeated trials`
15. `feat(evals): add structured extraction and numeric benchmark suites`
16. `feat(evals): add long-context and tool-use benchmark suites`
17. `feat(evals): add consistency and cost-per-success metrics`
18. `feat(routing): add static, round-robin, and rule-based routers`
19. `feat(routing): implement constrained Pareto routing`
20. `feat(routing): add dynamic reliability-aware routing`
21. `feat(ml): build learned routing dataset and feature pipeline`
22. `feat(ml): add calibrated supervised learned router`
23. `feat(ml): add optional PyTorch router baseline`
24. `feat(routing): add confidence-aware escalation and abstention`
25. `feat(agents): add typed tool-use agent simulator`
26. `feat(chaos): implement deterministic provider and tool fault injection`
27. `feat(recovery): add retry, fallback, and circuit-breaker policies`
28. `feat(recovery): add failure-aware model escalation`
29. `feat(benchmarks): add resilience and recovery benchmark suites`
30. `feat(stats): add confidence intervals and experiment comparison`
31. `feat(regression): detect statistically meaningful benchmark regressions`
32. `feat(reporting): generate reproducible benchmark reports and plots`
33. `feat(cli): expose benchmark, routing, chaos, and regression commands`
34. `feat(api): add FastAPI routing and experiment service`
35. `feat(dashboard): add Pareto and reliability experiment explorer`
36. `test: add integration, property, and regression test coverage`
37. `ci: add quality gates, security checks, and benchmark smoke tests`
38. `perf: optimize benchmark concurrency and routing overhead`
39. `docs: publish benchmark methodology and reproducibility guide`
40. `docs: build evidence-first README and architecture documentation`
41. `chore: prepare reproducible v0.1.0 release`

## Technical decisions

- **Dataframe library: Polars**, for native Arrow interop with DuckDB and a typed API.
  Pandas is not used in core paths.
- **Type checker: mypy --strict**, run in CI and pre-commit.
- **Package manager: uv**, `src/` layout, Python 3.12.
- **LLM SDKs used directly** (no LangChain/LlamaIndex) so orchestration logic stays
  auditable in this codebase.

## Out of scope for v0.1.0

Kubernetes, Kafka, Redis, Postgres, vector databases, auth systems, and other
infrastructure not required to demonstrate the core routing/eval/chaos thesis.
