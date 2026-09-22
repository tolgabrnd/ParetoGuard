# Security Policy

## Reporting a vulnerability

Please do not open a public GitHub issue for security vulnerabilities. Instead, open a
[GitHub private security advisory](../../security/advisories/new) on this repository,
or contact the maintainers directly if that option is unavailable to you. Include:

- a description of the vulnerability and its impact;
- steps to reproduce;
- affected version/commit.

We aim to acknowledge reports within a reasonable timeframe and will coordinate
disclosure with you.

## Scope and design constraints

ParetoGuard is designed with the following security constraints, enforced throughout
the codebase (see `CLAUDE.md`):

- **No secrets in the repository.** `.env` is gitignored; only `.env.example` (key
  names, no values) is committed. Provider API keys are read from environment
  variables at runtime and are never logged.
- **No arbitrary code execution from agent tools.** The tool simulator in
  `src/paretoguard/agents` exposes a fixed set of typed, deterministic tools
  (calculator, inventory/order/shipment lookups, safe local read-only queries) — no
  shell execution, no unrestricted `eval`.
- **Input validation at boundaries.** The FastAPI service validates all request
  bodies with Pydantic models; it never passes unvalidated input to providers or to
  the storage layer.
- **Budget guards.** Runtime and benchmark execution respect configured spend/call/
  concurrency limits (`PARETOGUARD_MAX_RUN_USD`, `PARETOGUARD_MAX_CALLS`,
  `PARETOGUARD_MAX_CONCURRENCY`) to prevent unbounded API spend from a
  misconfigured or compromised run.
- **Dependency auditing.** CI includes a security workflow that checks dependencies
  for known vulnerabilities.

## Supported versions

Pre-1.0: only the latest released version on `main` is supported with security fixes.
