"""Deterministic identifier derivation.

`InferenceRequest.request_id` defaults to a fresh random UUID4 per
construction — fine for a single live call, but it silently breaks
reproducibility for any provider whose behavior is a function of the
request (e.g. `ProfiledMockProvider`'s seeded RNG, keyed on
`request.request_id`, or `MockProvider`'s FLAKY scenario without an
explicit fail count): the *same* (seed, task, candidate, repetition) would
otherwise produce a *different* simulated outcome on every process run,
contradicting this repo's reproducibility claims (CLAUDE.md: "never
fabricate benchmark results..."; every offline experiment script documents
itself as reproducible given fixed seeds). Callers that need a stable
identity for a (task, candidate, repetition) triple — `evals.matrix.
MatrixRunner`, `routing.execution.RoutedBenchmarkRunner` — should pass the
result of `deterministic_request_id()` as `InferenceRequest.request_id`
explicitly rather than relying on the random default.
"""

from uuid import NAMESPACE_OID, UUID, uuid5


def deterministic_request_id(*parts: str) -> UUID:
    """A UUID5 derived from `parts`, joined with `:` — the same `parts`
    always yields the same id, with no shared mutable state required."""
    return uuid5(NAMESPACE_OID, ":".join(parts))
