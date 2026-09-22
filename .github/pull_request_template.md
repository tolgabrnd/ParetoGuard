## Summary

<!-- What does this PR change, and why? -->

## Related issue

<!-- Closes #... -->

## Checklist

- [ ] `uv run ruff check .` passes
- [ ] `uv run ruff format --check .` passes
- [ ] `uv run mypy src/paretoguard` passes
- [ ] `uv run pytest tests/unit tests/integration tests/property tests/regression -v` passes
- [ ] No tests added that call paid provider APIs (use `MockProvider`, or mark `@pytest.mark.live`)
- [ ] No secrets committed
- [ ] Docs updated if behavior or module boundaries changed
- [ ] No fabricated/hard-coded benchmark numbers added to docs or README

## Notes for reviewers

<!-- Anything that needs extra scrutiny, e.g. new module boundaries, new dependencies. -->
