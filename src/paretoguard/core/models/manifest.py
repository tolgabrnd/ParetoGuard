"""Run manifest: reproducibility metadata captured once per benchmark/experiment run."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class RunManifest(BaseModel):
    """Everything needed to understand how a run was produced.

    See docs/REPRODUCIBILITY.md (added when the benchmark runner lands) for which
    fields make a run reproducible and which don't (e.g. live-provider nondeterminism).
    """

    run_id: str
    paretoguard_version: str
    git_sha: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    os: str
    python_version: str
    seed: int | None = None
    suite_name: str | None = None
    suite_version: str | None = None
    router_name: str | None = None
    router_config: dict[str, Any] = Field(default_factory=dict)
    pricing_config_version: str | None = None
    task_count: int | None = Field(default=None, ge=0)
    repetitions: int = Field(default=1, ge=1)
    environment: dict[str, str] = Field(
        default_factory=dict, description="Non-secret environment metadata only."
    )
