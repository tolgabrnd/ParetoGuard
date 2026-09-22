"""Runtime settings and YAML configuration loaders.

Settings come from environment variables / `.env` (see `.env.example`). Provider,
model, and pricing definitions come from versioned YAML files under `configs/` —
never hard-coded in Python, so changing them never requires a code change.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from paretoguard.core.models import ModelSpec, PricingEntry, ProviderSpec


class Settings(BaseSettings):
    """Process-wide settings, read from `PARETOGUARD_*` environment variables.

    All budget guards are optional (`None` means "no limit configured"); the
    runtime is responsible for actually enforcing them (see `paretoguard.runtime`).
    """

    model_config = SettingsConfigDict(env_prefix="PARETOGUARD_", env_file=".env", extra="ignore")

    max_run_usd: float | None = Field(default=None, ge=0)
    max_calls: int | None = Field(default=None, ge=0)
    max_concurrency: int = Field(default=4, ge=1)
    config_dir: Path = Field(default=Path("configs"))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def load_providers(path: Path) -> list[ProviderSpec]:
    """Load provider definitions from a YAML file with a top-level `providers:` list."""
    data = _load_yaml(path)
    return [ProviderSpec.model_validate(entry) for entry in data.get("providers", [])]


def load_models(path: Path) -> list[ModelSpec]:
    """Load model definitions from a YAML file with a top-level `models:` list."""
    data = _load_yaml(path)
    return [ModelSpec.model_validate(entry) for entry in data.get("models", [])]


def load_pricing_entries(path: Path) -> list[PricingEntry]:
    """Load pricing entries from a YAML file with a top-level `entries:` list."""
    data = _load_yaml(path)
    return [PricingEntry.model_validate(entry) for entry in data.get("entries", [])]


class PricingTable(BaseModel):
    """Looks up the most recent applicable price for a (provider, model) pair."""

    entries: list[PricingEntry] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "PricingTable":
        return cls(entries=load_pricing_entries(path))

    def price_for(self, provider: str, model: str) -> PricingEntry | None:
        matches = [e for e in self.entries if e.provider == provider and e.model == model]
        if not matches:
            return None
        return max(matches, key=lambda e: e.effective_date)
