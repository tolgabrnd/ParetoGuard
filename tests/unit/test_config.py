"""Unit tests for settings and YAML config loading."""

from datetime import date
from pathlib import Path

from paretoguard.core.config import PricingTable, Settings, load_models, load_providers
from paretoguard.core.models import PricingEntry

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def test_settings_defaults_have_no_spend_limit() -> None:
    settings = Settings(_env_file=None)
    assert settings.max_run_usd is None
    assert settings.max_calls is None
    assert settings.max_concurrency == 4


def test_settings_reads_budget_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("PARETOGUARD_MAX_RUN_USD", "5.0")
    monkeypatch.setenv("PARETOGUARD_MAX_CALLS", "100")
    monkeypatch.setenv("PARETOGUARD_MAX_CONCURRENCY", "8")
    settings = Settings(_env_file=None)
    assert settings.max_run_usd == 5.0
    assert settings.max_calls == 100
    assert settings.max_concurrency == 8


def test_load_providers_example_file() -> None:
    providers = load_providers(CONFIGS_DIR / "providers.example.yaml")
    names = {p.name for p in providers}
    assert "mock" in names
    mock = next(p for p in providers if p.name == "mock")
    assert mock.api_key_env_var is None


def test_load_models_example_file() -> None:
    models = load_models(CONFIGS_DIR / "models.example.yaml")
    names = {m.name for m in models}
    assert {"mock-strong", "mock-cheap"} <= names


def test_pricing_table_from_example_file() -> None:
    table = PricingTable.from_yaml(CONFIGS_DIR / "pricing.example.yaml")
    price = table.price_for("mock", "mock-strong")
    assert price is not None
    assert price.input_price_per_million_usd == 0.0


def test_pricing_table_returns_none_for_unknown_pair() -> None:
    table = PricingTable(entries=[])
    assert table.price_for("mock", "unknown-model") is None


def test_pricing_table_selects_most_recent_effective_date() -> None:
    table = PricingTable(
        entries=[
            PricingEntry(
                provider="p",
                model="m",
                input_price_per_million_usd=1.0,
                output_price_per_million_usd=2.0,
                version="old",
                effective_date=date(2025, 1, 1),
            ),
            PricingEntry(
                provider="p",
                model="m",
                input_price_per_million_usd=3.0,
                output_price_per_million_usd=4.0,
                version="new",
                effective_date=date(2026, 1, 1),
            ),
        ]
    )
    price = table.price_for("p", "m")
    assert price is not None
    assert price.version == "new"
