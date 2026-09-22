"""Provider and model configuration types.

These describe *what providers/models exist and how to reach them*, not pricing
(see `cost.py`) and not which one is "best" — that is a routing-time decision made
from data, never a hard-coded assumption here.
"""

from pydantic import BaseModel, Field

from paretoguard.core.models.common import ProviderKind


class ProviderSpec(BaseModel):
    """A configured LLM provider connection."""

    name: str
    kind: ProviderKind
    base_url: str | None = None
    api_key_env_var: str | None = Field(
        default=None,
        description="Name of the environment variable holding the API key. Never the key itself.",
    )
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)


class ModelSpec(BaseModel):
    """A model made available through a configured provider."""

    name: str
    provider: str = Field(description="Must match a ProviderSpec.name")
    context_window: int = Field(gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    supports_tools: bool = False
    supports_structured_output: bool = False
    capabilities: list[str] = Field(default_factory=list)
