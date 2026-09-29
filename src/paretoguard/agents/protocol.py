"""Typed contracts for the agent simulator: the deterministic-tool boundary.

A model-generated tool call is untrusted input. `Tool.validate_arguments`
validates it against the tool's own Pydantic `args_model` *before* `run` ever
sees it — the executor (`paretoguard.agents.executor`) never calls `run`
directly on raw model output.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from paretoguard.core.models import ToolResult, ToolSpec


class ToolArgumentError(ValueError):
    """Raised by `Tool.validate_arguments` when a model-supplied argument
    dict fails Pydantic validation against `args_model`. Caught by the
    executor and turned into a `TerminationReason.INVALID_TOOL_ARGUMENTS`
    outcome — never propagated as an unhandled exception."""


class Tool[ArgsT: BaseModel](ABC):
    """A deterministic tool: same arguments in, same `ToolResult` out, every
    time. No network access, no shell, no `eval`, no unrestricted filesystem
    access, no real customer data — every built-in tool
    (`paretoguard.agents.tools`) operates only over fixed in-memory fixture
    data or pure computation.

    Generic over its own `args_model` type (`ArgsT`) so each concrete tool's
    `run(self, args: MyArgs)` is a valid override, not an LSP violation —
    `validate_arguments` returns `ArgsT`, matching what `run` expects.
    """

    name: str
    description: str
    args_model: type[ArgsT]

    def spec(self) -> ToolSpec:
        """The normalized `ToolSpec` advertised to a provider, derived from
        `args_model`'s own JSON Schema — one source of truth for what a
        model is told it can pass and what `validate_arguments` will accept.
        """
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters_schema=self.args_model.model_json_schema(),
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> ArgsT:
        try:
            return self.args_model.model_validate(arguments)
        except Exception as exc:
            raise ToolArgumentError(str(exc)) from exc

    @abstractmethod
    def run(self, args: ArgsT) -> ToolResult:
        """Executes against already-validated arguments. `tool_call_id` is
        left blank here — the executor fills it in from the model's actual
        tool-call id, which this method has no reason to know."""
        raise NotImplementedError
