"""structured_agent_v1: multi-step tool-use agent tasks over the deterministic
fixture world `paretoguard.agents.tools` defines (`_INVENTORY`/`_ORDERS`/
`_SHIPMENTS`/`_DOCUMENTS`), completing the scope `tool_use_v1` explicitly left
for Phase E (see that module's docstring: "this suite does not execute tool
calls against a stateful simulator... that needs the multi-step tool-execution
loop `paretoguard.agents` will provide").

Unlike every other suite in this package, there is no LLM being approximated
by `MockProvider` here: reasoning over "which tool, in what order, with what
final answer" is exactly what a real agent policy would have to produce, and
scripting that deterministically is the only way to keep this suite offline
and reproducible (see CLAUDE.md: zero network access in tests/benchmarks).
`ScriptedAgentProvider` plays that role — a fixed (task_id -> response
sequence) script, not a model. Every result this suite produces is therefore
SIMULATION in the same sense `routing.reliability_simulation` is: it measures
the *executor's* handling of a scripted trajectory (tool errors, invalid
arguments, multi-hop chaining), not any real model's tool-use competence.

Two tasks (`invalid-args`, `invalid-tool-name`) deliberately script a
malformed step to exercise `TerminationReason.INVALID_TOOL_ARGUMENTS`/
`INVALID_TOOL` — their `ExpectedTrajectory` correctly predicts failure; they
are not suite bugs.
"""

from dataclasses import dataclass
from uuid import uuid4

from paretoguard.agents.simulator import AgentTask, ExpectedTrajectory
from paretoguard.core.models import (
    FinishReason,
    InferenceRequest,
    InferenceResponse,
    LatencyRecord,
    TokenUsage,
    ToolCall,
)
from paretoguard.providers.base import Provider

VERSION = "1.0.0"
NAME = "structured_agent_v1"


def _final(text: str) -> InferenceResponse:
    return InferenceResponse(
        request_id=uuid4(),
        provider="mock",
        model="m",
        output_text=text,
        finish_reason=FinishReason.STOP,
        token_usage=TokenUsage(input_tokens=4, output_tokens=4),
        latency=LatencyRecord(total_latency_ms=1.0),
    )


def _tool_call(name: str, arguments: dict[str, object]) -> InferenceResponse:
    return InferenceResponse(
        request_id=uuid4(),
        provider="mock",
        model="m",
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=[ToolCall(id=f"call-{name}", name=name, arguments=arguments)],
        token_usage=TokenUsage(input_tokens=4, output_tokens=2),
        latency=LatencyRecord(total_latency_ms=1.0),
    )


@dataclass(frozen=True)
class _TaskDefinition:
    task: AgentTask
    script: list[InferenceResponse]


def _task_definitions() -> list[_TaskDefinition]:
    return [
        _TaskDefinition(
            AgentTask(
                task_id=f"{NAME}-inventory-in-stock",
                user_prompt="How many units of SKU-1001 are in stock?",
                expected=ExpectedTrajectory(
                    expected_tool_sequence=["inventory_lookup"], expected_answer_substring="42"
                ),
            ),
            [
                _tool_call("inventory_lookup", {"sku": "SKU-1001"}),
                _final("There are 42 units of SKU-1001 in stock."),
            ],
        ),
        _TaskDefinition(
            AgentTask(
                task_id=f"{NAME}-inventory-out-of-stock",
                user_prompt="How many units of SKU-1002 are in stock?",
                expected=ExpectedTrajectory(
                    expected_tool_sequence=["inventory_lookup"],
                    expected_answer_substring="out of stock",
                ),
            ),
            [
                _tool_call("inventory_lookup", {"sku": "SKU-1002"}),
                _final("SKU-1002 is currently out of stock (0 units)."),
            ],
        ),
        _TaskDefinition(
            AgentTask(
                task_id=f"{NAME}-order-status",
                user_prompt="What is the status of order ORD-1002?",
                expected=ExpectedTrajectory(
                    expected_tool_sequence=["order_lookup"], expected_answer_substring="processing"
                ),
            ),
            [
                _tool_call("order_lookup", {"order_id": "ORD-1002"}),
                _final("Order ORD-1002 is currently processing."),
            ],
        ),
        _TaskDefinition(
            AgentTask(
                task_id=f"{NAME}-order-shipment-multihop",
                user_prompt="What's the shipment status for order ORD-1001?",
                expected=ExpectedTrajectory(
                    expected_tool_sequence=["order_lookup", "shipment_lookup"],
                    expected_answer_substring="in_transit",
                ),
            ),
            [
                _tool_call("order_lookup", {"order_id": "ORD-1001"}),
                _tool_call("shipment_lookup", {"shipment_id": "SHIP-5001"}),
                _final(
                    "Order ORD-1001's shipment (SHIP-5001) is in_transit via "
                    "FastShip, arriving in 3 days."
                ),
            ],
        ),
        _TaskDefinition(
            AgentTask(
                task_id=f"{NAME}-calculator",
                user_prompt="What is 12 * 7 + 5?",
                expected=ExpectedTrajectory(
                    expected_tool_sequence=["calculator"], expected_answer_substring="89"
                ),
            ),
            [
                _tool_call("calculator", {"expression": "12 * 7 + 5"}),
                _final("The result is 89."),
            ],
        ),
        _TaskDefinition(
            AgentTask(
                task_id=f"{NAME}-document-retrieval",
                user_prompt="What is the total on invoice DOC-INV-1001?",
                expected=ExpectedTrajectory(
                    expected_tool_sequence=["document_retrieval"],
                    expected_answer_substring="129.99",
                ),
            ),
            [
                _tool_call("document_retrieval", {"document_id": "DOC-INV-1001"}),
                _final("The total on invoice DOC-INV-1001 is $129.99."),
            ],
        ),
        _TaskDefinition(
            AgentTask(
                task_id=f"{NAME}-order-not-found-graceful",
                user_prompt="What is the status of order ORD-9999?",
                expected=ExpectedTrajectory(
                    expected_tool_sequence=["order_lookup"],
                    expected_answer_substring="couldn't find",
                ),
            ),
            [
                _tool_call("order_lookup", {"order_id": "ORD-9999"}),
                _final("I couldn't find order ORD-9999 in the system."),
            ],
        ),
        _TaskDefinition(
            AgentTask(
                task_id=f"{NAME}-invalid-args",
                user_prompt="Check stock for the item with no SKU given.",
                expected=ExpectedTrajectory(expected_tool_sequence=["inventory_lookup"]),
            ),
            # Missing the required `sku` field: Tool.validate_arguments raises
            # ToolArgumentError, so the executor terminates INVALID_TOOL_ARGUMENTS
            # before the second scripted response is ever reached. Grading
            # correctly predicts `succeeded=False` for this task by design.
            [_tool_call("inventory_lookup", {})],
        ),
        _TaskDefinition(
            AgentTask(
                task_id=f"{NAME}-invalid-tool-name",
                user_prompt="Restock SKU-1002 by 10 units.",
                expected=ExpectedTrajectory(expected_tool_sequence=["restock_item"]),
            ),
            # No `restock_item` tool exists (every built-in tool is read-only —
            # see agents.tools's module docstring): the executor terminates
            # INVALID_TOOL. Grading correctly predicts `succeeded=False`.
            [_tool_call("restock_item", {"sku": "SKU-1002", "quantity": 10})],
        ),
    ]


def build_tasks(seed: int = 0) -> list[AgentTask]:
    """Returns the fixed `structured_agent_v1` task set. `seed` is accepted
    for interface symmetry with the other suites' `build_suite(seed, ...)`
    but currently unused: every task here is hand-authored against the fixed
    fixture world, not procedurally generated, so there is nothing to
    reseed — kept as a parameter so callers don't need special-casing and so
    a future procedurally-generated extension of this suite can add
    seed-driven variation without an interface change."""
    return [definition.task for definition in _task_definitions()]


class ScriptedAgentProvider(Provider):
    """A fixed (task_id -> response sequence) script played back by call
    count within each task — the deterministic stand-in for "the model"
    this suite needs (see module docstring). Not reusable across a batch
    with repeated task ids beyond the scripted sequence length: the last
    scripted response repeats indefinitely past the end of its script,
    matching every other scripted-provider test helper in this repo."""

    def __init__(self, name: str = "mock") -> None:
        self.name = name
        self._scripts = {d.task.task_id: d.script for d in _task_definitions()}
        self._counts: dict[str, int] = {}

    async def complete(self, request: InferenceRequest) -> InferenceResponse:
        task_id = request.task_id or ""
        script = self._scripts.get(task_id)
        if not script:
            raise ValueError(f"ScriptedAgentProvider has no script for task_id={task_id!r}")
        count = self._counts.get(task_id, 0)
        response = script[min(count, len(script) - 1)]
        self._counts[task_id] = count + 1
        return response.model_copy(
            update={
                "request_id": request.request_id,
                "provider": request.provider,
                "model": request.model,
            }
        )
