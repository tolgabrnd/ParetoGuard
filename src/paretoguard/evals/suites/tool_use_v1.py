"""tool_use_v1: single-turn tool-selection and argument-validity evaluation.

Each case describes a scenario that should prompt the model to call exactly one
of three safe, read-only, deterministic tools (calculator, inventory_lookup,
order_lookup) with specific arguments; grading uses TOOL_TRAJECTORY, which
compares the ordered list of tool calls a single InferenceResponse returned.

Scope note: this suite does not execute tool calls against a stateful simulator
or grade a resulting "final state" — that needs the multi-step tool-execution
loop `paretoguard.agents` will provide (Phase E, not yet implemented). See
paretoguard.evals.graders.tool_use for the full rationale. No tool here performs
unrestricted execution: each is a pure lookup/computation over fixed schemas.
"""

import random

from paretoguard.core.models import Message, Role, ToolSpec
from paretoguard.evals.models import (
    EvalCase,
    EvalSuite,
    ExpectedToolCall,
    GraderConfig,
    GraderKind,
    GroundTruth,
)

VERSION = "1.0.0"
NAME = "tool_use_v1"

CALCULATOR_TOOL = ToolSpec(
    name="calculator",
    description="Evaluates a basic arithmetic expression and returns the numeric result.",
    parameters_schema={
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
)
INVENTORY_LOOKUP_TOOL = ToolSpec(
    name="inventory_lookup",
    description="Looks up the current stock quantity for a given SKU.",
    parameters_schema={
        "type": "object",
        "properties": {"sku": {"type": "string"}},
        "required": ["sku"],
    },
)
ORDER_LOOKUP_TOOL = ToolSpec(
    name="order_lookup",
    description="Looks up the status of a customer order by order ID.",
    parameters_schema={
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
)
TOOLS = [CALCULATOR_TOOL, INVENTORY_LOOKUP_TOOL, ORDER_LOOKUP_TOOL]


def _generate_case(rng: random.Random) -> tuple[str, ExpectedToolCall]:
    kind = rng.choice(["calculator", "inventory", "order"])

    if kind == "calculator":
        a, b = rng.randint(1, 100), rng.randint(1, 100)
        op = rng.choice(["+", "-", "*"])
        prompt = f"What is {a} {op} {b}? Use the calculator tool to compute it."
        expected = ExpectedToolCall(name="calculator", arguments={"expression": f"{a} {op} {b}"})
    elif kind == "inventory":
        sku = f"SKU-{rng.randint(1000, 9999)}"
        prompt = f"Check how many units of {sku} are currently in stock."
        expected = ExpectedToolCall(name="inventory_lookup", arguments={"sku": sku})
    else:
        order_id = f"ORD-{rng.randint(100000, 999999)}"
        prompt = f"What is the current status of order {order_id}?"
        expected = ExpectedToolCall(name="order_lookup", arguments={"order_id": order_id})

    return prompt, expected


def build_suite(seed: int = 42, num_cases: int = 20) -> EvalSuite:
    """Deterministically generates `num_cases` single-tool-call scenarios."""
    rng = random.Random(seed)
    cases = []
    for i in range(num_cases):
        prompt, expected = _generate_case(rng)
        cases.append(
            EvalCase(
                case_id=f"{NAME}-{i:03d}",
                messages=[Message(role=Role.USER, content=prompt)],
                tools=TOOLS,
                grader=GraderConfig(kind=GraderKind.TOOL_TRAJECTORY),
                ground_truth=GroundTruth(expected_tool_calls=[expected]),
                tags=["tool_use", expected.name],
                metadata={
                    "mock_scenario": "success",
                    "mock_tool_calls": [{"name": expected.name, "arguments": expected.arguments}],
                    "template_id": f"{NAME}:{expected.name}",
                },
            )
        )
    return EvalSuite(
        name=NAME,
        version=VERSION,
        description=(
            "Single-turn tool-selection and argument-validity scenarios over three "
            "safe simulated tools, graded by exact tool-call trajectory match."
        ),
        seed=seed,
        cases=cases,
    )
