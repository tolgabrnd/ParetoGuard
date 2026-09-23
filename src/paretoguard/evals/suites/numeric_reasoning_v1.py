"""numeric_reasoning_v1: multi-step arithmetic word problems with irrelevant
context (distractor sentences), deterministic answers.

Three problem templates are used specifically so a naive "grab the first/only
number in the prompt" heuristic doesn't trivially solve the suite — each
template combines the same three generated numbers with a different multi-step
operation order, and a distractor sentence unrelated to the arithmetic is always
present.
"""

import random

from paretoguard.core.models import Message, Role
from paretoguard.evals.models import EvalCase, EvalSuite, GraderConfig, GraderKind, GroundTruth

VERSION = "1.0.0"
NAME = "numeric_reasoning_v1"

_NAMES = ["Alice", "Bob", "Priya", "Wei", "Fatima", "Diego"]
_COLORS = ["red", "blue", "green", "purple", "orange"]


def _distractor(rng: random.Random) -> str:
    name = rng.choice(_NAMES)
    color = rng.choice(_COLORS)
    return f"{name}'s favorite color is {color}, which has no bearing on this problem."


def _generate_case(rng: random.Random) -> tuple[str, float, str]:
    a = rng.randint(2, 20)
    b = rng.randint(2, 20)
    c = rng.randint(2, 10)
    distractor = _distractor(rng)
    template = rng.choice(["multiply_add", "add_multiply", "subtract_multiply"])

    if template == "multiply_add":
        prompt = (
            f"A store sells {a} boxes, each containing {b} items. {distractor} "
            f"After the sale, {c} more items were added to inventory. "
            "How many items are there in total?"
        )
        answer = float(a * b + c)
    elif template == "add_multiply":
        prompt = (
            f"A warehouse has {a} crates and receives {b} more crates. {distractor} "
            f"Each crate holds {c} units. How many units are there in total?"
        )
        answer = float((a + b) * c)
    else:
        prompt = (
            f"A factory produced {a} units per day for {b} days. {distractor} "
            f"Of those, {c} units per day were defective and discarded. "
            "How many good units were produced in total?"
        )
        answer = float((a - c) * b)

    return prompt, answer, template


def build_suite(seed: int = 42, num_cases: int = 20) -> EvalSuite:
    """Deterministically generates `num_cases` multi-step arithmetic word problems."""
    rng = random.Random(seed)
    cases = []
    for i in range(num_cases):
        prompt, answer, template = _generate_case(rng)
        cases.append(
            EvalCase(
                case_id=f"{NAME}-{i:03d}",
                messages=[Message(role=Role.USER, content=prompt)],
                grader=GraderConfig(kind=GraderKind.NUMERIC, tolerance=1e-6),
                ground_truth=GroundTruth(number=answer),
                tags=["numeric_reasoning", "multi_step", template],
                metadata={
                    "mock_scenario": "success",
                    "mock_answer_text": f"Working through the steps, the final answer is {answer:g}.",
                    "template_id": f"{NAME}:{template}",
                },
            )
        )
    return EvalSuite(
        name=NAME,
        version=VERSION,
        description=(
            "Multi-step arithmetic word problems with an irrelevant distractor "
            "sentence, graded by exact numeric match within tolerance."
        ),
        seed=seed,
        cases=cases,
    )
