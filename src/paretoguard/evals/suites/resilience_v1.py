"""resilience_v1: a fixed, trivially-answerable task load for exercising the
Phase E recovery layer, not the task-difficulty axis.

Every case here is designed so `MockProvider`'s default SUCCESS scenario
answers it correctly with certainty — the point of this suite is never "is
the model smart enough", it's "does the recovery layer keep the task
succeeding when the *provider* fails". Difficulty is deliberately held
constant; `paretoguard.chaos.scenarios.resilience_fault_levels` supplies the
actual independent variable (injected provider fault probability). See
`paretoguard.routing.resilience_benchmark` for the flagship comparison this
suite feeds (Commit 29) — every number that benchmark produces from this
suite is SIMULATION (see CLAUDE.md: never fabricate reliability numbers).
"""

import random

from paretoguard.core.models import Message, Role
from paretoguard.evals.models import EvalCase, EvalSuite, GraderConfig, GraderKind, GroundTruth

VERSION = "1.0.0"
NAME = "resilience_v1"

_TOPICS = [
    ("capital of France", "Paris"),
    ("capital of Japan", "Tokyo"),
    ("capital of Egypt", "Cairo"),
    ("capital of Canada", "Ottawa"),
    ("capital of Australia", "Canberra"),
    ("capital of Brazil", "Brasilia"),
    ("capital of Kenya", "Nairobi"),
    ("capital of Norway", "Oslo"),
    ("chemical symbol for gold", "Au"),
    ("chemical symbol for iron", "Fe"),
    ("number of continents on Earth", "seven"),
    ("largest planet in the solar system", "Jupiter"),
]


def build_suite(seed: int = 0, num_cases: int = 24) -> EvalSuite:
    """Deterministically generates `num_cases` single-turn factual-recall
    questions, cycling through `_TOPICS` (reshuffled per case index via
    `seed`) so every case still has a distinct `case_id` even when
    `num_cases > len(_TOPICS)`."""
    rng = random.Random(seed)
    order = list(range(len(_TOPICS)))
    cases = []
    for i in range(num_cases):
        if i % len(_TOPICS) == 0:
            rng.shuffle(order)
        question, answer = _TOPICS[order[i % len(_TOPICS)]]
        cases.append(
            EvalCase(
                case_id=f"{NAME}-{i:03d}",
                messages=[Message(role=Role.USER, content=f"What is the {question}?")],
                grader=GraderConfig(kind=GraderKind.NORMALIZED_TEXT_MATCH),
                ground_truth=GroundTruth(text=answer),
                tags=["resilience", "fixed_difficulty"],
                metadata={
                    "mock_scenario": "success",
                    "mock_answer_text": answer,
                },
            )
        )
    return EvalSuite(
        name=NAME,
        version=VERSION,
        description=(
            "Trivially-answerable single-turn factual-recall tasks with fixed "
            "difficulty, used as a fixed task load for the Phase E recovery-"
            "policy comparison (paretoguard.routing.resilience_benchmark) — "
            "failure in this suite comes only from injected chaos, never from "
            "task difficulty."
        ),
        seed=seed,
        cases=cases,
    )
