"""long_context_retrieval_v1: find a target "needle" fact among synthetic
distractor sentences, at controllable difficulty (distractor count).

Documents are synthetic — generated deterministically from a seed, not sourced
from any real or copyrighted text.
"""

import random

from paretoguard.core.models import Message, Role
from paretoguard.evals.models import EvalCase, EvalSuite, GraderConfig, GraderKind, GroundTruth

VERSION = "1.0.0"
NAME = "long_context_retrieval_v1"

# distractor count per difficulty level — the context-length/difficulty knob.
DIFFICULTIES: dict[str, int] = {"short": 10, "medium": 30, "long": 60}

_CODE_PREFIXES = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT"]
_ATTRIBUTES = [
    "was founded in",
    "is headquartered in",
    "specializes in",
    "was renamed to",
    "reported revenue of",
]
_VALUES = [
    "1998",
    "Berlin",
    "logistics",
    "Vantage Corp",
    "2004",
    "Tokyo",
    "finance",
    "$4.2M",
    "2011",
    "Austin",
]


def _generate_document(rng: random.Random, num_distractors: int) -> tuple[str, str]:
    code = f"{rng.choice(_CODE_PREFIXES)}-{rng.randint(1000, 9999)}"
    facts = []
    for _ in range(num_distractors):
        entity = f"Entity-{rng.randint(100, 999)}"
        attribute = rng.choice(_ATTRIBUTES)
        value = rng.choice(_VALUES)
        facts.append(f"{entity} {attribute} {value}.")

    needle_position = rng.randint(0, len(facts))
    facts.insert(needle_position, f"The secret access code for this session is {code}.")
    return " ".join(facts), code


def build_suite(seed: int = 42, cases_per_difficulty: int = 5) -> EvalSuite:
    """Deterministically generates retrieval cases across every entry in
    `DIFFICULTIES`, `cases_per_difficulty` cases each."""
    rng = random.Random(seed)
    cases = []
    index = 0
    for level, num_distractors in DIFFICULTIES.items():
        for _ in range(cases_per_difficulty):
            document, code = _generate_document(rng, num_distractors)
            prompt = (
                "Read the following document and answer the question using only "
                f"information in it.\n\nDocument:\n{document}\n\n"
                "Question: What is the secret access code mentioned in the document?"
            )
            cases.append(
                EvalCase(
                    case_id=f"{NAME}-{level}-{index:03d}",
                    messages=[Message(role=Role.USER, content=prompt)],
                    grader=GraderConfig(kind=GraderKind.SUBSTRING_PRESENCE),
                    ground_truth=GroundTruth(expected_substrings=[code]),
                    tags=["long_context", level],
                    metadata={
                        "mock_scenario": "success",
                        "mock_answer_text": f"The secret access code is {code}.",
                    },
                )
            )
            index += 1
    return EvalSuite(
        name=NAME,
        version=VERSION,
        description=(
            "Locate a target fact among synthetic distractor sentences at "
            "controllable difficulty (short/medium/long), graded by exact "
            "presence of the expected code in the answer."
        ),
        seed=seed,
        cases=cases,
    )
