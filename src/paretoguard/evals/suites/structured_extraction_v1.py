"""structured_extraction_v1: extract structured fields from synthetic records.

Cases are generated deterministically from a seed — there are no static fixture
files, so a suite's cases are fully reproducible from (this module's version,
seed) alone. Records are synthetic invoices, not copied from any real dataset.
"""

import random

from paretoguard.core.models import Message, Role
from paretoguard.evals.models import EvalCase, EvalSuite, GraderConfig, GraderKind, GroundTruth

VERSION = "1.0.0"
NAME = "structured_extraction_v1"

_VENDORS = [
    "Acme Corp",
    "Globex LLC",
    "Initech",
    "Umbrella Supplies",
    "Wayne Industries",
    "Stark Materials",
]
_ITEMS = ["Widget", "Gadget", "Gizmo", "Sprocket", "Bracket", "Fastener"]

_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "invoice_number": {"type": "string"},
        "date": {"type": "string"},
        "item": {"type": "string"},
        "quantity": {"type": "integer"},
        "unit_price": {"type": "number"},
        "total": {"type": "number"},
    },
    "required": ["vendor", "invoice_number", "date", "item", "quantity", "unit_price", "total"],
    "additionalProperties": False,
}


def _generate_invoice(rng: random.Random) -> tuple[str, dict[str, object]]:
    vendor = rng.choice(_VENDORS)
    invoice_number = f"INV-{rng.randint(10000, 99999)}"
    date = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    item = rng.choice(_ITEMS)
    quantity = rng.randint(1, 50)
    unit_price = round(rng.uniform(5.0, 500.0), 2)
    total = round(quantity * unit_price, 2)

    text = (
        "INVOICE\n"
        f"Vendor: {vendor}\n"
        f"Invoice Number: {invoice_number}\n"
        f"Date: {date}\n"
        f"Line item: {quantity} x {item} @ ${unit_price:.2f} each\n"
        f"Total due: ${total:.2f}\n"
    )
    fields: dict[str, object] = {
        "vendor": vendor,
        "invoice_number": invoice_number,
        "date": date,
        "item": item,
        "quantity": quantity,
        "unit_price": unit_price,
        "total": total,
    }
    return text, fields


def build_suite(seed: int = 42, num_cases: int = 20) -> EvalSuite:
    """Deterministically generates `num_cases` synthetic-invoice extraction tasks."""
    rng = random.Random(seed)
    cases = []
    for i in range(num_cases):
        text, fields = _generate_invoice(rng)
        prompt = (
            "Extract the following fields from this invoice as a JSON object with "
            "exactly these keys: vendor, invoice_number, date, item, quantity, "
            "unit_price, total.\n\n" + text
        )
        cases.append(
            EvalCase(
                case_id=f"{NAME}-{i:03d}",
                messages=[Message(role=Role.USER, content=prompt)],
                structured_output_schema=_SCHEMA,
                grader=GraderConfig(kind=GraderKind.FIELD_SCORING),
                ground_truth=GroundTruth(expected_fields=fields),
                tags=["structured_extraction", "invoice"],
                metadata={
                    # MockProvider-recognized keys (harmlessly ignored by real
                    # providers): makes this case demonstrably solvable offline.
                    "mock_scenario": "exact_json",
                    "mock_json_answer": fields,
                },
            )
        )
    return EvalSuite(
        name=NAME,
        version=VERSION,
        description=(
            "Extract structured fields from synthetic invoice text into JSON, "
            "scored by field-level precision/recall/F1."
        ),
        seed=seed,
        cases=cases,
    )
