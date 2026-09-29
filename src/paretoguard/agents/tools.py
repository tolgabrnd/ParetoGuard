"""Safe, deterministic tools for the agent simulator.

Every tool here operates only over fixed in-memory fixture data
(`_INVENTORY`/`_ORDERS`/`_SHIPMENTS`/`_DOCUMENTS` below — synthetic, never
real customer data) or pure arithmetic. No network access, no shell, no
`eval`, no unrestricted filesystem access. `CalculatorTool` parses and
evaluates expressions through Python's `ast` module with an explicit
node-type allowlist rather than `eval`, so a malicious/malformed expression
can only ever raise `ValueError`, never execute arbitrary code.

The fixtures form one coherent, deterministic world so multi-hop
`structured_agent_v1` tasks (Commit 29) can chain calls meaningfully: an
order references a shipment id, a document references an order id.
"""

import ast
import operator
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from paretoguard.agents.protocol import Tool
from paretoguard.core.models import ToolResult

# -- fixture data (synthetic, deterministic, no real customer data) --------

_INVENTORY: dict[str, int] = {
    "SKU-1001": 42,
    "SKU-1002": 0,
    "SKU-1003": 17,
}

_ORDERS: dict[str, dict[str, Any]] = {
    "ORD-1001": {
        "customer": "A. Rivera",
        "sku": "SKU-1001",
        "status": "shipped",
        "shipment_id": "SHIP-5001",
    },
    "ORD-1002": {
        "customer": "J. Okafor",
        "sku": "SKU-1003",
        "status": "processing",
        "shipment_id": None,
    },
}

_SHIPMENTS: dict[str, dict[str, Any]] = {
    "SHIP-5001": {"carrier": "FastShip", "status": "in_transit", "eta_days": 3},
}

_DOCUMENTS: dict[str, dict[str, Any]] = {
    "DOC-INV-1001": {"type": "invoice", "order_id": "ORD-1001", "total": 129.99},
}


# -- calculator ---------------------------------------------------------

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {ast.USub: operator.neg}


def _safe_eval(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


class CalculatorArgs(BaseModel):
    expression: str = Field(description="A basic arithmetic expression, e.g. '3 * (4 + 2)'.")


class CalculatorTool(Tool[CalculatorArgs]):
    name = "calculator"
    description = "Evaluates a basic arithmetic expression and returns the numeric result."
    args_model = CalculatorArgs

    def run(self, args: CalculatorArgs) -> ToolResult:
        try:
            tree = ast.parse(args.expression, mode="eval")
            value = _safe_eval(tree.body)
        except Exception as exc:
            return ToolResult(
                tool_call_id="", is_error=True, error_message=f"invalid expression: {exc}"
            )
        return ToolResult(tool_call_id="", output=value)


# -- inventory_lookup -----------------------------------------------------


class InventoryLookupArgs(BaseModel):
    sku: str


class InventoryLookupTool(Tool[InventoryLookupArgs]):
    name = "inventory_lookup"
    description = "Looks up the current stock quantity for a given SKU."
    args_model = InventoryLookupArgs

    def run(self, args: InventoryLookupArgs) -> ToolResult:
        if args.sku not in _INVENTORY:
            return ToolResult(
                tool_call_id="", is_error=True, error_message=f"unknown SKU: {args.sku!r}"
            )
        return ToolResult(
            tool_call_id="", output={"sku": args.sku, "quantity": _INVENTORY[args.sku]}
        )


# -- order_lookup -----------------------------------------------------------


class OrderLookupArgs(BaseModel):
    order_id: str


class OrderLookupTool(Tool[OrderLookupArgs]):
    name = "order_lookup"
    description = "Looks up the status, SKU, and shipment id of a customer order by order id."
    args_model = OrderLookupArgs

    def run(self, args: OrderLookupArgs) -> ToolResult:
        order = _ORDERS.get(args.order_id)
        if order is None:
            return ToolResult(
                tool_call_id="", is_error=True, error_message=f"unknown order: {args.order_id!r}"
            )
        return ToolResult(tool_call_id="", output={"order_id": args.order_id, **order})


# -- shipment_lookup ----------------------------------------------------------


class ShipmentLookupArgs(BaseModel):
    shipment_id: str


class ShipmentLookupTool(Tool[ShipmentLookupArgs]):
    name = "shipment_lookup"
    description = "Looks up the carrier, status, and ETA of a shipment by shipment id."
    args_model = ShipmentLookupArgs

    def run(self, args: ShipmentLookupArgs) -> ToolResult:
        shipment = _SHIPMENTS.get(args.shipment_id)
        if shipment is None:
            return ToolResult(
                tool_call_id="",
                is_error=True,
                error_message=f"unknown shipment: {args.shipment_id!r}",
            )
        return ToolResult(tool_call_id="", output={"shipment_id": args.shipment_id, **shipment})


# -- document_retrieval -------------------------------------------------------


class DocumentRetrievalArgs(BaseModel):
    document_id: str


class DocumentRetrievalTool(Tool[DocumentRetrievalArgs]):
    name = "document_retrieval"
    description = "Retrieves a stored document (e.g. an invoice) by document id."
    args_model = DocumentRetrievalArgs

    def run(self, args: DocumentRetrievalArgs) -> ToolResult:
        document = _DOCUMENTS.get(args.document_id)
        if document is None:
            return ToolResult(
                tool_call_id="",
                is_error=True,
                error_message=f"unknown document: {args.document_id!r}",
            )
        return ToolResult(tool_call_id="", output={"document_id": args.document_id, **document})


def default_tools() -> list[Tool[Any]]:
    """The standard tool set used by the built-in agent benchmarks
    (`paretoguard.evals.suites.structured_agent_v1`)."""
    return [
        CalculatorTool(),
        InventoryLookupTool(),
        OrderLookupTool(),
        ShipmentLookupTool(),
        DocumentRetrievalTool(),
    ]
