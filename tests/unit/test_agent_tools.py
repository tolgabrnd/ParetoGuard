"""Unit tests for the deterministic agent tools."""

import pytest

from paretoguard.agents.protocol import ToolArgumentError
from paretoguard.agents.tools import (
    CalculatorTool,
    DocumentRetrievalTool,
    InventoryLookupTool,
    OrderLookupTool,
    ShipmentLookupTool,
    default_tools,
)


def test_default_tools_returns_all_five() -> None:
    tools = default_tools()
    assert {t.name for t in tools} == {
        "calculator",
        "inventory_lookup",
        "order_lookup",
        "shipment_lookup",
        "document_retrieval",
    }


def test_calculator_evaluates_basic_expression() -> None:
    tool = CalculatorTool()
    args = tool.validate_arguments({"expression": "3 * (4 + 2)"})
    result = tool.run(args)
    assert not result.is_error
    assert result.output == 18.0


def test_calculator_rejects_arbitrary_code_via_ast_allowlist() -> None:
    tool = CalculatorTool()
    args = tool.validate_arguments({"expression": "__import__('os').system('echo pwned')"})
    result = tool.run(args)
    assert result.is_error


def test_calculator_rejects_function_calls() -> None:
    tool = CalculatorTool()
    args = tool.validate_arguments({"expression": "abs(-5)"})
    result = tool.run(args)
    assert result.is_error


def test_tool_validate_arguments_rejects_missing_field() -> None:
    tool = CalculatorTool()
    with pytest.raises(ToolArgumentError):
        tool.validate_arguments({})


def test_inventory_lookup_known_and_unknown_sku() -> None:
    tool = InventoryLookupTool()
    known = tool.run(tool.validate_arguments({"sku": "SKU-1001"}))
    assert not known.is_error
    assert known.output == {"sku": "SKU-1001", "quantity": 42}

    unknown = tool.run(tool.validate_arguments({"sku": "SKU-9999"}))
    assert unknown.is_error


def test_order_lookup_chains_to_shipment_id() -> None:
    order_tool = OrderLookupTool()
    order_result = order_tool.run(order_tool.validate_arguments({"order_id": "ORD-1001"}))
    assert not order_result.is_error
    assert order_result.output["shipment_id"] == "SHIP-5001"

    shipment_tool = ShipmentLookupTool()
    shipment_result = shipment_tool.run(
        shipment_tool.validate_arguments({"shipment_id": order_result.output["shipment_id"]})
    )
    assert not shipment_result.is_error
    assert shipment_result.output["status"] == "in_transit"


def test_document_retrieval_known_and_unknown() -> None:
    tool = DocumentRetrievalTool()
    known = tool.run(tool.validate_arguments({"document_id": "DOC-INV-1001"}))
    assert not known.is_error
    assert known.output["order_id"] == "ORD-1001"

    unknown = tool.run(tool.validate_arguments({"document_id": "DOC-NOPE"}))
    assert unknown.is_error


def test_tool_spec_reflects_args_model_schema() -> None:
    tool = CalculatorTool()
    spec = tool.spec()
    assert spec.name == "calculator"
    assert "expression" in spec.parameters_schema.get("properties", {})
