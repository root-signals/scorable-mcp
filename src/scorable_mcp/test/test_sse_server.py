"""Integration tests for the SSEMCPServer module using a live server."""

import json
import logging
from typing import Any
from unittest.mock import patch

import pytest

from scorable_mcp.root_api_client import (
    ResponseValidationError,
    ScorableEvaluatorRepository,
)
from scorable_mcp.schema import EvaluationRequest
from scorable_mcp.settings import settings

pytestmark = [
    pytest.mark.skipif(
        settings.scorable_api_key.get_secret_value() == "",
        reason="SCORABLE_API_KEY environment variable not set or empty",
    ),
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="session"),
]

logger = logging.getLogger("scorable_mcp_tests")


@pytest.mark.asyncio
async def test_server_initialization(mcp_server: Any) -> None:
    """Test MCP server initialization."""
    assert mcp_server.evaluator_service is not None
    logger.info("MCP Server initialized successfully")


@pytest.mark.asyncio
async def test_list_tools(mcp_server: Any) -> None:
    """Test the list_tools method."""
    tools = await mcp_server.list_tools()
    assert len(tools) >= 3, f"Expected at least 3 tools, found {len(tools)}"

    tool_dict = {tool.name: tool for tool in tools}

    assert "list_evaluators" in tool_dict, "list_evaluators tool not found"
    assert "run_evaluation" in tool_dict, "run_evaluation tool not found"
    assert "run_evaluation_by_name" in tool_dict, "run_evaluation_by_name tool not found"
    assert "run_coding_policy_adherence" in tool_dict, "run_coding_policy_adherence tool not found"

    for tool in tools:
        assert hasattr(tool, "name"), f"Tool missing name: {tool}"
        assert hasattr(tool, "description"), f"Tool missing description: {tool.name}"
        assert hasattr(tool, "inputSchema"), f"Tool missing inputSchema: {tool.name}"

    logger.info(f"Found {len(tools)} tools: {[tool.name for tool in tools]}")


@pytest.mark.asyncio
async def test_call_tool_list_evaluators__basic_api_response_includes_expected_fields(
    mcp_server: Any,
) -> None:
    """Test basic functionality of the list_evaluators tool."""
    result = await mcp_server.call_tool("list_evaluators", {})

    assert len(result) == 1, "Expected single result content"
    assert result[0].type == "text", "Expected text content"

    response_data = json.loads(result[0].text)
    assert "evaluators" in response_data, "Response missing evaluators list"
    assert len(response_data["evaluators"]) > 0, "No evaluators found"
    logger.info(f"Found {len(response_data['evaluators'])} evaluators")


@pytest.mark.asyncio
async def test_call_tool_list_judges__basic_api_response_includes_expected_fields(
    mcp_server: Any,
) -> None:
    """Test basic functionality of the list_judges tool."""
    result = await mcp_server.call_tool("list_judges", {})

    assert len(result) == 1, "Expected single result content"
    assert result[0].type == "text", "Expected text content"

    response_data = json.loads(result[0].text)
    assert "judges" in response_data, "Response missing judges list"
    assert len(response_data["judges"]) > 0, "No judges found"

    logger.info(f"Found {len(response_data['judges'])} judges")


@pytest.mark.asyncio
async def test_call_tool_list_evaluators__returns_newest_evaluators_first_by_default(
    mcp_server: Any,
) -> None:
    """Test that evaluators are sorted by created_at date in descending order (newest first)."""
    result = await mcp_server.call_tool("list_evaluators", {})
    response_data = json.loads(result[0].text)

    assert "evaluators" in response_data, "Response missing evaluators list"
    evaluators = response_data["evaluators"]

    assert len(evaluators) > 2, "API should return at least native evaluators, which is more than 2"

    for i in range(len(evaluators) - 1):
        current_date = evaluators[i].get("created_at", "")
        next_date = evaluators[i + 1].get("created_at", "")

        if not current_date or not next_date:
            continue

        assert current_date >= next_date, (
            f"Evaluators not sorted by created_at in descending order. "
            f"Found {current_date} before {next_date}"
        )

    logger.info("Verified evaluators are sorted with newest first")


@pytest.mark.asyncio
async def test_call_tool_run_evaluation(mcp_server: Any) -> None:
    """Test calling the run_evaluation tool."""
    list_result = await mcp_server.call_tool("list_evaluators", {})
    evaluators_data = json.loads(list_result[0].text)

    standard_evaluator = next(
        (e for e in evaluators_data["evaluators"] if e.get("name") == "Clarity"),
        next(
            (e for e in evaluators_data["evaluators"] if not e.get("requires_contexts", False)),
            None,
        ),
    )

    assert standard_evaluator is not None, "No standard evaluator found"

    logger.info(f"Using evaluator: {standard_evaluator['name']}")

    arguments = {
        "evaluator_id": standard_evaluator["id"],
        "request": "What is the capital of France?",
        "response": "The capital of France is Paris, which is known as the City of Light.",
    }

    result = await mcp_server.call_tool("run_evaluation", arguments)

    assert len(result) == 1, "Expected single result content"
    assert result[0].type == "text", "Expected text content"

    response_data = json.loads(result[0].text)
    assert "score" in response_data, "Response missing score"
    assert "justification" in response_data, "Response missing justification"

    logger.info(f"Evaluation completed with score: {response_data['score']}")


@pytest.mark.asyncio
async def test_call_tool_run_evaluation_by_name(mcp_server: Any) -> None:
    """Test calling the run_evaluation_by_name tool."""
    list_result = await mcp_server.call_tool("list_evaluators", {})
    evaluators_data = json.loads(list_result[0].text)

    standard_evaluator = next(
        (e for e in evaluators_data["evaluators"] if e.get("name") == "Clarity"),
        next(
            (e for e in evaluators_data["evaluators"] if not e.get("requires_contexts", False)),
            None,
        ),
    )

    assert standard_evaluator is not None, "No standard evaluator found"

    logger.info(f"Using evaluator by name: {standard_evaluator['name']}")

    arguments = {
        "evaluator_name": standard_evaluator["name"],
        "request": "What is the capital of France?",
        "response": "The capital of France is Paris, which is known as the City of Light.",
    }

    result = await mcp_server.call_tool("run_evaluation_by_name", arguments)

    response_data = json.loads(result[0].text)
    assert "error" not in response_data, f"Expected no error, got {response_data['error']}"

    assert len(result) == 1, "Expected single result content"
    assert result[0].type == "text", "Expected text content"

    assert "score" in response_data, "Response missing score"
    assert "justification" in response_data, "Response missing justification"

    logger.info(f"Evaluation by name completed with score: {response_data['score']}")


@pytest.mark.asyncio
async def test_call_tool_run_rag_evaluation(mcp_server: Any) -> None:
    """Test calling the run_evaluation tool with contexts."""
    list_result = await mcp_server.call_tool("list_evaluators", {})
    evaluators_data = json.loads(list_result[0].text)

    rag_evaluator = next(
        (e for e in evaluators_data["evaluators"] if e.get("name") == "Faithfulness"),
        next(
            (e for e in evaluators_data["evaluators"] if e.get("requires_contexts") is True), None
        ),
    )

    assert rag_evaluator is not None, "No RAG evaluator found"

    logger.info(f"Using evaluator: {rag_evaluator['name']}")

    arguments = {
        "evaluator_id": rag_evaluator["id"],
        "request": "What is the capital of France?",
        "response": "The capital of France is Paris, which is known as the City of Light.",
        "contexts": [
            "Paris is the capital and most populous city of France. It is located on the Seine River.",
            "France is a country in Western Europe with several overseas territories and regions.",
        ],
    }

    result = await mcp_server.call_tool("run_evaluation", arguments)

    assert len(result) == 1, "Expected single result content"
    assert result[0].type == "text", "Expected text content"

    response_data = json.loads(result[0].text)
    assert "score" in response_data, "Response missing score"
    assert "justification" in response_data, "Response missing justification"

    logger.info(f"RAG evaluation completed with score: {response_data['score']}")


@pytest.mark.asyncio
async def test_call_tool_run_rag_evaluation_by_name(mcp_server: Any) -> None:
    """Test calling the run_evaluation_by_name tool with contexts."""
    list_result = await mcp_server.call_tool("list_evaluators", {})
    evaluators_data = json.loads(list_result[0].text)

    rag_evaluator = next(
        (e for e in evaluators_data["evaluators"] if e.get("name") == "Faithfulness"),
        next(
            (e for e in evaluators_data["evaluators"] if e.get("requires_contexts") is True), None
        ),
    )

    assert rag_evaluator is not None, "No RAG evaluator found"

    logger.info(f"Using evaluator by name: {rag_evaluator['name']}")

    arguments = {
        "evaluator_name": rag_evaluator["name"],
        "request": "What is the capital of France?",
        "response": "The capital of France is Paris, which is known as the City of Light.",
        "contexts": [
            "Paris is the capital and most populous city of France. It is located on the Seine River.",
            "France is a country in Western Europe with several overseas territories and regions.",
        ],
    }

    result = await mcp_server.call_tool("run_evaluation_by_name", arguments)

    assert len(result) == 1, "Expected single result content"
    assert result[0].type == "text", "Expected text content"

    response_data = json.loads(result[0].text)
    assert "error" not in response_data, f"Expected no error, got {response_data.get('error')}"
    assert "score" in response_data, "Response missing score"
    assert "justification" in response_data, "Response missing justification"

    logger.info(f"RAG evaluation by name completed with score: {response_data['score']}")


@pytest.mark.asyncio
async def test_call_unknown_tool(mcp_server: Any) -> None:
    """Test calling an unknown tool."""
    result = await mcp_server.call_tool("unknown_tool", {})

    assert len(result) == 1, "Expected single result content"
    assert result[0].type == "text", "Expected text content"

    response_data = json.loads(result[0].text)
    assert "error" in response_data, "Response missing error message"
    assert "Unknown tool" in response_data["error"], "Unexpected error message"

    logger.info("Unknown tool test passed with expected error")


@pytest.mark.asyncio
async def test_run_evaluation_validation_error(mcp_server: Any) -> None:
    """Test validation error in run_evaluation."""
    result = await mcp_server.call_tool("run_evaluation", {"evaluator_id": "some_id"})

    response_data = json.loads(result[0].text)
    assert "error" in response_data, "Response missing error message"

    logger.info(f"Validation error test passed with error: {response_data['error']}")


@pytest.mark.asyncio
async def test_run_rag_evaluation_missing_context(mcp_server: Any) -> None:
    """Test calling run_evaluation with missing contexts."""
    list_result = await mcp_server.call_tool("list_evaluators", {})
    evaluators_data = json.loads(list_result[0].text)

    rag_evaluators = [
        e
        for e in evaluators_data["evaluators"]
        if any(
            kw in e.get("name", "").lower()
            for kw in ["faithfulness", "context", "rag", "relevance"]
        )
    ]

    rag_evaluator = next(iter(rag_evaluators), None)

    assert rag_evaluator is not None, "No RAG evaluator found"

    arguments = {
        "evaluator_id": rag_evaluator["id"],
        "request": "Test request",
        "response": "Test response",
        "contexts": [],
    }

    result = await mcp_server.call_tool("run_evaluation", arguments)
    response_data = json.loads(result[0].text)

    if "error" in response_data:
        logger.info(f"Empty contexts test produced error as expected: {response_data['error']}")
    else:
        logger.info("Empty contexts were accepted by the evaluator")


@pytest.mark.asyncio
async def test_sse_server_schema_evolution__handles_new_fields_gracefully() -> None:
    """Test that our models handle new fields in API responses gracefully."""
    with patch.object(ScorableEvaluatorRepository, "_make_request") as mock_request:
        mock_request.return_value = {
            "result": {
                "evaluator_name": "Test Evaluator",
                "score": 0.95,
                "justification": "Good response",
                "new_field_from_api": "This field doesn't exist in our schema",
                "another_new_field": {"nested": "value", "that": ["should", "be", "ignored"]},
            }
        }

        client = ScorableEvaluatorRepository()
        result = await client.run_evaluator(
            evaluator_id="test-id", request="Test request", response="Test response"
        )

        assert result.evaluator_name == "Test Evaluator"
        assert result.score == 0.95
        assert result.justification == "Good response"

        assert not hasattr(result, "new_field_from_api")
        assert not hasattr(result, "another_new_field")


@pytest.mark.asyncio
async def test_root_client_schema_compatibility__detects_api_schema_changes() -> None:
    """Test that our schema models detect changes in the API response format."""
    with patch.object(ScorableEvaluatorRepository, "_make_request") as mock_request:
        mock_request.return_value = {
            "result": {
                "score": 0.9,
                "justification": "Some justification",
            }
        }

        client = ScorableEvaluatorRepository()

        with pytest.raises(ResponseValidationError) as excinfo:
            await client.run_evaluator(
                evaluator_id="test-id", request="Test request", response="Test response"
            )

        error_message = str(excinfo.value)
        assert "Invalid evaluation response format" in error_message, (
            "Expected validation error message"
        )
        assert "evaluator_name" in error_message.lower(), "Error should reference the missing field"

        mock_request.return_value = {
            "result": {
                "evaluator_name": "Test Evaluator",
                "justification": "Some justification",
            }
        }

        with pytest.raises(ResponseValidationError) as excinfo:
            await client.run_evaluator(
                evaluator_id="test-id", request="Test request", response="Test response"
            )

        error_message = str(excinfo.value)
        assert "Invalid evaluation response format" in error_message, (
            "Expected validation error message"
        )
        assert "score" in error_message.lower(), "Error should reference the missing field"

        mock_request.return_value = {}

        with pytest.raises(ResponseValidationError) as excinfo:
            await client.run_evaluator(
                evaluator_id="test-id", request="Test request", response="Test response"
            )


@pytest.mark.asyncio
async def test_sse_server_request_validation__detects_extra_field_errors() -> None:
    """Test that request validation raises specific ValidationError instances for extra fields.

    This test verifies that we get proper Pydantic ValidationError objects
    with the expected error details when extra fields are provided.
    """

    # Extra fields should be silently ignored in the new domain-level models
    model_instance = EvaluationRequest(
        evaluator_id="test-id",
        request="Test request",
        response="Test response",
        unknown_field="This will be ignored",
    )

    assert not hasattr(model_instance, "unknown_field"), "Unexpected extra field was not ignored"

    request = EvaluationRequest(
        evaluator_id="test-id", request="Test request", response="Test response"
    )
    assert request.evaluator_id == "test-id", "evaluator_id not set correctly"
    assert request.request == "Test request", "request not set correctly"
    assert request.response == "Test response", "response not set correctly"


@pytest.mark.asyncio
async def test_sse_server_unknown_tool_request__explicitly_allows_any_fields() -> None:
    """Test that UnknownToolRequest explicitly allows any fields via model_config.

    This special model is used for debugging purposes with unknown tools,
    so it needs to capture any arbitrary fields.
    """
    from scorable_mcp.schema import UnknownToolRequest

    assert UnknownToolRequest.model_config.get("extra") == "allow", (
        "UnknownToolRequest model_config should be set to allow extra fields"
    )

    arbitrary_fields = {
        "any_field": "value",
        "another_field": 123,
        "nested_field": {"key": "value", "list": [1, 2, 3]},
        "list_field": ["a", "b", "c"],
    }

    request = UnknownToolRequest(**arbitrary_fields)
    result = request.model_dump()

    for key, value in arbitrary_fields.items():
        assert key in result, f"Field {key} not found in model_dump()"
        assert result[key] == value, f"Field {key} has wrong value in model_dump()"

    empty_request = UnknownToolRequest()
    assert isinstance(empty_request, UnknownToolRequest), (
        "Empty request should be valid UnknownToolRequest instance"
    )


@pytest.mark.asyncio
async def test_call_tool_run_judge(mcp_server: Any) -> None:
    """Test calling the run_judge tool."""
    list_result = await mcp_server.call_tool("list_judges", {})
    judges_data = json.loads(list_result[0].text)

    judge = next(iter(judges_data["judges"]), None)

    assert judge is not None, "No judge found"

    logger.info(f"Using judge: {judge['name']}")

    arguments = {
        "judge_id": judge["id"],
        "judge_name": judge["name"],
        "request": "What is the capital of France?",
        "response": "The capital of France is Paris, which is known as the City of Light.",
    }

    result = await mcp_server.call_tool("run_judge", arguments)

    assert len(result) == 1, "Expected single result content"
    assert result[0].type == "text", "Expected text content"

    response_data = json.loads(result[0].text)
    assert "evaluator_results" in response_data, "Response missing evaluator_results"
    assert len(response_data["evaluator_results"]) > 0, "No evaluator results in response"
    assert "score" in response_data["evaluator_results"][0], "Response missing score"
    assert "justification" in response_data["evaluator_results"][0], (
        "Response missing justification"
    )

    logger.info(f"Judge completed with score: {response_data['evaluator_results'][0]['score']}")
