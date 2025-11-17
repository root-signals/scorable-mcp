"""Integration tests for the Scorable MCP Server using SSE transport."""

import logging
from typing import Any

import pytest

from scorable_mcp.client import ScorableMCPClient
from scorable_mcp.evaluator import EvaluatorService
from scorable_mcp.schema import (
    EvaluationRequest,
    EvaluationRequestByName,
    EvaluationResponse,
    EvaluatorInfo,
    EvaluatorsListResponse,
)
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
async def test_list_tools(compose_up_mcp_server: Any) -> None:
    """Test listing tools via SSE transport."""
    logger.info("Connecting to MCP server")
    client: ScorableMCPClient = ScorableMCPClient()

    try:
        await client.connect()

        tools: list[dict[str, Any]] = await client.list_tools()

        tool_names: set[str] = {tool["name"] for tool in tools}
        expected_tools: set[str] = {
            "list_evaluators",
            "run_evaluation",
            "run_coding_policy_adherence",
            "list_judges",
            "run_judge",
        }

        assert expected_tools.issubset(tool_names), f"Missing expected tools. Found: {tool_names}"
        logger.info(f"Found expected tools: {tool_names}")
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_list_evaluators(compose_up_mcp_server: Any) -> None:
    """Test listing evaluators via SSE transport."""
    logger.info("Connecting to MCP server")
    client: ScorableMCPClient = ScorableMCPClient()

    try:
        await client.connect()

        evaluators: list[dict[str, Any]] = await client.list_evaluators()

        assert len(evaluators) > 0, "No evaluators found"
        logger.info(f"Found {len(evaluators)} evaluators")
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_list_judges(compose_up_mcp_server: Any) -> None:
    """Test listing judges via SSE transport."""
    logger.info("Connecting to MCP server")
    client: ScorableMCPClient = ScorableMCPClient()

    try:
        await client.connect()

        judges: list[dict[str, Any]] = await client.list_judges()

        assert len(judges) > 0, "No judges found"
        logger.info(f"Found {len(judges)} judges")
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_run_evaluation(compose_up_mcp_server: Any) -> None:
    """Test running a standard evaluation via SSE transport."""
    logger.info("Connecting to MCP server")
    client: ScorableMCPClient = ScorableMCPClient()

    try:
        await client.connect()
        evaluators: list[dict[str, Any]] = await client.list_evaluators()

        clarity_evaluator: dict[str, Any] | None = next(
            (e for e in evaluators if e.get("name", "") == "Clarity"),
            next((e for e in evaluators if not e.get("inputs", {}).get("contexts")), None),
        )

        if not clarity_evaluator:
            pytest.skip("No standard evaluator found")

        logger.info(f"Using evaluator: {clarity_evaluator['name']}")

        result: dict[str, Any] = await client.run_evaluation(
            evaluator_id=clarity_evaluator["id"],
            request="What is the capital of France?",
            response="The capital of France is Paris, which is known as the City of Light.",
        )

        assert "score" in result, "No score in evaluation result"
        assert "justification" in result, "No justification in evaluation result"
        logger.info(f"Evaluation completed with score: {result['score']}")
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_run_rag_evaluation(compose_up_mcp_server: Any) -> None:
    """Test running a RAG evaluation via SSE transport."""
    logger.info("Connecting to MCP server")
    client: ScorableMCPClient = ScorableMCPClient()

    try:
        await client.connect()
        evaluators: list[dict[str, Any]] = await client.list_evaluators()

        faithfulness_evaluator: dict[str, Any] | None = next(
            (e for e in evaluators if e.get("name", "") == "Faithfulness"),
            next((e for e in evaluators if e.get("requires_contexts", False)), None),
        )

        assert faithfulness_evaluator is not None, "No RAG evaluator found"

        logger.info(f"Using evaluator: {faithfulness_evaluator['name']}")

        result: dict[str, Any] = await client.run_evaluation(
            evaluator_id=faithfulness_evaluator["id"],
            request="What is the capital of France?",
            response="The capital of France is Paris, which is known as the City of Light.",
            contexts=[
                "Paris is the capital and most populous city of France. It is located on the Seine River.",
                "France is a country in Western Europe with several overseas territories and regions.",
            ],
        )

        assert "score" in result, "No score in RAG evaluation result"
        assert "justification" in result, "No justification in RAG evaluation result"
        logger.info(f"RAG evaluation completed with score: {result['score']}")
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_evaluator_service_integration__standard_evaluation_by_id(
    compose_up_mcp_server: Any,
) -> None:
    """Test the standard evaluation by ID functionality through the evaluator service."""
    logger.info("Initializing EvaluatorService")
    service: EvaluatorService = EvaluatorService()

    evaluators_response: EvaluatorsListResponse = await service.list_evaluators()
    assert len(evaluators_response.evaluators) > 0, "No evaluator objects in the response"

    standard_evaluator: EvaluatorInfo | None = next(
        (e for e in evaluators_response.evaluators if not getattr(e, "requires_contexts", False)),
        None,
    )

    assert standard_evaluator is not None, (
        "No standard evaluator found - this is a test prerequisite"
    )

    logger.info(
        f"Using standard evaluator by ID: {standard_evaluator.name} ({standard_evaluator.id})"
    )

    retrieved_evaluator: EvaluatorInfo | None = await service.get_evaluator_by_id(
        standard_evaluator.id
    )
    assert retrieved_evaluator is not None, "Failed to retrieve evaluator by ID"
    assert retrieved_evaluator.id == standard_evaluator.id, (
        "Retrieved evaluator ID doesn't match requested ID"
    )

    eval_request = EvaluationRequest(
        evaluator_id=standard_evaluator.id,
        request="What is the capital of France?",
        response="The capital of France is Paris, which is known as the City of Light.",
    )

    eval_result: EvaluationResponse = await service.run_evaluation(eval_request)
    assert hasattr(eval_result, "score"), "Evaluation response missing score field"
    assert isinstance(eval_result.score, float), "Evaluation score should be a float"
    assert 0 <= eval_result.score <= 1, "Evaluation score should be between 0 and 1"
    assert eval_result.evaluator_name, "Evaluation response missing evaluator_name field"
    logger.info(f"Standard evaluation by ID result: score={eval_result.score}")


@pytest.mark.asyncio
async def test_evaluator_service_integration__standard_evaluation_by_name(
    compose_up_mcp_server: Any,
) -> None:
    """Test the standard evaluation by name functionality through the evaluator service."""
    logger.info("Initializing EvaluatorService")
    service: EvaluatorService = EvaluatorService()

    evaluators_response: EvaluatorsListResponse = await service.list_evaluators()
    assert len(evaluators_response.evaluators) > 0, "No evaluator objects in the response"

    standard_evaluator: EvaluatorInfo | None = next(
        (e for e in evaluators_response.evaluators if not getattr(e, "requires_contexts", False)),
        None,
    )

    assert standard_evaluator is not None, (
        "No standard evaluator found - this is a test prerequisite"
    )

    logger.info(f"Using standard evaluator by name: {standard_evaluator.name}")

    eval_request = EvaluationRequestByName(
        evaluator_name=standard_evaluator.name,
        request="What is the capital of France?",
        response="The capital of France is Paris, which is known as the City of Light.",
    )

    eval_result: EvaluationResponse = await service.run_evaluation_by_name(eval_request)
    assert hasattr(eval_result, "score"), "Evaluation response missing score field"
    assert isinstance(eval_result.score, float), "Evaluation score should be a float"
    assert 0 <= eval_result.score <= 1, "Evaluation score should be between 0 and 1"
    assert eval_result.evaluator_name, "Evaluation response missing evaluator_name field"
    logger.info(f"Standard evaluation by name result: score={eval_result.score}")


@pytest.mark.asyncio
async def test_evaluator_service_integration__rag_evaluation_by_id(
    compose_up_mcp_server: Any,
) -> None:
    """Test the RAG evaluation by ID functionality through the evaluator service."""
    logger.info("Initializing EvaluatorService")
    service: EvaluatorService = EvaluatorService()

    evaluators_response: EvaluatorsListResponse = await service.list_evaluators()
    assert len(evaluators_response.evaluators) > 0, "No evaluator objects in the response"

    rag_evaluator: EvaluatorInfo | None = next(
        (e for e in evaluators_response.evaluators if getattr(e, "requires_contexts", False)),
        None,
    )

    assert rag_evaluator is not None, "No RAG evaluator found - this is a test prerequisite"

    logger.info(f"Using RAG evaluator by ID: {rag_evaluator.name} ({rag_evaluator.id})")

    retrieved_evaluator: EvaluatorInfo | None = await service.get_evaluator_by_id(rag_evaluator.id)
    assert retrieved_evaluator is not None, "Failed to retrieve evaluator by ID"
    assert retrieved_evaluator.id == rag_evaluator.id, (
        "Retrieved evaluator ID doesn't match requested ID"
    )

    rag_request: EvaluationRequest = EvaluationRequest(
        evaluator_id=rag_evaluator.id,
        request="What is the capital of France?",
        response="The capital of France is Paris, which is known as the City of Light.",
        contexts=[
            "Paris is the capital and most populous city of France.",
            "France is a country in Western Europe.",
        ],
    )

    rag_result: EvaluationResponse = await service.run_evaluation(rag_request)
    assert hasattr(rag_result, "score"), "RAG evaluation response missing score field"
    assert isinstance(rag_result.score, float), "RAG evaluation score should be a float"
    assert 0 <= rag_result.score <= 1, "RAG evaluation score should be between 0 and 1"
    assert rag_result.evaluator_name, "RAG evaluation response missing evaluator_name field"
    logger.info(f"RAG evaluation by ID result: score={rag_result.score}")


@pytest.mark.asyncio
async def test_evaluator_service_integration__rag_evaluation_by_name(
    compose_up_mcp_server: Any,
) -> None:
    """Test the RAG evaluation by name functionality through the evaluator service."""
    logger.info("Initializing EvaluatorService")
    service: EvaluatorService = EvaluatorService()

    evaluators_response: EvaluatorsListResponse = await service.list_evaluators(
        max_count=120
    )  # Workaround to find one in long lists of custom evaluators, until RS-2660 is implemented
    assert len(evaluators_response.evaluators) > 0, "No evaluator objects in the response"

    rag_evaluator: EvaluatorInfo | None = next(
        (e for e in evaluators_response.evaluators if getattr(e, "requires_contexts", False)),
        None,
    )

    assert rag_evaluator is not None, "No RAG evaluator found - this is a test prerequisite"

    logger.info(f"Using RAG evaluator by name: {rag_evaluator.name}")

    rag_request: EvaluationRequestByName = EvaluationRequestByName(
        evaluator_name=rag_evaluator.name,
        request="What is the capital of France?",
        response="The capital of France is Paris, which is known as the City of Light.",
        contexts=[
            "Paris is the capital and most populous city of France.",
            "France is a country in Western Europe.",
        ],
    )

    rag_result: EvaluationResponse = await service.run_evaluation_by_name(rag_request)
    assert hasattr(rag_result, "score"), "RAG evaluation response missing score field"
    assert isinstance(rag_result.score, float), "RAG evaluation score should be a float"
    assert 0 <= rag_result.score <= 1, "RAG evaluation score should be between 0 and 1"
    assert rag_result.evaluator_name, "RAG evaluation response missing evaluator_name field"
    logger.info(f"RAG evaluation by name result: score={rag_result.score}")


@pytest.mark.asyncio
async def test_run_coding_policy_adherence(compose_up_mcp_server: Any) -> None:
    """Test running a coding policy adherence evaluation via SSE transport."""
    logger.info("Connecting to MCP server")
    client: ScorableMCPClient = ScorableMCPClient()

    try:
        await client.connect()

        result: dict[str, Any] = await client.run_coding_policy_adherence(
            policy_documents=[
                """
                # Your rule content

                Code Style and Structure:
                Python Style guide: Use Python 3.11 or later and modern language features such as match statements and the walrus operator. Always use type-hints and keyword arguments. Create Pydantic 2.0+ models for complicated data or function interfaces. Prefer readability of code and context locality to high layers of cognitively complex abstractions, even if some code is breaking DRY principles.

                Design approach: Domain Driven Design. E.g. model distinct domains, such as 3rd party API, as distinct pydantic models and translate between them and the local business logic with adapters.
                """,
            ],
            code="""
            def send_data_to_api(data):
                payload = {
                    "user": data["user_id"],
                    "timestamp": data["ts"],
                    "details": data.get("info", {}),
                }
                requests.post("https://api.example.com/data", json=payload)
            """,
        )

        assert "score" in result, "No score in coding policy adherence evaluation result"
        assert "justification" in result, (
            "No justification in coding policy adherence evaluation result"
        )
        logger.info(f"Coding policy adherence evaluation completed with score: {result['score']}")
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_run_judge(compose_up_mcp_server: Any) -> None:
    """Test running a judge via SSE transport."""
    logger.info("Connecting to MCP server")
    client: ScorableMCPClient = ScorableMCPClient()

    try:
        await client.connect()
        judges: list[dict[str, Any]] = await client.list_judges()

        judge: dict[str, Any] | None = next(iter(judges), None)

        if not judge:
            pytest.skip("No judge found")

        logger.info(f"Using judge: {judge['name']}")

        result: dict[str, Any] = await client.run_judge(
            judge_id=judge["id"],
            judge_name=judge["name"],
            request="What is the capital of France?",
            response="The capital of France is Paris, which is known as the City of Light.",
        )

        assert "evaluator_results" in result, "No evaluator results in judge result"
        assert len(result["evaluator_results"]) > 0, "No evaluator results in judge result"
        logger.info(f"Judge completed with score: {result['evaluator_results'][0]['score']}")
    finally:
        await client.disconnect()
