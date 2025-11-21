"""Integration tests for the Scorable MCP Client."""

import logging
from typing import Any

import pytest

from scorable_mcp.client import ScorableMCPClient
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
async def test_client_connection(compose_up_mcp_server: Any) -> None:
    """Test client connection and disconnection with a real server."""
    logger.info("Testing client connection")
    client = ScorableMCPClient()

    try:
        await client.connect()
        assert client.connected is True
        assert client.session is not None

        await client._ensure_connected()
        logger.info("Successfully connected to the MCP server")
    finally:
        await client.disconnect()
        assert client.session is None
        assert client.connected is False
        logger.info("Successfully disconnected from the MCP server")


@pytest.mark.asyncio
async def test_client_list_tools(compose_up_mcp_server: Any) -> None:
    """Test client list_tools method with a real server."""
    logger.info("Testing list_tools")
    client = ScorableMCPClient()

    try:
        await client.connect()

        tools = await client.list_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0

        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            # The schema key could be either inputSchema or input_schema depending on the MCP version
            assert "inputSchema" in tool or "input_schema" in tool, (
                f"Missing schema in tool: {tool}"
            )

        tool_names = [tool["name"] for tool in tools]
        logger.info(f"Found tools: {tool_names}")

        expected_tools = {
            "list_evaluators",
            "list_judges",
            "run_judge",
            "run_evaluation",
            "run_evaluation_by_name",
            "run_coding_policy_adherence",
        }
        assert expected_tools.issubset(set(tool_names)), (
            f"Missing expected tools. Found: {tool_names}"
        )
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_client_list_evaluators(compose_up_mcp_server: Any) -> None:
    """Test client list_evaluators method with a real server."""
    logger.info("Testing list_evaluators")
    client = ScorableMCPClient()

    try:
        await client.connect()

        evaluators = await client.list_evaluators()

        assert isinstance(evaluators, list)
        assert len(evaluators) > 0

        first_evaluator = evaluators[0]
        assert "id" in first_evaluator
        assert "name" in first_evaluator

        logger.info(f"Found {len(evaluators)} evaluators")
        logger.info(f"First evaluator: {first_evaluator['name']}")
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_client_list_judges(compose_up_mcp_server: Any) -> None:
    """Test client list_judges method with a real server."""
    logger.info("Testing list_judges")
    client = ScorableMCPClient()

    try:
        await client.connect()

        judges = await client.list_judges()

        assert isinstance(judges, list)
        assert len(judges) > 0

        first_judge = judges[0]
        assert "id" in first_judge
        assert "name" in first_judge

        assert "evaluators" in first_judge
        assert isinstance(first_judge["evaluators"], list)
        assert len(first_judge["evaluators"]) > 0

        for evaluator in first_judge["evaluators"]:
            assert "id" in evaluator
            assert "name" in evaluator

        logger.info(f"Found {len(judges)} judges")
        logger.info(f"First judge: {first_judge['name']}")
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_client_run_evaluation(compose_up_mcp_server: Any) -> None:
    """Test client run_evaluation method with a real server."""
    logger.info("Testing run_evaluation")
    client = ScorableMCPClient()

    try:
        await client.connect()

        evaluators = await client.list_evaluators()

        standard_evaluator = next(
            (e for e in evaluators if not e.get("requires_contexts", False)), None
        )

        assert standard_evaluator is not None, "No standard evaluator found"

        logger.info(f"Using evaluator: {standard_evaluator['name']}")

        result = await client.run_evaluation(
            evaluator_id=standard_evaluator["id"],
            request="What is the capital of France?",
            response="The capital of France is Paris, which is known as the City of Light.",
        )

        assert "score" in result
        assert "justification" in result
        logger.info(f"Evaluation score: {result['score']}")
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_client_run_judge(compose_up_mcp_server: Any) -> None:
    """Test client run_judge method with a real server."""
    logger.info("Testing run_judge")
    client = ScorableMCPClient()

    try:
        await client.connect()

        judges = await client.list_judges()

        judge = next(iter(judges), None)
        assert judge is not None, "No judge found"

        logger.info(f"Using judge: {judge['name']}")

        result = await client.run_judge(
            judge["id"],
            judge["name"],
            "What is the capital of France?",
            "The capital of France is Paris, which is known as the City of Light.",
        )

        assert "evaluator_results" in result
        assert len(result["evaluator_results"]) > 0

        evaluator_result = result["evaluator_results"][0]
        assert "evaluator_name" in evaluator_result
        assert "score" in evaluator_result
        assert "justification" in evaluator_result

        logger.info(f"Judge score: {evaluator_result['score']}")
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_client_run_evaluation_by_name(compose_up_mcp_server: Any) -> None:
    """Test client run_evaluation_by_name method with a real server."""
    logger.info("Testing run_evaluation_by_name")
    client = ScorableMCPClient()

    try:
        await client.connect()

        evaluators = await client.list_evaluators()

        standard_evaluator = next(
            (e for e in evaluators if not e.get("inputs", {}).get("contexts")), None
        )

        assert standard_evaluator is not None, "No standard evaluator found"

        logger.info(f"Using evaluator by name: {standard_evaluator['name']}")

        result = await client.run_evaluation_by_name(
            evaluator_name=standard_evaluator["name"],
            request="What is the capital of France?",
            response="The capital of France is Paris, which is known as the City of Light.",
        )

        assert "score" in result, "Result should contain a score"
        assert isinstance(result["score"], int | float), "Score should be numeric"
        assert "justification" in result, "Result should contain a justification"
        logger.info(f"Evaluation by name score: {result['score']}")
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_client_run_rag_evaluation(compose_up_mcp_server: Any) -> None:
    """Test client run_rag_evaluation method with a real server."""
    logger.info("Testing run_evaluation with contexts")
    client = ScorableMCPClient()

    try:
        await client.connect()

        evaluators = await client.list_evaluators()

        faithfulness_evaluators = [
            e
            for e in evaluators
            if any(
                kw in e.get("name", "").lower()
                for kw in ["faithfulness", "context", "rag", "relevance"]
            )
        ]

        rag_evaluator = next(iter(faithfulness_evaluators), None)

        assert rag_evaluator is not None, "Required RAG evaluator not found - test cannot proceed"

        logger.info(f"Using evaluator: {rag_evaluator['name']}")

        result = await client.run_evaluation(
            evaluator_id=rag_evaluator["id"],
            request="What is the capital of France?",
            response="The capital of France is Paris, which is known as the City of Light.",
            contexts=[
                "Paris is the capital and most populous city of France. It is located on the Seine River.",
                "France is a country in Western Europe with several overseas territories and regions.",
            ],
        )

        assert "score" in result, "Result should contain a score"
        assert isinstance(result["score"], int | float), "Score should be numeric"
        assert "justification" in result, "Result should contain a justification"
        logger.info(f"RAG evaluation score: {result['score']}")
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_client_run_rag_evaluation_by_name(compose_up_mcp_server: Any) -> None:
    """Test client run_rag_evaluation_by_name method with a real server."""
    logger.info("Testing run_evaluation_by_name with contexts")
    client = ScorableMCPClient()

    try:
        await client.connect()

        evaluators = await client.list_evaluators()

        faithfulness_evaluators = [
            e
            for e in evaluators
            if any(kw in e.get("name", "").lower() for kw in ["faithfulness", "context", "rag"])
            and "relevance"
            not in e.get("name", "").lower()  # Exclude known duplicate to avoid test flakyness
        ]

        rag_evaluator = next(iter(faithfulness_evaluators), None)

        assert rag_evaluator is not None, "Required RAG evaluator not found - test cannot proceed"

        logger.info(f"Using evaluator by name: {rag_evaluator['name']}")

        result = await client.run_rag_evaluation_by_name(
            evaluator_name=rag_evaluator["name"],
            request="What is the capital of France?",
            response="The capital of France is Paris, which is known as the City of Light.",
            contexts=[
                "Paris is the capital and most populous city of France. It is located on the Seine River.",
                "France is a country in Western Europe with several overseas territories and regions.",
            ],
        )

        assert "score" in result, "Result should contain a score"
        assert isinstance(result["score"], int | float), "Score should be numeric"
        assert "justification" in result, "Result should contain a justification"
        logger.info(f"RAG evaluation by name score: {result['score']}")
    finally:
        await client.disconnect()
