"""Integration tests for the Scorable MCP Server using stdio transport."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult

from scorable_mcp.settings import settings

pytestmark = [
    pytest.mark.skipif(
        settings.scorable_api_key.get_secret_value() == "",
        reason="SCORABLE_API_KEY environment variable not set or empty",
    ),
    pytest.mark.integration,
    pytest.mark.asyncio,
]

logger = logging.getLogger("scorable_mcp_tests")
PROJECT_ROOT = Path(__file__).parents[4]


@pytest.mark.asyncio
async def test_direct_core_list_tools() -> None:
    """Test listing tools directly from the RootMCPServerCore."""
    from scorable_mcp.core import RootMCPServerCore

    logger.info("Testing direct core tool listing")
    core = RootMCPServerCore()

    tools = await core.list_tools()

    tool_names = {tool.name for tool in tools}
    expected_tools = {
        "list_evaluators",
        "run_evaluation",
        "run_evaluation_by_name",
        "run_coding_policy_adherence",
    }

    assert expected_tools.issubset(tool_names), f"Missing expected tools. Found: {tool_names}"
    logger.info(f"Found expected tools: {tool_names}")


@pytest.mark.asyncio
async def test_direct_core_list_evaluators() -> None:
    """Test calling the list_evaluators tool directly from the RootMCPServerCore."""
    from scorable_mcp.core import RootMCPServerCore

    logger.info("Testing direct core list_evaluators")
    core = RootMCPServerCore()

    result = await core.call_tool("list_evaluators", {})

    assert len(result) > 0, "No content in response"
    text_content = result[0]
    assert text_content.type == "text", "Response is not text type"

    evaluators_response = json.loads(text_content.text)

    assert "evaluators" in evaluators_response, "No evaluators in response"
    evaluators = evaluators_response["evaluators"]
    assert len(evaluators) > 0, "No evaluators found"

    evaluator = evaluators[0]
    assert "id" in evaluator, "Evaluator missing ID"
    assert "name" in evaluator, "Evaluator missing name"

    logger.info(f"Found {len(evaluators)} evaluators")


@pytest.mark.asyncio
async def test_direct_core_list_judges() -> None:
    """Test calling the list_judges tool directly from the RootMCPServerCore."""
    from scorable_mcp.core import RootMCPServerCore

    logger.info("Testing direct core list_judges")
    core = RootMCPServerCore()

    result = await core.call_tool("list_judges", {})

    assert len(result) > 0, "No content in response"
    text_content = result[0]
    assert text_content.type == "text", "Response is not text type"

    judges_response = json.loads(text_content.text)

    assert "judges" in judges_response, "No judges in response"
    judges = judges_response["judges"]
    assert len(judges) > 0, "No judges found"


@pytest.mark.asyncio
async def test_stdio_client_list_tools() -> None:
    """Use the upstream MCP stdio client to talk to our stdio server and list tools.

    This replaces the previous hand-rolled subprocess test with an end-to-end
    check that exercises the *actual* MCP handshake and client-side logic.
    """

    server_env = os.environ.copy()
    server_env["SCORABLE_API_KEY"] = settings.scorable_api_key.get_secret_value()

    server_params = StdioServerParameters(  # type: ignore[call-arg]
        command=sys.executable,
        args=["-m", "scorable_mcp.stdio_server"],
        env=server_env,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):  # type: ignore[attr-defined]
        async with ClientSession(read_stream, write_stream) as session:  # type: ignore
            await session.initialize()

            tools_response = await session.list_tools()
            tool_names = {tool.name for tool in tools_response.tools}

            expected_tools = {
                "list_evaluators",
                "run_evaluation",
                "run_evaluation_by_name",
                "run_coding_policy_adherence",
            }

            missing = expected_tools - tool_names
            assert not missing, f"Missing expected tools: {missing}"
            logger.info("stdio-client -> list_tools OK: %s", tool_names)


@pytest.mark.asyncio
async def test_stdio_client_run_evaluation_by_name() -> None:
    """Test running an evaluation by name using the stdio client."""

    server_env = os.environ.copy()
    server_env["SCORABLE_API_KEY"] = settings.scorable_api_key.get_secret_value()

    server_params = StdioServerParameters(  # type: ignore[call-arg]
        command=sys.executable,
        args=["-m", "scorable_mcp.stdio_server"],
        env=server_env,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):  # type: ignore[attr-defined]
        async with ClientSession(read_stream, write_stream) as session:  # type: ignore
            await session.initialize()

            tools_response = await session.list_tools()
            assert any(tool.name == "list_evaluators" for tool in tools_response.tools), (
                "list_evaluators tool not found"
            )

            call_result = await session.call_tool("list_evaluators", {})
            evaluators_json = _extract_text_payload(call_result)
            evaluators_data = json.loads(evaluators_json)

            relevance_evaluator = None
            for evaluator in evaluators_data["evaluators"]:
                if evaluator["name"] == "Relevance":
                    relevance_evaluator = evaluator
                    break

            if not relevance_evaluator:
                for evaluator in evaluators_data["evaluators"]:
                    if not evaluator.get("requires_contexts", False):
                        relevance_evaluator = evaluator
                        break

            assert relevance_evaluator is not None, "No suitable evaluator found for testing"
            logger.info(f"Using evaluator: {relevance_evaluator['name']}")

            call_result = await session.call_tool(
                "run_evaluation_by_name",
                {
                    "evaluator_name": relevance_evaluator["name"],
                    "request": "What is the capital of France?",
                    "response": "The capital of France is Paris, which is known as the City of Light.",
                },
            )
            assert call_result is not None
            assert len(call_result.content) > 0

            logger.info(f"Call result: {call_result}")
            print(f"Call result: {call_result}")
            evaluation_json = _extract_text_payload(call_result)
            evaluation_data = json.loads(evaluation_json)

            # Verify evaluation response
            assert "score" in evaluation_data, "No score in evaluation response"
            assert "evaluator_name" in evaluation_data, "No evaluator_name in evaluation response"
            assert 0 <= float(evaluation_data["score"]) <= 1, "Score should be between 0 and 1"

            logger.info(f"Evaluation completed with score: {evaluation_data['score']}")


@pytest.mark.asyncio
async def test_stdio_client_run_judge() -> None:
    """Test running a judge using the stdio client."""

    server_env = os.environ.copy()
    server_env["SCORABLE_API_KEY"] = settings.scorable_api_key.get_secret_value()

    server_params = StdioServerParameters(  # type: ignore[call-arg]
        command=sys.executable,
        args=["-m", "scorable_mcp.stdio_server"],
        env=server_env,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):  # type: ignore[attr-defined]
        async with ClientSession(read_stream, write_stream) as session:  # type: ignore
            await session.initialize()

            call_result = await session.call_tool("list_judges", {})
            judges_json = _extract_text_payload(call_result)
            judges_data = json.loads(judges_json)

            assert "judges" in judges_data and len(judges_data["judges"]) > 0

            judge = judges_data["judges"][0]

            call_result = await session.call_tool(
                "run_judge",
                {
                    "judge_id": judge["id"],
                    "request": "What is the capital of France?",
                    "response": "The capital of France is Paris, which is known as the City of Light.",
                },
            )

            assert call_result is not None
            assert len(call_result.content) > 0

            judge_result_json = _extract_text_payload(call_result)
            response_data = json.loads(judge_result_json)

            assert "evaluator_results" in response_data, "Response missing evaluator_results"
            assert len(response_data["evaluator_results"]) > 0, "No evaluator results in response"
            assert "score" in response_data["evaluator_results"][0], "Response missing score"
            assert "justification" in response_data["evaluator_results"][0], (
                "Response missing justification"
            )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _extract_text_payload(call_tool_result: CallToolResult) -> str:
    """Return the text content from a *CallToolResult* as emitted by the MCP SDK.

    The upstream type wraps returned *content* in a list of *Content* objects
    (``TextContent``, ``ImageContent``, …).  For text-based tools we expect a
    single ``TextContent`` item; this helper centralises the extraction logic
    to avoid copy-pasting error-prone indexing throughout the tests.
    """

    assert call_tool_result is not None and len(call_tool_result.content) > 0, (
        "CallToolResult has no content"
    )

    first_item = call_tool_result.content[0]
    assert first_item.type == "text", f"Unexpected content type: {first_item.type}"

    return getattr(first_item, "text")


@pytest.mark.asyncio
async def test_stdio_client_call_tool_list_evaluators() -> None:
    """Verify that calling *list_evaluators* via the stdio client returns JSON."""

    server_env = os.environ.copy()
    server_env["SCORABLE_API_KEY"] = settings.scorable_api_key.get_secret_value()

    server_params = StdioServerParameters(  # type: ignore[call-arg]
        command=sys.executable,
        args=["-m", "scorable_mcp.stdio_server"],
        env=server_env,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):  # type: ignore[attr-defined]
        async with ClientSession(read_stream, write_stream) as session:  # type: ignore
            await session.initialize()

            call_result = await session.call_tool("list_evaluators", {})
            evaluators_json = _extract_text_payload(call_result)
            evaluators_data = json.loads(evaluators_json)

            assert "evaluators" in evaluators_data and len(evaluators_data["evaluators"]) > 0


@pytest.mark.asyncio
async def test_stdio_client_call_tool_list_judges() -> None:
    """Verify that calling *list_judges* via the stdio client returns JSON."""

    server_env = os.environ.copy()
    server_env["SCORABLE_API_KEY"] = settings.scorable_api_key.get_secret_value()

    server_params = StdioServerParameters(  # type: ignore[call-arg]
        command=sys.executable,
        args=["-m", "scorable_mcp.stdio_server"],
        env=server_env,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):  # type: ignore[attr-defined]
        async with ClientSession(read_stream, write_stream) as session:  # type: ignore
            await session.initialize()

            call_result = await session.call_tool("list_judges", {})
            judges_json = _extract_text_payload(call_result)
            judges_data = json.loads(judges_json)

            assert "judges" in judges_data and len(judges_data["judges"]) > 0
