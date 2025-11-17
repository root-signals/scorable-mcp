"""Transport-agnostic core implementation of the Scorable MCP server.
Each transport layer only needs to:

1. instantiate `RootMCPServerCore`
2. expose its `app` through the chosen I/O mechanism.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from mcp.server.lowlevel import Server
from mcp.types import TextContent, Tool

from scorable_mcp import tools as tool_catalogue
from scorable_mcp.evaluator import EvaluatorService
from scorable_mcp.judge import JudgeService
from scorable_mcp.schema import (
    CodingPolicyAdherenceEvaluationRequest,
    EvaluationRequest,
    EvaluationRequestByName,
    EvaluationResponse,
    EvaluatorsListResponse,
    JudgesListResponse,
    ListEvaluatorsRequest,
    ListJudgesRequest,
    RunJudgeRequest,
    RunJudgeResponse,
    UnknownToolRequest,
)
from scorable_mcp.settings import settings

logger = logging.getLogger("scorable_mcp.core")


_Handler = Callable[[Any], Awaitable[Any]]


class RootMCPServerCore:  # noqa: D101
    def __init__(self) -> None:
        self.evaluator_service = EvaluatorService()
        self.judge_service = JudgeService()
        self.app = Server("Scorable Evaluators")

        @self.app.list_tools()
        async def _list_tools() -> list[Tool]:
            return await self.list_tools()

        @self.app.call_tool()
        async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            return await self.call_tool(name, arguments)

        self._function_map: dict[str, _Handler] = {
            "list_evaluators": self._handle_list_evaluators,
            "run_evaluation": self._handle_run_evaluation,
            "run_evaluation_by_name": self._handle_run_evaluation_by_name,
            "run_coding_policy_adherence": self._handle_coding_style_evaluation,
            "list_judges": self._handle_list_judges,
            "run_judge": self._handle_run_judge,
        }

    # ---------------------------------------------------------------------
    # Public API used by transports
    # ---------------------------------------------------------------------

    async def list_tools(self) -> list[Tool]:
        return tool_catalogue.get_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Validate *arguments* and dispatch to the proper *tool* handler."""

        logger.debug("Tool call %s with args %s", name, arguments)

        handler = self._function_map.get(name)
        if not handler:
            logger.warning("Unknown tool: %s", name)
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"}),
                )
            ]

        model_cls = tool_catalogue.get_request_model(name) or UnknownToolRequest
        try:
            request_model = model_cls(**arguments)  # type: ignore[arg-type]
        except Exception as exc:
            logger.error("Validation error for tool %s: %s", name, exc, exc_info=settings.debug)
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": f"Invalid arguments for {name}: {exc}"}),
                )
            ]

        try:
            result = await handler(request_model)  # type: ignore[arg-type]
            return [
                TextContent(
                    type="text",
                    text=result.model_dump_json(exclude_none=True),
                )
            ]
        except Exception as exc:
            logger.error("Error executing tool %s: %s", name, exc, exc_info=settings.debug)
            return [
                TextContent(
                    type="text",
                    text=json.dumps({"error": f"Error calling tool {name}: {exc}"}),
                )
            ]

    # ------------------------------------------------------------------
    # Handlers (internal)
    # ------------------------------------------------------------------

    async def _handle_list_evaluators(
        self, params: ListEvaluatorsRequest
    ) -> EvaluatorsListResponse:
        logger.debug("Handling list_evaluators request")
        return await self.evaluator_service.list_evaluators()

    async def _handle_run_evaluation(self, params: EvaluationRequest) -> EvaluationResponse:
        logger.debug("Handling run_evaluation for evaluator %s", params.evaluator_id)
        return await self.evaluator_service.run_evaluation(params)

    async def _handle_run_evaluation_by_name(
        self, params: EvaluationRequestByName
    ) -> EvaluationResponse:
        logger.debug("Handling run_evaluation_by_name for evaluator %s", params.evaluator_name)
        return await self.evaluator_service.run_evaluation_by_name(params)

    async def _handle_coding_style_evaluation(
        self, params: CodingPolicyAdherenceEvaluationRequest
    ) -> EvaluationResponse:
        logger.debug("Handling run_coding_policy_adherence request")

        rag_request = EvaluationRequest(
            evaluator_id=settings.coding_policy_evaluator_id,
            request=settings.coding_policy_evaluator_request,
            response=params.code,
            contexts=params.policy_documents,
        )

        return await self.evaluator_service.run_evaluation(rag_request)

    async def _handle_list_judges(self, _params: ListJudgesRequest) -> JudgesListResponse:
        """Handle list_judges tool call."""
        logger.debug("Handling list_judges request")
        return await self.judge_service.list_judges()

    async def _handle_run_judge(self, params: RunJudgeRequest) -> RunJudgeResponse:
        """Handle run_judge tool call."""
        logger.debug("Handling run_judge request for judge %s", params.judge_id)
        return await self.judge_service.run_judge(params)
