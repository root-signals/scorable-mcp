"""Tool catalogue for the Scorable MCP server."""

from __future__ import annotations

from mcp.types import Tool

from scorable_mcp.schema import (
    CodingPolicyAdherenceEvaluationRequest,
    EvaluationRequest,
    EvaluationRequestByName,
    ListEvaluatorsRequest,
    ListJudgesRequest,
    RunJudgeRequest,
)


def get_tools() -> list[Tool]:
    """Return the list of MCP *tools* supported by Scorable."""

    return [
        Tool(
            name="list_evaluators",
            description="List all available evaluators from Scorable",
            inputSchema=ListEvaluatorsRequest.model_json_schema(),
        ),
        Tool(
            name="run_evaluation",
            description="Run a standard evaluation using a Scorable evaluator by ID",
            inputSchema=EvaluationRequest.model_json_schema(),
        ),
        Tool(
            name="run_evaluation_by_name",
            description="Run a standard evaluation using a Scorable evaluator by name",
            inputSchema=EvaluationRequestByName.model_json_schema(),
        ),
        Tool(
            name="run_coding_policy_adherence",
            description="Evaluate code against repository coding policy documents using a dedicated Scorable evaluator",
            inputSchema=CodingPolicyAdherenceEvaluationRequest.model_json_schema(),
        ),
        Tool(
            name="list_judges",
            description="List all available judges from Scorable. Judge is a collection of evaluators forming LLM-as-a-judge.",
            inputSchema=ListJudgesRequest.model_json_schema(),
        ),
        Tool(
            name="run_judge",
            description="Run a judge using a Scorable judge by ID",
            inputSchema=RunJudgeRequest.model_json_schema(),
        ),
    ]


def get_request_model(tool_name: str) -> type | None:
    """Return the Pydantic *request* model class for a given tool.

    This is useful for validating the *arguments* dict passed to
    MCP-`call_tool` before dispatching.
    Returns ``None`` if the name is unknown; caller can then fall back to
    a generic model or raise.
    """

    mapping: dict[str, type] = {
        "list_evaluators": ListEvaluatorsRequest,
        "list_judges": ListJudgesRequest,
        "run_coding_policy_adherence": CodingPolicyAdherenceEvaluationRequest,
        "run_evaluation_by_name": EvaluationRequestByName,
        "run_evaluation": EvaluationRequest,
        "run_judge": RunJudgeRequest,
    }

    return mapping.get(tool_name)
