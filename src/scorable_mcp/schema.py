"""Type definitions for the Scorable MCP Server.

This module defines Pydantic models and other types used across the server.
"""

from typing import Literal, TypeVar

from pydantic import BaseModel, Field, model_validator

K = TypeVar("K")
V = TypeVar("V")


class BaseToolRequest(BaseModel):
    """Base class for all tool request models."""

    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
    }


class ListEvaluatorsRequest(BaseToolRequest):
    """Request model for listing evaluators.

    This is an empty request as list_evaluators doesn't require any parameters.
    """

    pass


#####################################################################
### Implementation specific models                                ###
#####################################################################


class UnknownToolRequest(BaseToolRequest):
    """Request model for handling unknown tools.

    This allows for capturing any parameters passed to unknown tools for debugging.
    """

    model_config = {
        "extra": "allow",  # Allow any fields for debugging purposes
    }


class BaseScorableModel(BaseModel):
    """Base class for all models that interact with the Scorable API.

    This class sets up handling of schema evolution to:
    1. Ignore new fields that might be added to the API in the future
    2. Still fail if expected fields are removed from the API response
    """

    model_config = {
        "extra": "ignore",
        "strict": True,
        "validate_assignment": True,
    }


#####################################################################
### LLM Facing Models                                             ###
### Make sure to add good descriptions and examples, where needed ###
#####################################################################


class MessageTurn(BaseModel):
    """A single turn in a multi-turn conversation."""

    model_config = {
        "extra": "forbid",
    }

    role: Literal["user", "assistant"] = Field(..., description="Role of the speaker")
    content: str = Field(..., description="Content of the turn")
    contexts: list[str] | None = Field(
        default=None,
        description="Context documents for this turn (assistant role only)",
    )
    tool_name: str | None = Field(
        default=None,
        description="Tool name if this turn represents a tool call result (assistant role only)",
    )


class BaseEvaluationRequest(BaseScorableModel):
    """Fields common to all evaluation requests."""

    request: str | None = Field(
        default=None,
        description="The user query to evaluate. Provide this with 'response' for single-turn, or use 'turns' for multi-turn conversations.",
    )
    response: str | None = Field(
        default=None,
        description="The AI assistant's response to evaluate. Provide this with 'request' for single-turn, or use 'turns' for multi-turn conversations.",
    )
    turns: list[MessageTurn] | None = Field(
        default=None,
        description="Multi-turn conversation to evaluate. Use this instead of 'request'/'response' for multi-turn conversations.",
    )
    contexts: list[str] | None = Field(
        default=None,
        description="List of required context strings for evaluation. Used only for evaluators that have 'contexts' defined in their inputs.",
    )
    expected_output: str | None = Field(
        default=None,
        description="The expected LLM response. Used only for evaluators that have 'expected_output' defined in their inputs.",
    )
    tags: list[str] | None = Field(
        default=None,
        description="Optional tags to attach to this evaluation for tracking and filtering.",
    )
    user_id: str | None = Field(
        default=None,
        description="Optional external user identifier for tracking evaluations per user.",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional external session identifier for tracking evaluations per session.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt used in the LLM interaction being evaluated.",
    )

    @model_validator(mode="after")
    def validate_input_format(self) -> "BaseEvaluationRequest":
        has_turns = bool(self.turns)
        has_single = bool(
            (self.request and self.request.strip()) or (self.response and self.response.strip())
        )
        if has_turns and has_single:
            raise ValueError("Cannot provide both 'turns' and 'request'/'response'")
        if not has_turns and not has_single:
            raise ValueError(
                "Either 'turns' or at least one of 'request'/'response' must be provided"
            )
        return self


class EvaluationRequestByName(BaseEvaluationRequest):
    """
    Model for evaluation request parameters.

    this is based on the EvaluatorExecutionRequest model from the Scorable API
    """

    evaluator_name: str = Field(
        ...,
        description="The EXACT name of the evaluator as returned by the `list_evaluators` tool, including spaces and special characters",
        examples=[
            "Compliance-preview",
            "Truthfulness - Global",
            "Safety for Children",
            "Context Precision",
        ],
    )


class EvaluationRequest(BaseEvaluationRequest):
    """
    Model for evaluation request parameters.

    this is based on the EvaluatorExecutionRequest model from the Scorable API
    """

    evaluator_id: str = Field(..., description="The ID of the evaluator to use")


class CodingPolicyAdherenceEvaluationRequest(BaseToolRequest):
    """Request model for coding policy adherence evaluation tool."""

    policy_documents: list[str] = Field(
        ...,
        description="The policy documents which describe the coding policy, such as cursor/rules file contents",
    )
    code: str = Field(..., description="The code to evaluate")


#####################################################################
### Simplified Scorable Platform API models                    ###
### We trim them down to save tokens                              ###
#####################################################################
class EvaluationResponse(BaseScorableModel):
    """
    Model for evaluation response.

    Trimmed down version of
    root.generated.openapi_aclient.models.evaluator_execution_result.EvaluatorExecutionResult
    """

    evaluator_name: str = Field(..., description="Name of the evaluator")
    score: float = Field(..., description="Evaluation score (0-1)")
    justification: str | None = Field(None, description="Justification for the score")
    execution_log_id: str | None = Field(None, description="Execution log ID for use in monitoring")
    cost: float | int | None = Field(None, description="Cost of the evaluation")
    confidence: float | None = Field(None, description="Confidence score of the evaluation (0-1)")


class ArrayInputItem(BaseModel):
    type: str


class RequiredInput(BaseModel):
    type: str
    items: ArrayInputItem | None = None


class EvaluatorInfo(BaseScorableModel):
    """
    Model for evaluator information.

    Trimmed down version of root.generated.openapi_aclient.models.evaluator.Evaluator
    """

    name: str = Field(..., description="Name of the evaluator")
    id: str = Field(..., description="ID of the evaluator")
    created_at: str = Field(..., description="Creation timestamp of the evaluator")
    intent: str | None = Field(None, description="Intent of the evaluator")
    inputs: dict[str, RequiredInput] = Field(
        ...,
        description="Schema defining the input parameters required for running the evaluator (run_evaluation parameters).",
    )

    @property
    def requires_contexts(self) -> bool:
        return self.inputs.get("contexts") is not None

    @property
    def requires_expected_output(self) -> bool:
        return self.inputs.get("expected_output") is not None


class EvaluatorsListResponse(BaseScorableModel):
    """List of evaluators returned by `list_evaluators`."""

    evaluators: list[EvaluatorInfo] = Field(..., description="List of evaluators")


class ListJudgesRequest(BaseToolRequest):
    """Request model for listing judges.

    This is an empty request as list_judges doesn't require any parameters.
    """

    pass


class JudgeInfo(BaseScorableModel):
    """
    Model for judge information.
    """

    class NestedEvaluatorInfo(BaseScorableModel):
        """Nested evaluator info."""

        name: str = Field(..., description="Name of the evaluator")
        id: str = Field(..., description="ID of the evaluator")
        intent: str | None = Field(default="", description="Intent of the evaluator")

    name: str = Field(..., description="Name of the judge")
    id: str = Field(..., description="ID of the judge")
    created_at: str = Field(..., description="Creation timestamp of the judge")
    evaluators: list[NestedEvaluatorInfo] = Field(..., description="List of evaluators")
    description: str | None = Field(None, description="Description of the judge")


class JudgesListResponse(BaseScorableModel):
    """Model for judges list response."""

    judges: list[JudgeInfo] = Field(..., description="List of judges")


class RunJudgeRequest(BaseEvaluationRequest):
    """Request model for run_judge tool."""

    judge_id: str = Field(..., description="The ID of the judge to use")
    judge_name: str = Field(
        default="-",
        description="The name of the judge to use. Optional, only for logging purposes.",
    )


class JudgeEvaluatorResult(BaseScorableModel):
    """Model for judge evaluator result."""

    evaluator_name: str = Field(..., description="Name of the evaluator")
    score: float | None = Field(..., description="Score of the evaluator")
    justification: str | None = Field(..., description="Justification for the score")
    confidence: float | None = Field(None, description="Confidence score of the evaluation (0-1)")


class RunJudgeResponse(BaseScorableModel):
    """Model for judge response."""

    evaluator_results: list[JudgeEvaluatorResult] = Field(
        ..., description="List of evaluator results"
    )


# Re-export MessageTurn so callers can import it from this module
__all__ = [
    "ArrayInputItem",
    "BaseEvaluationRequest",
    "BaseScorableModel",
    "BaseToolRequest",
    "CodingPolicyAdherenceEvaluationRequest",
    "EvaluationRequest",
    "EvaluationRequestByName",
    "EvaluationResponse",
    "EvaluatorInfo",
    "EvaluatorsListResponse",
    "JudgeEvaluatorResult",
    "JudgeInfo",
    "JudgesListResponse",
    "ListEvaluatorsRequest",
    "ListJudgesRequest",
    "MessageTurn",
    "RequiredInput",
    "RunJudgeRequest",
    "RunJudgeResponse",
    "UnknownToolRequest",
]
