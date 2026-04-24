"""Unit tests for schema models."""

import pytest
from pydantic import ValidationError

from scorable_mcp.schema import (
    BaseEvaluationRequest,
    EvaluationRequest,
    EvaluationRequestByName,
    EvaluationResponse,
    JudgeEvaluatorResult,
    MessageTurn,
    RunJudgeRequest,
)

# ---------------------------------------------------------------------------
# MessageTurn
# ---------------------------------------------------------------------------


def test_message_turn__valid_user_turn() -> None:
    turn = MessageTurn(role="user", content="Hello")
    assert turn.role == "user"
    assert turn.content == "Hello"
    assert turn.contexts is None
    assert turn.tool_name is None


def test_message_turn__valid_assistant_turn_with_optional_fields() -> None:
    turn = MessageTurn(
        role="assistant",
        content="Here is the result",
        contexts=["doc1", "doc2"],
        tool_name="search",
    )
    assert turn.role == "assistant"
    assert turn.contexts == ["doc1", "doc2"]
    assert turn.tool_name == "search"


def test_message_turn__rejects_invalid_role() -> None:
    with pytest.raises(ValidationError):
        MessageTurn(role="system", content="Hello")  # type: ignore[arg-type]


def test_message_turn__rejects_missing_content() -> None:
    with pytest.raises(ValidationError):
        MessageTurn(role="user")  # type: ignore[call-arg]


def test_message_turn__rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        MessageTurn(role="user", content="Hi", unknown_field="oops")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# BaseEvaluationRequest — input format validation
# ---------------------------------------------------------------------------


def test_base_evaluation_request__accepts_single_turn_with_both_fields() -> None:
    req = BaseEvaluationRequest(request="What is the capital of France?", response="Paris")
    assert req.request == "What is the capital of France?"
    assert req.response == "Paris"
    assert req.turns is None


def test_base_evaluation_request__accepts_request_only() -> None:
    req = BaseEvaluationRequest(request="What is 2+2?")
    assert req.request == "What is 2+2?"
    assert req.response is None


def test_base_evaluation_request__accepts_response_only() -> None:
    req = BaseEvaluationRequest(response="Paris")
    assert req.response == "Paris"
    assert req.request is None


def test_base_evaluation_request__accepts_multi_turn() -> None:
    turns = [
        MessageTurn(role="user", content="Hello"),
        MessageTurn(role="assistant", content="Hi there!"),
    ]
    req = BaseEvaluationRequest(turns=turns)
    assert req.turns is not None
    assert len(req.turns) == 2
    assert req.request is None
    assert req.response is None


def test_base_evaluation_request__rejects_turns_combined_with_request() -> None:
    with pytest.raises(ValidationError, match="Cannot provide both"):
        BaseEvaluationRequest(
            request="Hello",
            turns=[MessageTurn(role="user", content="Hello")],
        )


def test_base_evaluation_request__rejects_turns_combined_with_response() -> None:
    with pytest.raises(ValidationError, match="Cannot provide both"):
        BaseEvaluationRequest(
            response="World",
            turns=[MessageTurn(role="user", content="Hello")],
        )


def test_base_evaluation_request__rejects_no_input_at_all() -> None:
    with pytest.raises(ValidationError, match="Either"):
        BaseEvaluationRequest()


def test_base_evaluation_request__rejects_whitespace_only_fields() -> None:
    with pytest.raises(ValidationError, match="Either"):
        BaseEvaluationRequest(request="   ", response="   ")


def test_base_evaluation_request__accepts_optional_metadata_fields() -> None:
    req = BaseEvaluationRequest(
        request="What is 2+2?",
        response="4",
        tags=["prod", "math"],
        user_id="user-123",
        session_id="session-456",
        system_prompt="You are a helpful assistant",
    )
    assert req.tags == ["prod", "math"]
    assert req.user_id == "user-123"
    assert req.session_id == "session-456"
    assert req.system_prompt == "You are a helpful assistant"


def test_base_evaluation_request__metadata_fields_default_to_none() -> None:
    req = BaseEvaluationRequest(request="Hello", response="Hi")
    assert req.tags is None
    assert req.user_id is None
    assert req.session_id is None
    assert req.system_prompt is None


# ---------------------------------------------------------------------------
# EvaluationRequest
# ---------------------------------------------------------------------------


def test_evaluation_request__requires_evaluator_id() -> None:
    with pytest.raises(ValidationError):
        EvaluationRequest(request="Hello", response="Hi")  # type: ignore[call-arg]


def test_evaluation_request__inherits_multi_turn_support() -> None:
    req = EvaluationRequest(
        evaluator_id="eval-123",
        turns=[MessageTurn(role="user", content="Hello")],
    )
    assert req.evaluator_id == "eval-123"
    assert req.turns is not None


def test_evaluation_request__inherits_metadata_fields() -> None:
    req = EvaluationRequest(
        evaluator_id="eval-123",
        request="Hello",
        response="Hi",
        tags=["test"],
        user_id="u1",
    )
    assert req.tags == ["test"]
    assert req.user_id == "u1"


# ---------------------------------------------------------------------------
# EvaluationRequestByName
# ---------------------------------------------------------------------------


def test_evaluation_request_by_name__rejects_turns_with_single_turn() -> None:
    with pytest.raises(ValidationError, match="Cannot provide both"):
        EvaluationRequestByName(
            evaluator_name="Clarity",
            request="Hello",
            turns=[MessageTurn(role="user", content="Hello")],
        )


def test_evaluation_request_by_name__accepts_multi_turn() -> None:
    req = EvaluationRequestByName(
        evaluator_name="Clarity",
        turns=[MessageTurn(role="user", content="Hello")],
    )
    assert req.evaluator_name == "Clarity"
    assert req.turns is not None


def test_evaluation_request_by_name__inherits_metadata_fields() -> None:
    req = EvaluationRequestByName(
        evaluator_name="Clarity",
        request="Hello",
        response="Hi",
        session_id="s1",
        system_prompt="Be terse",
    )
    assert req.session_id == "s1"
    assert req.system_prompt == "Be terse"


# ---------------------------------------------------------------------------
# RunJudgeRequest — inherits from BaseEvaluationRequest
# ---------------------------------------------------------------------------


def test_run_judge_request__accepts_all_new_fields() -> None:
    req = RunJudgeRequest(
        judge_id="judge-123",
        request="What is 2+2?",
        response="4",
        tags=["prod"],
        user_id="user-1",
        session_id="session-1",
        system_prompt="Be helpful",
    )
    assert req.judge_id == "judge-123"
    assert req.tags == ["prod"]
    assert req.user_id == "user-1"
    assert req.session_id == "session-1"
    assert req.system_prompt == "Be helpful"


def test_run_judge_request__accepts_multi_turn() -> None:
    req = RunJudgeRequest(
        judge_id="judge-123",
        turns=[
            MessageTurn(role="user", content="Hello"),
            MessageTurn(role="assistant", content="Hi!"),
        ],
    )
    assert req.turns is not None
    assert len(req.turns) == 2
    assert req.request is None


def test_run_judge_request__rejects_turns_combined_with_single_turn() -> None:
    with pytest.raises(ValidationError, match="Cannot provide both"):
        RunJudgeRequest(
            judge_id="judge-123",
            request="Hello",
            turns=[MessageTurn(role="user", content="Hello")],
        )


def test_run_judge_request__rejects_no_input() -> None:
    with pytest.raises(ValidationError, match="Either"):
        RunJudgeRequest(judge_id="judge-123")


# ---------------------------------------------------------------------------
# JudgeEvaluatorResult — nullable fields and new confidence field
# ---------------------------------------------------------------------------


def test_judge_evaluator_result__accepts_null_score() -> None:
    result = JudgeEvaluatorResult(
        evaluator_name="Test",
        score=None,
        justification="Could not determine a score",
    )
    assert result.score is None


def test_judge_evaluator_result__accepts_null_justification() -> None:
    result = JudgeEvaluatorResult(
        evaluator_name="Test",
        score=0.9,
        justification=None,
    )
    assert result.justification is None


def test_judge_evaluator_result__accepts_both_null() -> None:
    result = JudgeEvaluatorResult(evaluator_name="Test", score=None, justification=None)
    assert result.score is None
    assert result.justification is None


def test_judge_evaluator_result__accepts_confidence() -> None:
    result = JudgeEvaluatorResult(
        evaluator_name="Test",
        score=0.9,
        justification="Good",
        confidence=0.85,
    )
    assert result.confidence == 0.85


def test_judge_evaluator_result__confidence_defaults_to_none() -> None:
    result = JudgeEvaluatorResult(evaluator_name="Test", score=0.9, justification="Good")
    assert result.confidence is None


def test_judge_evaluator_result__ignores_extra_api_fields() -> None:
    result = JudgeEvaluatorResult.model_validate(
        {
            "evaluator_name": "Test",
            "score": 0.9,
            "justification": "Good",
            "evaluator_id": "some-uuid",
            "evaluator_version_id": "another-uuid",
        }
    )
    assert result.evaluator_name == "Test"
    assert not hasattr(result, "evaluator_id")
    assert not hasattr(result, "evaluator_version_id")


# ---------------------------------------------------------------------------
# EvaluationResponse — new confidence field
# ---------------------------------------------------------------------------


def test_evaluation_response__accepts_confidence() -> None:
    result = EvaluationResponse(
        evaluator_name="Test",
        score=0.9,
        confidence=0.85,
    )
    assert result.confidence == 0.85


def test_evaluation_response__confidence_defaults_to_none() -> None:
    result = EvaluationResponse(evaluator_name="Test", score=0.9)
    assert result.confidence is None


def test_evaluation_response__ignores_extra_api_fields() -> None:
    result = EvaluationResponse.model_validate(
        {
            "evaluator_name": "Test",
            "score": 0.9,
            "justification": "Good",
            "some_future_field": "value",
        }
    )
    assert result.evaluator_name == "Test"
    assert not hasattr(result, "some_future_field")
