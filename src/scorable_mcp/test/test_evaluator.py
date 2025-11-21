"""Unit tests for the EvaluatorService module."""

import logging
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scorable_mcp.evaluator import EvaluatorService
from scorable_mcp.root_api_client import (
    ResponseValidationError,
    ScorableAPIError,
)
from scorable_mcp.schema import (
    ArrayInputItem,
    EvaluationRequest,
    EvaluationRequestByName,
    EvaluationResponse,
    EvaluatorInfo,
    RequiredInput,
)

logger = logging.getLogger("test_evaluator")


@pytest.fixture
def mock_api_client() -> Generator[MagicMock]:
    """Create a mock API client for testing."""
    with patch("scorable_mcp.evaluator.ScorableEvaluatorRepository") as mock_client_class:
        mock_client = MagicMock()
        mock_client.list_evaluators = AsyncMock()
        mock_client.run_evaluator = AsyncMock()
        mock_client.run_evaluator_by_name = AsyncMock()
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.mark.asyncio
async def test_fetch_evaluators_passes_max_count(mock_api_client: MagicMock) -> None:
    """Test that max_count is passed correctly to the API client."""
    service = EvaluatorService()
    await service.fetch_evaluators(max_count=75)
    mock_api_client.list_evaluators.assert_called_once_with(75)


@pytest.mark.asyncio
async def test_fetch_evaluators_uses_default_when_max_count_is_none(
    mock_api_client: MagicMock,
) -> None:
    """Test that default max_count is used when not specified."""
    service = EvaluatorService()
    await service.fetch_evaluators()
    mock_api_client.list_evaluators.assert_called_once_with(None)


@pytest.mark.asyncio
async def test_fetch_evaluators_handles_api_error(mock_api_client: MagicMock) -> None:
    """Test handling of ScorableAPIError in fetch_evaluators."""
    service = EvaluatorService()
    mock_api_client.list_evaluators.side_effect = ScorableAPIError(
        status_code=500, detail="Internal server error"
    )

    with pytest.raises(RuntimeError) as excinfo:
        await service.fetch_evaluators()

    assert "Cannot fetch evaluators" in str(excinfo.value)
    assert "Internal server error" in str(excinfo.value)


@pytest.mark.asyncio
async def test_fetch_evaluators_handles_validation_error(mock_api_client: MagicMock) -> None:
    """Test handling of ResponseValidationError in fetch_evaluators."""
    service = EvaluatorService()
    mock_api_client.list_evaluators.side_effect = ResponseValidationError(
        "Missing required field: 'id'", {"name": "Test"}
    )

    with pytest.raises(RuntimeError) as excinfo:
        await service.fetch_evaluators()

    assert "Invalid evaluators response" in str(excinfo.value)
    assert "Missing required field" in str(excinfo.value)


@pytest.mark.asyncio
async def test_get_evaluator_by_id_returns_correct_evaluator(mock_api_client: MagicMock) -> None:
    """Test that get_evaluator_by_id returns the correct evaluator when found."""
    service = EvaluatorService()
    mock_evaluators = [
        EvaluatorInfo(
            id="eval-1",
            name="Evaluator 1",
            created_at="2024-01-01T00:00:00Z",
            intent=None,
            inputs={},
        ),
        EvaluatorInfo(
            id="eval-2",
            name="Evaluator 2",
            created_at="2024-01-02T00:00:00Z",
            intent=None,
            inputs={
                "contexts": RequiredInput(type="array", items=ArrayInputItem(type="string")),
            },
        ),
    ]
    mock_api_client.list_evaluators.return_value = mock_evaluators

    evaluator = await service.get_evaluator_by_id("eval-2")

    assert evaluator is not None
    assert evaluator.id == "eval-2"
    assert evaluator.name == "Evaluator 2"


@pytest.mark.asyncio
async def test_get_evaluator_by_id_returns_none_when_not_found(mock_api_client: MagicMock) -> None:
    """Test that get_evaluator_by_id returns None when the evaluator is not found."""
    service = EvaluatorService()
    mock_evaluators = [
        EvaluatorInfo(
            id="eval-1",
            name="Evaluator 1",
            created_at="2024-01-01T00:00:00Z",
            intent=None,
            inputs={},
        ),
        EvaluatorInfo(
            id="eval-2",
            name="Evaluator 2",
            created_at="2024-01-02T00:00:00Z",
            intent=None,
            inputs={
                "contexts": RequiredInput(type="array", items=ArrayInputItem(type="string")),
            },
        ),
    ]
    mock_api_client.list_evaluators.return_value = mock_evaluators

    evaluator = await service.get_evaluator_by_id("eval-3")

    assert evaluator is None


@pytest.mark.asyncio
async def test_run_evaluation_passes_correct_parameters(mock_api_client: MagicMock) -> None:
    """Test that parameters are passed correctly to the API client in run_evaluation."""
    service = EvaluatorService()
    mock_response = EvaluationResponse(
        evaluator_name="Test Evaluator",
        score=0.95,
        justification="This is a justification",
        execution_log_id=None,
        cost=None,
    )
    mock_api_client.run_evaluator.return_value = mock_response

    request = EvaluationRequest(
        evaluator_id="eval-123",
        request="Test request",
        response="Test response",
        contexts=["Test context"],
        expected_output="Test expected output",
    )

    result = await service.run_evaluation(request)

    mock_api_client.run_evaluator.assert_called_once_with(
        evaluator_id="eval-123",
        request="Test request",
        response="Test response",
        contexts=["Test context"],
        expected_output="Test expected output",
    )

    assert result.evaluator_name == "Test Evaluator"
    assert result.score == 0.95
    assert result.justification == "This is a justification"


@pytest.mark.asyncio
async def test_run_evaluation_by_name_passes_correct_parameters(mock_api_client: MagicMock) -> None:
    """Test that parameters are passed correctly to the API client in run_evaluation_by_name."""
    service = EvaluatorService()
    mock_response = EvaluationResponse(
        evaluator_name="Test Evaluator",
        score=0.95,
        justification="This is a justification",
        execution_log_id=None,
        cost=None,
    )
    mock_api_client.run_evaluator_by_name.return_value = mock_response

    request = EvaluationRequestByName(
        evaluator_name="Clarity",
        request="Test request",
        response="Test response",
        contexts=["Test context"],
        expected_output="Test expected output",
    )

    result = await service.run_evaluation_by_name(request)

    mock_api_client.run_evaluator_by_name.assert_called_once_with(
        evaluator_name="Clarity",
        request="Test request",
        response="Test response",
        contexts=["Test context"],
        expected_output="Test expected output",
    )

    assert result.evaluator_name == "Test Evaluator"
    assert result.score == 0.95
    assert result.justification == "This is a justification"


@pytest.mark.asyncio
async def test_run_evaluation_handles_not_found_error(mock_api_client: MagicMock) -> None:
    """Test handling of 404 errors in run_evaluation."""
    service = EvaluatorService()
    mock_api_client.run_evaluator.side_effect = ScorableAPIError(
        status_code=404, detail="Evaluator not found"
    )

    request = EvaluationRequest(
        evaluator_id="nonexistent-id", request="Test request", response="Test response"
    )

    with pytest.raises(RuntimeError) as excinfo:
        await service.run_evaluation(request)

    assert "Failed to run evaluation" in str(excinfo.value)
    assert "Evaluator not found" in str(excinfo.value)


@pytest.mark.asyncio
async def test_transient_error_not_retried(mock_api_client: MagicMock) -> None:
    """Test that transient errors are not retried by default."""
    service = EvaluatorService()
    mock_api_client.run_evaluator.side_effect = ScorableAPIError(
        status_code=500, detail="Internal server error - may be transient"
    )

    request = EvaluationRequest(
        evaluator_id="eval-123", request="Test request", response="Test response"
    )

    with pytest.raises(RuntimeError):
        await service.run_evaluation(request)

    assert mock_api_client.run_evaluator.call_count == 1
