"""Unit tests for the JudgeService module."""

import logging
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scorable_mcp.judge import JudgeService
from scorable_mcp.root_api_client import ResponseValidationError, ScorableAPIError
from scorable_mcp.schema import JudgeEvaluatorResult, RunJudgeRequest, RunJudgeResponse

logger = logging.getLogger("test_judge")


@pytest.fixture
def mock_api_client() -> Generator[MagicMock]:
    """Create a mock API client for testing."""
    with patch("scorable_mcp.judge.ScorableJudgeRepository") as mock_client_class:
        mock_client = MagicMock()
        mock_client.list_judges = AsyncMock()
        mock_client.run_judge = AsyncMock()
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.mark.asyncio
async def test_fetch_judges_passes_max_count(mock_api_client: MagicMock) -> None:
    """Test that max_count is passed correctly to the API client."""
    service = JudgeService()
    await service.fetch_judges(max_count=75)
    mock_api_client.list_judges.assert_called_once_with(75)


@pytest.mark.asyncio
async def test_fetch_judges_handles_api_error(mock_api_client: MagicMock) -> None:
    """Test handling of ScorableAPIError in fetch_judges."""
    service = JudgeService()
    mock_api_client.list_judges.side_effect = ScorableAPIError(
        status_code=500, detail="Internal server error"
    )

    with pytest.raises(RuntimeError) as excinfo:
        await service.fetch_judges()

    assert "Cannot fetch judges" in str(excinfo.value)
    assert "Internal server error" in str(excinfo.value)


@pytest.mark.asyncio
async def test_run_judge_passes_correct_parameters(mock_api_client: MagicMock) -> None:
    """Test that parameters are passed correctly to the API client in run_judge."""
    service = JudgeService()
    evaluator_results = [
        JudgeEvaluatorResult(
            evaluator_name="Test Evaluator", score=0.95, justification="This is a justification"
        )
    ]
    mock_response = RunJudgeResponse(evaluator_results=evaluator_results)
    mock_api_client.run_judge.return_value = mock_response

    request = RunJudgeRequest(
        judge_id="judge-123",
        judge_name="Test Judge",
        request="Test request",
        response="Test response",
    )

    result = await service.run_judge(request)

    mock_api_client.run_judge.assert_called_once_with(request)

    assert result.evaluator_results[0].evaluator_name == "Test Evaluator"
    assert result.evaluator_results[0].score == 0.95
    assert result.evaluator_results[0].justification == "This is a justification"


@pytest.mark.asyncio
async def test_run_judge_handles_not_found_error(mock_api_client: MagicMock) -> None:
    """Test handling of 404 errors in run_judge."""
    service = JudgeService()
    mock_api_client.run_judge.side_effect = ScorableAPIError(
        status_code=404, detail="Judge not found"
    )

    request = RunJudgeRequest(
        judge_id="nonexistent-id",
        judge_name="Test Judge",
        request="Test request",
        response="Test response",
    )

    with pytest.raises(RuntimeError) as excinfo:
        await service.run_judge(request)

    assert "Judge execution failed" in str(excinfo.value)
    assert "Judge not found" in str(excinfo.value)


@pytest.mark.asyncio
async def test_run_judge_handles_validation_error(mock_api_client: MagicMock) -> None:
    """Test handling of ResponseValidationError in run_judge."""
    service = JudgeService()
    mock_api_client.run_judge.side_effect = ResponseValidationError(
        "Missing required field: 'score'", {"evaluator_name": "Test Evaluator"}
    )

    request = RunJudgeRequest(
        judge_id="judge-123",
        judge_name="Test Judge",
        request="Test request",
        response="Test response",
    )

    with pytest.raises(RuntimeError) as excinfo:
        await service.run_judge(request)

    assert "Invalid judge response" in str(excinfo.value)
    assert "Missing required field" in str(excinfo.value)
