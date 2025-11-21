"""Scorable evaluator service module.

This module handles the integration with Scorable evaluators.
"""

import logging

from scorable_mcp.root_api_client import (
    ResponseValidationError,
    ScorableAPIError,
    ScorableEvaluatorRepository,
)
from scorable_mcp.schema import (
    EvaluationRequest,
    EvaluationRequestByName,
    EvaluationResponse,
    EvaluatorInfo,
    EvaluatorsListResponse,
)
from scorable_mcp.settings import settings

logger = logging.getLogger("scorable_mcp.evaluator")


class EvaluatorService:
    """Service for interacting with Scorable evaluators."""

    def __init__(self) -> None:
        """Initialize the evaluator service."""
        self.async_client = ScorableEvaluatorRepository(
            api_key=settings.scorable_api_key.get_secret_value(),
            base_url=settings.scorable_api_url,
        )

    async def fetch_evaluators(self, max_count: int | None = None) -> list[EvaluatorInfo]:
        """Fetch available evaluators from the API.

        Args:
            max_count: Maximum number of evaluators to fetch

        Returns:
            List[EvaluatorInfo]: List of evaluator information.

        Raises:
            RuntimeError: If evaluators cannot be retrieved from the API.
        """
        logger.info(
            f"Fetching evaluators from Scorable API (max: {max_count or settings.max_evaluators})"
        )

        try:
            evaluators_data = await self.async_client.list_evaluators(max_count)

            total = len(evaluators_data)
            logger.info(f"Retrieved {total} evaluators from Scorable API")

            return evaluators_data

        except ScorableAPIError as e:
            logger.error(f"Failed to fetch evaluators from API: {e}", exc_info=settings.debug)
            raise RuntimeError(f"Cannot fetch evaluators: {str(e)}") from e
        except ResponseValidationError as e:
            logger.error(f"Response validation error: {e}", exc_info=settings.debug)
            if e.response_data:
                logger.debug(f"Response data: {e.response_data}")
            raise RuntimeError(f"Invalid evaluators response: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error fetching evaluators: {e}", exc_info=settings.debug)
            raise RuntimeError(f"Cannot fetch evaluators: {str(e)}") from e

    async def list_evaluators(self, max_count: int | None = None) -> EvaluatorsListResponse:
        """List all available evaluators.

        Args:
            max_count: Maximum number of evaluators to fetch

        Returns:
            EvaluatorsListResponse: A response containing all available evaluators.
        """
        evaluators = await self.fetch_evaluators(max_count)

        return EvaluatorsListResponse(evaluators=evaluators)

    async def get_evaluator_by_id(self, evaluator_id: str) -> EvaluatorInfo | None:
        """Get evaluator details by ID.

        Args:
            evaluator_id: The ID of the evaluator to retrieve.

        Returns:
            Optional[EvaluatorInfo]: The evaluator details or None if not found.
        """
        evaluators = await self.fetch_evaluators()

        for evaluator in evaluators:
            if evaluator.id == evaluator_id:
                return evaluator

        return None

    async def run_evaluation(self, request: EvaluationRequest) -> EvaluationResponse:
        """Run a standard evaluation asynchronously.

        This method is used by the SSE server which requires async operation.

        Args:
            evaluator_id: The ID of the evaluator to use.
            request: The evaluation request parameters.

        Returns:
            EvaluationResponse: The evaluation results.
        """
        try:
            result = await self.async_client.run_evaluator(
                evaluator_id=request.evaluator_id,
                request=request.request,
                response=request.response,
                contexts=request.contexts,
                expected_output=request.expected_output,
            )

            return result
        except ScorableAPIError as e:
            logger.error(f"API error running evaluation: {e}", exc_info=settings.debug)
            raise RuntimeError(f"Failed to run evaluation: {str(e)}") from e
        except ResponseValidationError as e:
            logger.error(f"Response validation error: {e}", exc_info=settings.debug)
            if e.response_data:
                logger.debug(f"Response data: {e.response_data}")
            raise RuntimeError(f"Invalid evaluation response: {str(e)}") from e
        except Exception as e:
            logger.error(f"Error running evaluation: {e}", exc_info=settings.debug)
            raise RuntimeError(f"Failed to run evaluation: {str(e)}") from e

    async def run_evaluation_by_name(self, request: EvaluationRequestByName) -> EvaluationResponse:
        """Run a standard evaluation using the evaluator's name instead of ID.

        Args:
            request: The evaluation request parameters.
                    The evaluator_id field will be treated as the evaluator name.

        Returns:
            EvaluationResponse: The evaluation results.
        """
        try:
            result = await self.async_client.run_evaluator_by_name(
                evaluator_name=request.evaluator_name,
                request=request.request,
                response=request.response,
                contexts=request.contexts,
                expected_output=request.expected_output,
            )

            return result
        except ScorableAPIError as e:
            logger.error(f"API error running evaluation by name: {e}", exc_info=settings.debug)
            raise RuntimeError(f"Failed to run evaluation by name: {str(e)}") from e
        except ResponseValidationError as e:
            logger.error(f"Response validation error: {e}", exc_info=settings.debug)
            if e.response_data:
                logger.debug(f"Response data: {e.response_data}")
            raise RuntimeError(f"Invalid evaluation response: {str(e)}") from e
        except Exception as e:
            logger.error(f"Error running evaluation by name: {e}", exc_info=settings.debug)
            raise RuntimeError(f"Failed to run evaluation by name: {str(e)}") from e
