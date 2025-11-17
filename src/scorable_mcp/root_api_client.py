"""Scorable HTTP client module.

This module provides a simple httpx-based client for the Scorable API,
replacing the official SDK with a minimal implementation for our specific needs.
"""

import logging
from datetime import datetime
from typing import Any, Literal, cast

import httpx

from scorable_mcp.schema import (
    EvaluationResponse,
    EvaluatorInfo,
    JudgeInfo,
    RunJudgeRequest,
    RunJudgeResponse,
)
from scorable_mcp.settings import settings

logger = logging.getLogger("scorable_mcp.root_client")


class ScorableAPIError(Exception):
    """Exception raised for Scorable API errors."""

    def __init__(self, status_code: int, detail: str):
        """Initialize ScorableAPIError.

        Args:
            status_code: HTTP status code of the error
            detail: Error message
        """
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Scorable API error (HTTP {status_code}): {detail}")


class ResponseValidationError(Exception):
    """Exception raised when API response doesn't match expected schema."""

    def __init__(self, message: str, response_data: Any | None = None):
        """Initialize ResponseValidationError.

        Args:
            message: Error message
            response_data: The response data that failed validation
        """
        self.response_data = response_data
        super().__init__(f"Response validation error: {message}")


class ScorableRepositoryBase:
    """Base class for Scorable API clients."""

    def __init__(
        self,
        api_key: str = settings.scorable_api_key.get_secret_value(),
        base_url: str = settings.scorable_api_url,
    ):
        """Initialize the HTTP client for Scorable API.

        Args:
            api_key: Scorable API key
            base_url: Base URL for the Scorable API
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

        self.headers = {
            "Authorization": f"Api-Key {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"scorable-mcp/{settings.version}",
        }

        logger.debug(
            f"Initialized Scorable API client with User-Agent: {self.headers['User-Agent']}"
        )

    async def _make_request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> Any:
        """Make an HTTP request to the Scorable API.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path
            params: URL parameters
            json_data: JSON body data for POST/PUT requests

        Returns:
            Response data as a dictionary or list

        Raises:
            ScorableAPIError: If the API returns an error
        """
        url = f"{self.base_url}/{path.lstrip('/')}"

        logger.debug(f"Making {method} request to {url}")
        if settings.debug:
            logger.debug(f"Request headers: {self.headers}")
            if params:
                logger.debug(f"Request params: {params}")
            if json_data:
                logger.debug(f"Request payload: {json_data}")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    headers=self.headers,
                    timeout=settings.scorable_api_timeout,
                )

                logger.debug(f"Response status: {response.status_code}")
                if settings.debug:
                    logger.debug(f"Response headers: {dict(response.headers)}")

                if response.status_code >= 400:  # noqa: PLR2004
                    try:
                        error_data = response.json()
                        error_message = error_data.get("detail", str(error_data))
                    except Exception:
                        error_message = response.text or f"HTTP {response.status_code}"

                    logger.error(f"API error response: {error_message}")
                    raise ScorableAPIError(response.status_code, error_message)

                if response.status_code == 204:  # noqa: PLR2004
                    return {}

                response_data = response.json()
                if settings.debug:
                    logger.debug(f"Response data: {response_data}")
                return response_data

            except httpx.RequestError as e:
                logger.error(f"Request error: {str(e)}")
                raise ScorableAPIError(0, f"Connection error: {str(e)}") from e

    async def _fetch_paginated_results(  # noqa: PLR0915, PLR0912
        self,
        initial_url: str,
        max_to_fetch: int,
        resource_type: Literal["evaluators", "judges"],
        url_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:  # noqa: PLR0915, PLR0912
        items_raw: list[dict[str, Any]] = []
        next_page_url = initial_url

        while next_page_url and len(items_raw) < max_to_fetch:
            if next_page_url.startswith("http"):
                next_page_url = "/" + next_page_url.split("/", 3)[3]

            response = await self._make_request("GET", next_page_url)
            logger.debug(f"Raw {resource_type} response: {response}")

            if isinstance(response, dict):
                next_page_url = response.get("next", "")

                # Preserve any specified URL parameters
                if next_page_url and url_params:
                    for param_name, param_value in url_params.items():
                        if param_value is not None and f"{param_name}=" not in next_page_url:
                            if "?" in next_page_url:
                                next_page_url += f"&{param_name}={param_value}"
                            else:
                                next_page_url += f"?{param_name}={param_value}"

                if "results" in response and isinstance(response["results"], list):
                    current_page_items = response["results"]
                    logger.debug(
                        f"Found {len(current_page_items)} {resource_type} in 'results' field"
                    )
                else:
                    raise ResponseValidationError(
                        "Could not find 'results' field in response", response
                    )
            elif isinstance(response, list):
                logger.debug(f"Response is a direct list of {resource_type}")
                current_page_items = response
                next_page_url = ""
            else:
                raise ResponseValidationError(
                    f"Expected response to be a dict or list, got {type(response).__name__}",
                    cast(dict[str, Any], response),
                )

            items_raw.extend(current_page_items)
            logger.info(
                f"Fetched {len(current_page_items)} more {resource_type}, total now: {len(items_raw)}"
            )

            if len(current_page_items) == 0:
                logger.debug("Received empty page, stopping pagination")
                break

        if len(items_raw) > max_to_fetch:
            items_raw = items_raw[:max_to_fetch]
            logger.debug(f"Trimmed results to {max_to_fetch} {resource_type}")

        logger.info(f"Found {len(items_raw)} {resource_type} total after pagination")
        return items_raw


class ScorableEvaluatorRepository(ScorableRepositoryBase):
    """HTTP client for the Scorable Evaluators API."""

    async def list_evaluators(self, max_count: int | None = None) -> list[EvaluatorInfo]:
        """List all available evaluators with pagination support.

        Args:
            max_count: Maximum number of evaluators to fetch (defaults to settings.max_evaluators)

        Returns:
            List of evaluator information

        Raises:
            ResponseValidationError: If a required field is missing in any evaluator
        """
        max_to_fetch = max_count if max_count is not None else settings.max_evaluators
        page_size = min(max_to_fetch, 40)
        initial_url = f"/v1/evaluators?page_size={page_size}"

        evaluators_raw = await self._fetch_paginated_results(
            initial_url=initial_url,
            max_to_fetch=max_to_fetch,
            resource_type="evaluators",
        )

        evaluators = []
        for i, evaluator_data in enumerate(evaluators_raw):
            try:
                logger.debug(f"Processing evaluator {i}: {evaluator_data}")

                id_value = evaluator_data["id"]
                name_value = evaluator_data["name"]
                created_at = evaluator_data["created_at"]

                if isinstance(created_at, datetime):
                    created_at = created_at.isoformat()

                intent = None
                if "objective" in evaluator_data and isinstance(evaluator_data["objective"], dict):
                    objective = evaluator_data["objective"]
                    intent = objective.get("intent")

                inputs = evaluator_data["inputs"]

                evaluator = EvaluatorInfo(
                    id=id_value,
                    name=name_value,
                    created_at=created_at,
                    intent=intent,
                    inputs=inputs,
                )
                evaluators.append(evaluator)
            except KeyError as e:
                missing_field = str(e).strip("'")
                logger.warning(f"Evaluator at index {i} missing required field: '{missing_field}'")
                logger.warning(f"Evaluator data: {evaluator_data}")
                raise ResponseValidationError(
                    f"Evaluator at index {i} missing required field: '{missing_field}'",
                    evaluator_data,
                ) from e

        return evaluators

    async def run_evaluator(
        self,
        evaluator_id: str,
        request: str,
        response: str,
        contexts: list[str] | None = None,
        expected_output: str | None = None,
    ) -> EvaluationResponse:
        """Run an evaluation with the specified evaluator.

        Args:
            evaluator_id: ID of the evaluator to use
            request: User query/request to evaluate
            response: Model's response to evaluate
            contexts: Optional list of context passages for RAG evaluations
            expected_output: Optional expected output for reference-based evaluations

        Returns:
            Evaluation response with score and justification

        Raises:
            ResponseValidationError: If the response is missing required fields
        """
        payload: dict[str, Any] = {
            "request": request,
            "response": response,
        }

        if contexts:
            payload["contexts"] = contexts

        if expected_output:
            payload["expected_output"] = expected_output

        response_data = await self._make_request(
            "POST", f"/v1/evaluators/execute/{evaluator_id}/", json_data=payload
        )

        logger.debug(f"Raw evaluation response: {response_data}")

        try:
            result_data = (
                response_data.get("result", response_data)
                if isinstance(response_data, dict)
                else response_data
            )

            return EvaluationResponse.model_validate(result_data)
        except ValueError as e:
            raise ResponseValidationError(
                f"Invalid evaluation response format: {str(e)}",
                response_data,
            ) from e

    async def run_evaluator_by_name(
        self,
        evaluator_name: str,
        request: str,
        response: str,
        contexts: list[str] | None = None,
        expected_output: str | None = None,
    ) -> EvaluationResponse:
        """Run an evaluation with an evaluator specified by name.

        Args:
            evaluator_name: Name of the evaluator to use
            request: User query/request to evaluate
            response: Model's response to evaluate
            contexts: Optional list of context passages for RAG evaluations
            expected_output: Optional expected output for reference-based evaluations

        Returns:
            Evaluation response with score and justification

        Raises:
            ResponseValidationError: If the response is missing required fields
        """
        payload: dict[str, Any] = {
            "request": request,
            "response": response,
        }

        if contexts:
            payload["contexts"] = contexts

        if expected_output:
            payload["expected_output"] = expected_output

        params = {"name": evaluator_name}

        response_data = await self._make_request(
            "POST", "/v1/evaluators/execute/by-name/", params=params, json_data=payload
        )

        logger.debug(f"Raw evaluation by name response: {response_data}")

        try:
            # Extract the result field if it exists, otherwise use the whole response
            result_data = (
                response_data.get("result", response_data)
                if isinstance(response_data, dict)
                else response_data
            )

            # Let Pydantic handle validation through the model
            return EvaluationResponse.model_validate(result_data)
        except ValueError as e:
            # Pydantic will raise ValueError for validation errors
            raise ResponseValidationError(
                f"Invalid evaluation response format: {str(e)}",
                response_data,
            ) from e


class ScorableJudgeRepository(ScorableRepositoryBase):
    """HTTP client for the Scorable Judges API."""

    async def list_judges(self, max_count: int | None = None) -> list[JudgeInfo]:
        """List all available judges with pagination support.

        Args:
            max_count: Maximum number of judges to fetch (defaults to settings.max_judges)

        Returns:
            List of judge information

        Raises:
            ResponseValidationError: If a required field is missing in any judge
        """
        max_to_fetch = max_count if max_count is not None else settings.max_judges
        page_size = min(max_to_fetch, 40)
        initial_url = f"/v1/judges?page_size={page_size}&show_global={settings.show_public_judges}"
        url_params = {"show_global": settings.show_public_judges}

        judges_raw = await self._fetch_paginated_results(
            initial_url=initial_url,
            max_to_fetch=max_to_fetch,
            resource_type="judges",
            url_params=url_params,
        )

        judges = []
        for i, judge_data in enumerate(judges_raw):
            try:
                logger.debug(f"Processing judge {i}: {judge_data}")

                id_value = judge_data["id"]
                name_value = judge_data["name"]
                created_at = judge_data["created_at"]

                if isinstance(created_at, datetime):
                    created_at = created_at.isoformat()

                description = judge_data.get("intent")

                evaluators: list[JudgeInfo.NestedEvaluatorInfo] = []
                for evaluator_data in judge_data.get("evaluators", []):
                    evaluators.append(JudgeInfo.NestedEvaluatorInfo.model_validate(evaluator_data))

                judge = JudgeInfo(
                    id=id_value,
                    name=name_value,
                    created_at=created_at,
                    description=description,
                    evaluators=evaluators,
                )
                judges.append(judge)
            except KeyError as e:
                missing_field = str(e).strip("'")
                logger.warning(f"Judge at index {i} missing required field: '{missing_field}'")
                logger.warning(f"Judge data: {judge_data}")
                raise ResponseValidationError(
                    f"Judge at index {i} missing required field: '{missing_field}'",
                    judge_data,
                ) from e

        return judges

    async def run_judge(
        self,
        run_judge_request: RunJudgeRequest,
    ) -> RunJudgeResponse:
        """Run a judge by ID.

        Args:
            run_judge_request: The judge request containing request, response, and judge ID.

        Returns:
            Evaluation result

        Raises:
            ResponseValidationError: If response cannot be parsed
            ScorableAPIError: If API returns an error
        """
        logger.info(f"Running judge {run_judge_request.judge_id}")
        logger.debug(f"Judge request: {run_judge_request.request[:100]}...")
        logger.debug(f"Judge response: {run_judge_request.response[:100]}...")

        payload = {
            "request": run_judge_request.request,
            "response": run_judge_request.response,
        }

        result = await self._make_request(
            method="POST",
            path=f"/v1/judges/{run_judge_request.judge_id}/execute/",
            json_data=payload,
        )
        try:
            return RunJudgeResponse.model_validate(result)
        except ValueError as e:
            raise ResponseValidationError(
                f"Invalid judge response format: {str(e)}",
                result,
            ) from e
