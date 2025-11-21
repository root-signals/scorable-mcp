"""Common pytest configuration and fixtures for tests."""

import logging
import os
import time
from collections.abc import Generator
from http import HTTPStatus
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from python_on_whales import Container, DockerClient

from scorable_mcp.sse_server import SSEMCPServer

# Setup logging
logger = logging.getLogger("scorable_mcp_tests")
logger.setLevel(logging.DEBUG)
log_handler = logging.StreamHandler()
log_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)

docker = DockerClient()
PROJECT_ROOT = Path(__file__).parents[3]

# Constants
MAX_HEALTH_RETRIES = 15
RETRY_DELAY_SECONDS = 3
HEALTH_CHECK_TIMEOUT = 5
HEALTH_ENDPOINT = "http://localhost:9090/health"


def check_docker_running() -> None:
    """Verify that Docker is running and available."""
    try:
        info = docker.info()
        logger.info(f"Docker is running, version: {info.server_version}")
    except Exception as e:
        logger.error(f"Docker is not running: {e}")
        pytest.skip("Docker is not running")


def cleanup_existing_containers() -> None:
    """Stop any already running Docker Compose containers."""
    try:
        containers = docker.compose.ps()
        if containers and any(c.state.running for c in containers):
            logger.info("Docker Compose service is already running, stopping it first")
            docker.compose.down(volumes=True)
            time.sleep(2)
    except Exception as e:
        logger.warning(f"Error cleaning up existing containers: {e}")


def wait_for_container_health(max_retries: int) -> bool:
    """Wait for container to report healthy status.

    Args:
        max_retries: Maximum number of retry attempts

    Returns:
        True if container became healthy, False otherwise
    """
    retries = 0

    while retries < max_retries:
        try:
            containers = docker.compose.ps()

            if not containers:
                logger.info("No containers found, waiting...")
                time.sleep(RETRY_DELAY_SECONDS)
                retries += 1
                continue

            container = containers[0]
            health_status = get_container_health_status(container)

            if health_status == "healthy":
                logger.info("Docker Compose service is healthy")
                return True

            logger.info(f"Container not healthy yet, status: {health_status}")
            time.sleep(RETRY_DELAY_SECONDS)
            retries += 1

        except Exception as e:
            logger.error(f"Error checking service health: {e}")
            time.sleep(RETRY_DELAY_SECONDS)
            retries += 1

    return False


def get_container_health_status(container: Container) -> str:
    """Get the health status of a container.

    Args:
        container: Docker container object

    Returns:
        Health status as a string or "unknown" if unavailable
    """
    if container.state and container.state.health and container.state.health.status:
        return container.state.health.status
    return "unknown"


def check_health_endpoint() -> None:
    """Check if the health endpoint is responding correctly."""
    try:
        response = httpx.get(HEALTH_ENDPOINT, timeout=HEALTH_CHECK_TIMEOUT)
        if response.status_code != HTTPStatus.OK:
            logger.error(f"Health endpoint not healthy: {response.status_code}")
            logs = docker.compose.logs()
            logger.error(f"Docker Compose logs:\n{logs}")
            raise RuntimeError(f"Health endpoint returned status code {response.status_code}")
        logger.info(f"Health endpoint response: {response.status_code}")
    except Exception as e:
        logs = docker.compose.logs()
        logger.error(f"Docker Compose logs:\n{logs}")
        raise RuntimeError("Could not connect to health endpoint") from e


@pytest_asyncio.fixture(scope="module")
async def compose_up_mcp_server() -> Generator[None]:
    """Start and stop Docker Compose for integration tests.

    Docker setup can be flaky in CI environments, so this fixture includes
    extensive health checking and error handling to make tests more reliable.

    Uses the .env file from the root directory for environment variables.
    """
    try:
        check_docker_running()
        os.chdir(PROJECT_ROOT)

        # Check if .env file exists in the project root
        env_file_path = PROJECT_ROOT / ".env"
        if not env_file_path.exists():
            logger.warning(
                f".env file not found at {env_file_path}, tests may fail if API credentials are required"
            )
        else:
            logger.info(f"Found .env file at {env_file_path}")

        cleanup_existing_containers()

        logger.info("Starting Docker Compose service")
        # The env_file is already specified in docker-compose.yml, so it will be used automatically
        docker.compose.up(detach=True)

        is_healthy = wait_for_container_health(MAX_HEALTH_RETRIES)

        if not is_healthy:
            logs = docker.compose.logs()
            logger.error(f"Docker Compose logs:\n{logs}")
            raise RuntimeError("Docker Compose service failed to start or become healthy")

        check_health_endpoint()
        time.sleep(RETRY_DELAY_SECONDS)  # Allow service to stabilize

        yield
    except Exception as e:
        logger.error(f"Failed to set up Docker Compose: {e}")
        raise
    finally:
        logger.info("Cleaning up Docker Compose service")
        try:
            docker.compose.down(volumes=True)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


@pytest_asyncio.fixture(scope="module")
async def mcp_server() -> Generator[SSEMCPServer]:
    """Create and initialize a real SSEMCPServer."""
    yield SSEMCPServer()
