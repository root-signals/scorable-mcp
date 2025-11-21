"""Settings module for the Scorable MCP Server.

This module provides a settings model for the unified server using pydantic-settings.
"""

import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_package_version() -> str:
    """Get the version of the root-mcp-server package from pyproject.toml.

    Returns:
        The package version or a default value if not found
    """
    current_dir = Path(__file__).parent
    for _ in range(4):
        pyproject_path = current_dir / "pyproject.toml"
        if pyproject_path.exists():
            try:
                content = pyproject_path.read_text()
                version_match = re.search(r'version\s*=\s*"([^"]+)"', content)
                if version_match:
                    return version_match.group(1)
            except Exception:
                pass
        current_dir = current_dir.parent

    return "dev-version"


class Settings(BaseSettings):
    """Settings for the Scorable MCP Server.

    This class handles loading and validating configuration from environment variables.
    """

    scorable_api_key: SecretStr = Field(
        default=...,
        description="Scorable API key for authentication",
    )
    scorable_api_url: str = Field(
        default="https://api.scorable.ai",
        description="Scorable API URL",
    )
    scorable_api_timeout: float = Field(
        default=30.0,
        description="Timeout in seconds for Scorable API requests",
    )
    max_evaluators: int = Field(
        default=40,
        description="Maximum number of evaluators to fetch",
    )
    max_judges: int = Field(
        default=40,
        description="Maximum number of judges to fetch",
    )
    show_public_judges: bool = Field(
        default=False,
        description="Whether to show public judges",
    )
    version: str = Field(
        default_factory=get_package_version,
        description="Package version from pyproject.toml",
    )

    coding_policy_evaluator_id: str = Field(
        default="4613f248-b60e-403a-bcdc-157d1c44194a",
        description="Scorable evaluator ID for coding policy evaluation",
    )

    coding_policy_evaluator_request: str = Field(
        default="Is the response written according to the coding policy?",
        description="Request for the coding policy evaluation",
    )

    host: str = Field(default="0.0.0.0", description="Host to bind to", alias="HOST")
    port: int = Field(default=9090, description="Port to listen on", alias="PORT")
    log_level: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="info", description="Logging level", alias="LOG_LEVEL"
    )
    debug: bool = Field(default=False, description="Enable debug mode", alias="DEBUG")

    transport: Literal["stdio", "sse", "websocket"] = Field(
        default="sse",
        description="Transport mechanism to use (stdio, sse, websocket)",
        alias="TRANSPORT",
    )

    env: str = Field(
        default="development",
        description="Environment identifier (development, staging, production)",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )


try:
    settings = Settings()
except Exception as e:
    sys.stderr.write(f"Error loading settings: {str(e)}\n")
    sys.stderr.write("Check that your .env file exists with proper SCORABLE_API_KEY\n")
    raise
