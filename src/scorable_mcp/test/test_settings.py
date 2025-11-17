"""Tests for the settings module."""

import re

from scorable_mcp.settings import get_package_version, settings


def test_version_in_settings() -> None:
    """Test that the version is properly set in settings."""
    assert settings.version, "Version is not set in settings"
    assert isinstance(settings.version, str), "Version should be a string"

    direct_version = get_package_version()
    assert settings.version == direct_version, (
        "Version in settings doesn't match get_package_version()"
    )


def test_get_package_version() -> None:
    """Test that the package version can be retrieved."""
    version = get_package_version()
    assert version, "Failed to get package version"
    assert isinstance(version, str), "Version should be a string"

    if version != "dev-version":
        is_date_based = bool(re.match(r"^2\d{7}-\d+$", version))

        assert is_date_based, f"Version format is unexpected, looking for YYYYMMDD-n: {version}"
