"""Integration layer between Scorable *transport-agnostic* core and the upstream FastMCP
server implementation.

The stock FastMCP class provides the full MCP protocol plumbing (handshake,
stream management, etc.) but knows nothing about our domain-specific tools.

This adapter subclasses FastMCP so we can plug in our :class:`~scorable_mcp.core.RootMCPServerCore`
implementation while still re-using all the upstream functionality.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, Tool

from scorable_mcp.core import RootMCPServerCore

logger = logging.getLogger("scorable_mcp.fastmcp_adapter")


class ScorableFastMCP(FastMCP):
    """FastMCP subclass that delegates *tool* handling to :class:`RootMCPServerCore`."""

    def __init__(self, core: RootMCPServerCore, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        """Create a FastMCP server wired up to *core*.

        Parameters
        ----------
        core
            The transport-agnostic server core responsible for actual business
            logic (tool registration, validation, evaluator calls, …).
        *args, **kwargs
            Forwarded verbatim to :class:`~mcp.server.fastmcp.FastMCP`.
        """

        self._core = core
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # MCP protocol handlers – override built-in FastMCP implementations so
    # they forward to ``RootMCPServerCore`` instead of the internal tool
    # manager. This means we do **not** have to register each tool
    # individually with FastMCP; the core remains single source of truth.
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[Tool]:  # type: ignore[override]
        """Return the list of tools exposed by the Scorable server."""
        return await self._core.list_tools()

    async def call_tool(  # type: ignore[override]
        self, name: str, arguments: dict[str, Any]
    ) -> Sequence[TextContent]:
        """Validate arguments & dispatch *name* via the server core."""
        return await self._core.call_tool(name, arguments)
