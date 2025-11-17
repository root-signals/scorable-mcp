"""SSE transport for the Scorable MCP Server.

This module provides a dedicated implementation of the MCP server using
Server-Sent Events (SSE) transport for network/Docker environments.
"""

import logging
import os
import sys
from typing import Any

import uvicorn
from mcp import Tool
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route

from scorable_mcp.core import RootMCPServerCore
from scorable_mcp.settings import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("scorable_mcp.sse")


class SSEMCPServer:
    """MCP server implementation with SSE transport for Docker/network environments."""

    def __init__(self) -> None:
        """Initialize the SSE-based MCP server."""

        self.core = RootMCPServerCore()

        # For backward-comp
        self.app = self.core.app
        self.evaluator_service = self.core.evaluator_service

    async def list_tools(self) -> list[Tool]:
        return await self.core.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> list[TextContent]:
        return await self.core.call_tool(name, arguments)


def create_app(server: SSEMCPServer) -> Starlette:
    """Create a Starlette app with SSE routes.

    Includes the /sse endpoint from <1.5.0 for backward compatibility and the identical /mcp endpoint.
    """
    sse_transport = SseServerTransport("/sse/message/")
    mcp_transport = SseServerTransport("/mcp/message/")

    async def _run_server_app(
        request: Request, transport: SseServerTransport
    ) -> Any:  # pragma: no cover – trivial helper
        """Internal helper to bridge ASGI request with a given SSE transport."""
        logger.debug("SSE connection initiated")
        try:
            async with transport.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await server.app.run(
                    streams[0], streams[1], server.app.create_initialization_options()
                )
        except Exception as exc:
            logger.error("Error handling SSE/MCP connection", exc_info=True)
            return Response(f"Error: {exc}", status_code=500)

    async def handle_sse(request: Request) -> Any:  # /sse
        return await _run_server_app(request, sse_transport)

    async def handle_mcp(request: Request) -> Any:  # /mcp
        return await _run_server_app(request, mcp_transport)

    routes = [
        Route("/sse", endpoint=handle_sse),
        Mount("/sse/message/", app=sse_transport.handle_post_message),
        Route("/mcp", endpoint=handle_mcp),
        Mount("/mcp/message/", app=mcp_transport.handle_post_message),
        Route("/health", endpoint=lambda r: Response("OK", status_code=200)),
    ]

    return Starlette(routes=routes)


def run_server(host: str = "0.0.0.0", port: int = 9090) -> None:
    """Run the MCP server with SSE transport."""

    server = SSEMCPServer()

    app = create_app(server)
    logger.info(f"SSE server listening on http://{host}:{port}/sse")
    uvicorn.run(app, host=host, port=port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    try:
        host = os.environ.get("HOST", settings.host)
        port = int(os.environ.get("PORT", settings.port))

        logger.info("Starting Scorable MCP Server")
        logger.info(f"Targeting API: {settings.scorable_api_url}")
        logger.info(f"Environment: {settings.env}")
        logger.info(f"Transport: {settings.transport}")
        logger.info(f"Host: {host}, Port: {port}")

        run_server(host=host, port=port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=settings.debug)
        sys.exit(1)
