FROM python:3.13-slim
LABEL maintainer="hello@scorable.ai"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv and add to PATH permanently
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md ./
COPY ./src ./src

# Server port
EXPOSE 9090

# Health check using health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f -I http://localhost:9090/health || exit 1

# Run the SSE server directly
CMD ["uv", "run", "python", "-m", "src.scorable_mcp.sse_server"]
