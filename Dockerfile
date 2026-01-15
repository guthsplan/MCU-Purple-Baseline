# ===== MCU Purple Agent Dockerfile =====
FROM ghcr.io/astral-sh/uv:python3.11-trixie

# system deps
RUN apt-get update && apt-get install -y \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN adduser agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml uv.lock README.md ./
COPY src src

# deps
RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

# pre-download VPT models from Hugging Face during build
RUN \
    --mount=type=cache,target=/home/agent/.cache,uid=1000 \
    uv run python -m src.agent.download --agent vpt --device cpu

EXPOSE 9009
ENTRYPOINT ["uv", "run", "python", "-m", "src.server.app"]
CMD ["--host", "0.0.0.0", "--port", "9009", "--model", "vpt"]