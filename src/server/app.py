# src/server/app.py
from __future__ import annotations

import argparse
import logging
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from src.server.executor import Executor
from src.server.session_manager import SessionManager

from starlette.responses import PlainTextResponse

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Run MCU Purple Baseline")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9019)
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    parser.add_argument("--agent", default="steve1")
    parser.add_argument("--device", default=None)
    parser.add_argument("--state-ttl", type=int, default=3600)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Agent Card 
    # ------------------------------------------------------------------
    skill = AgentSkill(
        id="mcu-purple-policy",
        name="MCU Purple Policy",
        description="Purple policy server for MCU AgentBeats",
        tags=["mcu", "purple"],
        examples=[],
    )

    agent_card = AgentCard(
        name="MCU Purple Baseline",
        description="Purple agent compatible with MCU Green evaluator",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="0.1.0",
        default_input_modes=["text", "application/json"],
        default_output_modes=["text", "application/json"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )

    # ------------------------------------------------------------------
    # Session + Executor
    # ------------------------------------------------------------------
    sessions = SessionManager(ttl_seconds=args.state_ttl)

    executor = Executor(
        sessions=sessions,
        agent_name=args.agent,
        device=args.device,
        state_ttl_seconds=args.state_ttl,
        debug=True,
    )

    # ------------------------------------------------------------------
    # Request handler
    # ------------------------------------------------------------------
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    # ------------------------------------------------------------------
    # A2A application
    # ------------------------------------------------------------------
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    asgi_app = app.build()

    @asgi_app.route("/health")
    async def health(request):
        return PlainTextResponse("OK")

    uvicorn.run(
        asgi_app,
        host=args.host,
        port=args.port,
        log_level="debug",
    )


if __name__ == "__main__":
    main()