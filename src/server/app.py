from __future__ import annotations

import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from src.server.executor import Executor
from src.server.session_manager import SessionManager


def main():
    parser = argparse.ArgumentParser(description="Run the purple agent for MCU benchmark.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9008)
    parser.add_argument("--card-url", type=str, default=None, help="Public URL to advertise in agent card")
    parser.add_argument("--agent", type=str, default="noop", help="Policy agent name: noop | rocket1 | vpt ...")
    args = parser.parse_args()

    public_url = args.card_url or f"http://{args.host}:{args.port}/"

    skill = AgentSkill(
        id="mcu-purple-policy",
        name="MCU Purple Policy",
        description="Responds to init/obs with ack/action payloads for MineStudio MCU benchmark.",
        tags=["mcu", "minecraft", "minestudio", "purple"],
        examples=[],
    )

    agent_card = AgentCard(
        name="MCU Purple Baseline",
        description="Purple policy server for MCU AgentBeats (MineStudio).",
        url=public_url,
        version="0.1.0",
        default_input_modes=["text", "application/json"],
        default_output_modes=["text", "application/json"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    sessions = SessionManager()
    executor = Executor(sessions=sessions, agent_name=args.agent)

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
