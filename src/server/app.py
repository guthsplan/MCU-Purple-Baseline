from __future__ import annotations

import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from src.server.executor import Executor
from src.server.session_manager import SessionManager

from starlette.responses import PlainTextResponse

def main():
    parser = argparse.ArgumentParser(description="Run the purple agent for MCU benchmark.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9019)
    parser.add_argument("--card-url", type=str, default=None, help="Public URL to advertise in agent card")
    parser.add_argument("--agent", type=str, default="vpt",choices=["noop","vpt","steve1","rocket1"], help="Policy agent name (default: vpt)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--state-ttl", type=int, default=60 * 60)

    args = parser.parse_args()

    if args.card_url:
        public_url = args.card_url
    else:
        advertise_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        public_url = f"http://{advertise_host}:{args.port}/"
        
    public_url = public_url.rstrip("/") + "/"
    
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

    sessions = SessionManager(ttl_seconds=args.state_ttl)

    executor = Executor(
        sessions=sessions,
        agent_name=args.agent,
        debug=args.debug,
        device=args.device,
        state_ttl_seconds=args.state_ttl,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    asgi_app = app.build()

    @asgi_app.route("/health")
    async def health(request):
        return PlainTextResponse("OK")
    
    uvicorn.run(asgi_app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
