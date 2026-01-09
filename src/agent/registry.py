from agent.noop import NoOpAgent
from agent.rocket1.agent import Rocket1Agent

def build_agent(name: str, **kwargs):
    if name == "noop":
        return NoOpAgent(**kwargs)
    if name == "rocket1":
        return Rocket1Agent(**kwargs)
    raise ValueError(f"Unknown agent: {name}")
