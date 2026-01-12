from agent.noop import NoOpAgent
from src.agent.vpt.agent import VPTAgent
from src.agent.steve1.agent import Steve1Agent
from agent.rocket1.agent import Rocket1Agent

def build_agent(agent_name: str, **kwargs):
    name = (agent_name or "noop").lower()
    if name == "noop":
        return NoOpAgent(**kwargs)
    
    if name == "vpt":
        return VPTAgent(**kwargs)
    
    if name == "steve1":
        return Steve1Agent(**kwargs)
    
    if name == "rocket1":
        return Rocket1Agent(**kwargs)
    
    raise ValueError(f"Unknown agent: {name}")
