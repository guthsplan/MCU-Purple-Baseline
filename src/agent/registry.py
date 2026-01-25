from __future__ import annotations

from typing import Any, Optional

def build_agent(
    agent_name: str,
    *,
    device: Optional[str] = None,
    **kwargs: Any,
):
    """
    Build and return a NEW agent instance.

    Parameters
    ----------
    agent_name : str
        Agent identifier string.
        Supported values:
          - "noop"
          - "vpt"
          - "steve1"
          - "rocket1"
          - "groot1"
          - "llm"

    device : Optional[str]
        Device spec passed through to the agent (e.g. "cpu", "cuda", "cuda:0").
        Interpretation is agent-specific.

    Returns
    -------
    BaseAgent
        A fresh agent instance with isolated runtime state.

    Raises
    ------
    ValueError
        If agent_name is unknown.
    """

    name = (agent_name or "noop").lower()
    print(f"[BUILD_AGENT] name={name}, device={device}")
    
    # NoOp agent (debug / conformance / sanity checks)
    if name == "noop":
        from src.agent.noop import NoOpAgent
        return NoOpAgent(device=device, **kwargs)

    # VPT baseline agent
    if name == "vpt":
        from src.agent.vpt.agent import VPTAgent
        return VPTAgent(device=device, **kwargs)

    # STEVE-1 agent
    if name == "steve1":
        from src.agent.steve1.agent import Steve1Agent
        return Steve1Agent(device=device, **kwargs)

    # Rocket-1 (MineStudio) agent
    if name == "rocket1":
        from src.agent.rocket1.agent import Rocket1Agent
        return Rocket1Agent(device=device, **kwargs)

    # Groot-1 (Video-conditioned) agent
    if name == "groot1":
        from src.agent.groot1.agent import Groot1Agent
        return Groot1Agent(device=device, **kwargs)

    # LLM-based agent (planning / reasoning baselines)
    if name == "llm": 
        from src.agent.llm.agent import LLMAgent
        return LLMAgent(device=device, **kwargs)

    # ------------------------------------------------------------------
    # Unknown agent
    # ------------------------------------------------------------------
    raise ValueError(
        f"Unknown agent: {agent_name!r}. "
        f"Supported agents: noop, vpt, steve1, rocket1, groot1, llm."
    )
