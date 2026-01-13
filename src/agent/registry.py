from __future__ import annotations

from typing import Any, Optional

from src.agent.noop import NoOpAgent
from src.agent.vpt.agent import VPTAgent
from src.agent.steve1.agent import Steve1Agent
from src.agent.rocket1.agent import Rocket1Agent
from src.agent.llm.agent import LLMAgent


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

    
    # NoOp agent (debug / conformance / sanity checks)
    if name == "noop":
        return NoOpAgent(device=device, **kwargs)

    # VPT baseline agent
    if name == "vpt":
        return VPTAgent(device=device, **kwargs)

    # STEVE-1 agent
    if name == "steve1":
        return Steve1Agent(device=device, **kwargs)

    # Rocket-1 (MineStudio) agent
    if name == "rocket1":
        return Rocket1Agent(device=device, **kwargs)

    # LLM-based agent (planning / reasoning baselines)
    if name == "llm": 
        return LLMAgent(device=device, **kwargs)

    # ------------------------------------------------------------------
    # Unknown agent
    # ------------------------------------------------------------------
    raise ValueError(
        f"Unknown agent: {agent_name!r}. "
        f"Supported agents: noop, vpt, steve1, rocket1, llm."
    )
