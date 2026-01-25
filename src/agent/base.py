# src/agent/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """
    Minimal state for executor-managed sessions.

    idle_count:
      - Used by action pipeline for deterministic anti-idle (optional but recommended)
    """
    memory: Optional[Any] = None
    first: bool = False
    idle_count: int = 0


class BaseAgent(ABC):
    """
    Baseline Purple Agent Contract.

    - _act_impl(): may return ANY shape (dict/tensor/token/etc.)
    - act(): NEVER raises, returns (policy_out, new_state)
    """

    def __init__(self, device: Optional[str] = None):
        self._device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def reset(self) -> None:
        pass

    def initial_state(self, task_text: Optional[str] = None) -> AgentState:
        return AgentState(memory=None, first=True, idle_count=0)

    def act(
        self,
        obs: Dict[str, Any],
        state: AgentState,
        deterministic: bool = False,
    ) -> Tuple[Any, AgentState]:
        """
        Public entrypoint called by Executor.

        Returns:
          - policy_out: ANY (dict/tensor/token/None)
          - new_state: AgentState

        Safety:
          - NEVER raises
        """
        try:
            logger.info("[ACT] agent=%s first=%s", self.__class__.__name__, state.first)
            policy_out, new_state = self._act_impl(obs=obs, state=state, deterministic=deterministic)
            if not isinstance(new_state, AgentState):
                # harden: if policy forgot state type
                new_state = state
            new_state.first = False
            return policy_out, new_state
        except Exception as e:
            logger.exception("Agent.act failed -> policy_out=None. err=%s", e)
            # return safe sentinel; pipeline will noop
            state.first = False
            return None, state

    @abstractmethod
    def _act_impl(
        self,
        obs: Dict[str, Any],
        state: AgentState,
        deterministic: bool = True,
    ) -> Tuple[Any, AgentState]:
        raise NotImplementedError
