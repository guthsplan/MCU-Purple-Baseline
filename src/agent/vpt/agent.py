# src/agent/vpt/agent.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import logging
import numpy as np
import torch

from minestudio.models.vpt import VPTPolicy  # 네가 쓰던 import 유지

from src.agent.base import BaseAgent
from .model import VPTState
from .preprocess import build_vpt_obs
from src.action.action_space import build_vpt_action_space

logger = logging.getLogger("purple.vpt")


class VPTAgent(BaseAgent):
    """
    VPT wrapper for Purple baseline.

    핵심:
      - VPTPolicy에는 MineStudio가 기대하는 action_space만 사용 (Discrete 금지)
      - get_action() 결과(토큰)를 MineRL-style dict로 디코드해서 pipeline에 넘김
    """

    def __init__(
        self,
        device: Optional[str] = None,
        hf_id_rl: str = "CraftJarvis/MineStudio_VPT.rl_from_early_game_2x",
        hf_id_fallback: str = "CraftJarvis/MineStudio_VPT.foundation_2x",
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(device=device)

        self.hf_id_rl = hf_id_rl
        self.hf_id_fallback = hf_id_fallback

        # ✅ VPTPolicy 전용 action space (MultiDiscrete)
        self.action_space = build_vpt_action_space()

        self.model = self._load_model().to(self.device)
        self.model.eval()

        # ✅ action decoder 준비 (없으면 안전하게 noop로 떨어뜨리게 할 것)
        self._decoder = self._build_decoder()

        logger.info("VPTAgent loaded on device=%s", self.device)

    def _load_model(self) -> VPTPolicy:
        """
        중요:
          - action_space는 VPT 전용(MultiDiscrete)만
          - Discrete/Box 기반 MineRL space 절대 금지
        """
        try:
            logger.info("Loading VPT RL model: %s", self.hf_id_rl)
            return VPTPolicy.from_pretrained(
                self.hf_id_rl,
                action_space=self.action_space,
            )
        except Exception as e:
            logger.warning("Failed to load RL VPT model (%s): %s", self.hf_id_rl, e)
            logger.info("Loading VPT fallback model: %s", self.hf_id_fallback)
            return VPTPolicy.from_pretrained(
                self.hf_id_fallback,
                action_space=self.action_space,
            )

    def _build_decoder(self):
        """
        MineStudio vpt_lib.actions 안에 있는 공식 디코더를 최대한 활용한다.
        환경/버전에 따라 API가 다를 수 있어서 reflection으로 안전하게 찾는다.
        """
        try:
            import minestudio.utils.vpt_lib.actions as actions
        except Exception as e:
            logger.warning("MineStudio actions module import failed: %s", e)
            return None

        # 1) module-level decode 함수가 있으면 우선
        for fn_name in ("to_env_action", "decode_action", "tokens_to_action", "convert_to_env_action"):
            fn = getattr(actions, fn_name, None)
            if callable(fn):
                logger.info("Using MineStudio action decoder function: %s", fn_name)
                return fn

        # 2) ActionTransformer 류 클래스가 있으면 인스턴스화
        cls = getattr(actions, "ActionTransformer", None)
        if cls is not None:
            try:
                inst = cls()
                # 흔한 메서드명들
                for m in ("to_env_action", "decode_action", "decode", "transform"):
                    if hasattr(inst, m) and callable(getattr(inst, m)):
                        logger.info("Using MineStudio ActionTransformer.%s", m)
                        return getattr(inst, m)
            except Exception as e:
                logger.warning("ActionTransformer init failed: %s", e)

        logger.warning("No MineStudio action decoder found; VPT actions will likely noop.")
        return None

    def reset(self) -> None:
        return

    def initial_state(self, task_text: Optional[str] = None) -> VPTState:
        return VPTState(memory=None, first=True)

    @torch.inference_mode()
    def _act_impl(
        self,
        obs: Dict[str, Any],
        state: VPTState,
        deterministic: bool = True,
    ) -> Tuple[Dict[str, Any], VPTState]:

        vpt_obs = build_vpt_obs(obs)  # {"image": HWC uint8} 형태

        # VPTPolicy.get_action -> {"buttons": token, "camera": token} + new_memory
        action, new_memory = self.model.get_action(
            vpt_obs,
            state.memory,
            input_shape="*",
        )

        new_state = VPTState(memory=new_memory, first=False)
        action["buttons"] = action["buttons"].cpu().numpy().tolist()
        action["camera"] = action["camera"].cpu().numpy().tolist()
        return action, new_state

    def _decode_to_env_action(self, action_tokens: Dict[str, Any]) -> Dict[str, Any]:
        """
        action_tokens:
          - {"buttons": <0..8640>, "camera": <0..120>} 형태가 정상

        반환:
          - MineRL-style dict: {"attack":0/1, ..., "camera":[pitch,yaw], ...}
          - 디코드 실패 시 {"camera":[0,0]} + 나머지 0으로 만드는 게 안전
        """
        if self._decoder is None:
            return {"camera": [0.0, 0.0]}

        # 텐서/넘파이 정리 (가능한 범위에서)
        try:
            def _to_py(x):
                if hasattr(x, "detach") and hasattr(x, "cpu"):
                    x = x.detach().cpu()
                if hasattr(x, "item") and x.numel() == 1:
                    return x.item()
                if hasattr(x, "tolist"):
                    return x.tolist()
                return x

            tokens = {k: _to_py(v) for k, v in action_tokens.items()} if isinstance(action_tokens, dict) else action_tokens
        except Exception:
            tokens = action_tokens

        try:
            out = self._decoder(tokens)
            if isinstance(out, dict):
                return out
            # 일부 구현은 {"action": {...}}로 감싸기도 함
            if isinstance(out, dict) and "action" in out and isinstance(out["action"], dict):
                return out["action"]
        except Exception as e:
            logger.warning("VPT action decode failed: %s", e)

        return {"camera": [0.0, 0.0]}
