from __future__ import annotations

from typing import Dict

from src.server.session_manager import SessionState


class NoOpAgent:
    """
    No-op purple policy.

    - 아무 행동도 하지 않는다.
    - action shape만 MineStudio env.step(action)에 맞춰 보장한다.
    """

    def act(self, *, obs_base64: str, session: SessionState) -> Dict:
        """
        Args:
            obs_base64: base64-encoded image (unused in noop)
            session: SessionState (action shape 정보 포함)

        Returns:
            dict with keys:
              - buttons: list[int]
              - camera: list[float]
        """

        num_buttons = session.expected_num_buttons
        camera_dims = session.expected_camera_dims

        return {
            "buttons": [0] * num_buttons,
            "camera": [0.0] * camera_dims,
        }
