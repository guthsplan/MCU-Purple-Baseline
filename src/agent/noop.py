from __future__ import annotations

from typing import Dict

from src.server.session_manager import SessionState


class NoOpAgent:
    """No-op agent that returns zeroed actions."""
    def act(self, *, obs_base64: str, session: SessionState) -> Dict:
        """Return no-op action."""
        num_buttons = session.expected_num_buttons
        camera_dims = session.expected_camera_dims

        return {
            "buttons": [0] * num_buttons,
            "camera": [0.0] * camera_dims,
        }
