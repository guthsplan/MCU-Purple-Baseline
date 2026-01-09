from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import time
import threading

@dataclass
class SessionState:
    """
    Single session state for a context_id.
    """
    # identification
    context_id: str

    # timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # task info
    task_text: Optional[str] = None
    task_started_at: Optional[float] = None

    # observation tracking
    last_step: int = -1
    last_step_received: int = -1      
    num_obs: int = 0
    num_step_regressions: int = 0 

    # expected action shapes
    expected_num_buttons: int = 20   
    expected_camera_dims: int = 2    

    # update timestamp
    def touch(self) -> None:
        self.updated_at = time.time()


class SessionManager:
    """
    Manage multiple SessionState instances by context_id.
    """

    def __init__(self, *, ttl_seconds: Optional[int] = 60 * 60) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()

    # Get or create session by context_id
    def get_or_create(self, context_id: str) -> SessionState:
        self._gc_if_needed()

        s = self._sessions.get(context_id)
        if s is None:
            s = SessionState(context_id=context_id)
            self._sessions[context_id] = s
        s.touch()
        return s

    # Start a new task for the given context_id
    def start_new_task(
        self,
        context_id: str,
        task_text: str,
        *,
        expected_num_buttons: Optional[int] = None,
        expected_camera_dims: Optional[int] = None,
    ) -> SessionState:
        """
        initialize a new task session.
        """
        s = self.get_or_create(context_id)

        # reset task-related fields
        s.task_text = task_text
        s.task_started_at = time.time()

        s.last_step = -1
        s.last_step_received = -1
        s.num_obs = 0
        s.num_step_regressions = 0

        # allow overrides (from config or runtime)
        if expected_num_buttons is not None:
            s.expected_num_buttons = int(expected_num_buttons)
        if expected_camera_dims is not None:
            s.expected_camera_dims = int(expected_camera_dims)

        s.touch()
        return s

    def on_observation(self, context_id: str, step: int) -> SessionState:
        """
        observation received for the given context_id at step.
        Updates session state accordingly.
        """
        self._validate_context_id(context_id)

        try:
            step_i = int(step)
        except Exception:
            raise ValueError(f"Invalid step value: {step!r}")

        with self._lock:
            s = self.get_or_create(context_id)

            s.num_obs += 1
            s.last_step_received = step_i

            if step_i < s.last_step:
                s.num_step_regressions += 1
            else:
                s.last_step = step_i

            s.touch()
            return s

    def reset(self, context_id: str) -> None:
        """
        reset session state for the given context_id.
        """
        self._validate_context_id(context_id)

        with self._lock:
            s = self.get_or_create(context_id)

            s.task_text = None
            s.task_started_at = None
            s.last_step = -1
            s.last_step_received = -1
            s.num_obs = 0
            s.num_step_regressions = 0

            s.touch()

    # Validate context_id
    def _validate_context_id(self, context_id: str) -> None:
        if not isinstance(context_id, str) or not context_id:
            raise ValueError(f"Invalid context_id: {context_id!r}")
        
    # Garbage-collect expired sessions
    def _gc_if_needed(self) -> None:
        """
        garbage-collect expired sessions.
        """
        if self._ttl_seconds is None:
            return

        now = time.time()
        dead = [
            cid for cid, s in self._sessions.items()
            if (now - s.updated_at) > self._ttl_seconds
        ]
        for cid in dead:
            del self._sessions[cid]
