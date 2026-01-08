from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import time


@dataclass
class SessionState:
    """
    Purple 쪽 session state.
    - session key = context_id (A2A message의 contextId)
    - Green은 task 시작 시 init payload를 보내고, 이후 obs를 같은 context_id로 계속 보낸다.
    """

    context_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # init에서 설정되는 task 설명 텍스트
    task_text: Optional[str] = None
    task_started_at: Optional[float] = None

    # obs 흐름에서 추적
    last_step: int = -1
    num_obs: int = 0

    # MineStudio/Minecraft action space는 Purple이 직접 env를 안 돌리므로 확정 불가.
    # 따라서 기본값(또는 config)로 세션마다 "기대 shape"를 보관해서 일관성 유지.
    expected_num_buttons: int = 20   # MineRL 계열에서 흔히 20 전후. 필요시 config로 덮어씀.
    expected_camera_dims: int = 2    # camera는 일반적으로 (dx, dy) 2차원.

    def touch(self) -> None:
        self.updated_at = time.time()


class SessionManager:
    """
    context_id -> SessionState 매핑.

    설계 원칙:
    - init을 받으면 "새 task 시작"으로 보고 session을 task 단위로 reset한다.
    - obs step이 역행하더라도 baseline 안정성을 위해 hard-fail 하지 않고 기록만 갱신한다.
    """

    def __init__(self, *, ttl_seconds: Optional[int] = 60 * 60) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._ttl_seconds = ttl_seconds

    def get_or_create(self, context_id: str) -> SessionState:
        self._gc_if_needed()

        s = self._sessions.get(context_id)
        if s is None:
            s = SessionState(context_id=context_id)
            self._sessions[context_id] = s
        s.touch()
        return s

    def start_new_task(
        self,
        context_id: str,
        task_text: str,
        *,
        expected_num_buttons: Optional[int] = None,
        expected_camera_dims: Optional[int] = None,
    ) -> SessionState:
        """
        init payload 수신 시 호출.
        같은 context_id라도 새로운 init은 "새 task"로 보고 상태를 리셋한다.
        """
        s = self.get_or_create(context_id)

        # reset task-related fields
        s.task_text = task_text
        s.task_started_at = time.time()
        s.last_step = -1
        s.num_obs = 0

        # allow overrides (from config or runtime)
        if expected_num_buttons is not None:
            s.expected_num_buttons = int(expected_num_buttons)
        if expected_camera_dims is not None:
            s.expected_camera_dims = int(expected_camera_dims)

        s.touch()
        return s

    def on_observation(self, context_id: str, step: int) -> SessionState:
        """
        obs payload 수신 시 호출.
        step 역행은 기록만 하고 진행 (baseline 안정성 우선).
        """
        s = self.get_or_create(context_id)
        s.num_obs += 1
        if step > s.last_step:
            s.last_step = step
        s.touch()
        return s

    def reset(self, context_id: str) -> None:
        """
        필요 시 외부에서 강제 초기화.
        """
        s = self.get_or_create(context_id)
        s.task_text = None
        s.task_started_at = None
        s.last_step = -1
        s.num_obs = 0
        s.touch()

    def _gc_if_needed(self) -> None:
        """
        메모리 누수 방지: 오래된 세션 정리.
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
