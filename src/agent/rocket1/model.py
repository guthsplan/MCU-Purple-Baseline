from dataclasses import dataclass
from typing import Any

@dataclass
class RocketState:
    memory: Any
    first: bool = True
