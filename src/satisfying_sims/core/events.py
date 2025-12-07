# src/satisfying_sims/core/events.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from abc import ABC


@dataclass(kw_only=True)
class BaseEvent(ABC):
    """Marker base class so you can type on 'list[BaseEvent]'."""
    t: float  # simulation time when this event occurred
    a_id: int | None = None
    b_id: int | None = None
    def to_payload_dict(self) -> dict:
        """Convert event-specific data to a serializable dict."""
        return {}

@dataclass(kw_only=True)
class CollisionEvent(BaseEvent):
    a_id: int
    b_id: int
    pos: np.ndarray        # collision point (2,)
    impulse: float         # scalar impulse magnitude
    relative_speed: float  # |v_b - v_a| along normal
    
    def to_payload_dict(self) -> dict:
        return {
            "pos": self.pos.tolist(),
            "impulse": self.impulse,
            "relative_speed": self.relative_speed,
        }


@dataclass
class HitWallEvent(BaseEvent):
    body_id: int
    norm_vec: np.ndarray    # (2,) unit vector pointing into the world from wall
    impulse: float
    
    def __post_init__(self):
        self.a_id = self.body_id
        
    def to_payload_dict(self) -> dict:
        return {
            "norm_vec": self.norm_vec.tolist(),
            "impulse": self.impulse,
        }

@dataclass
class SpawnEvent(BaseEvent):
    child_id: int
    reason: str = "unknown"
    
    def __post_init__(self):
        self.a_id = self.child_id 
        
    def to_payload_dict(self) -> dict:
        return {
            "reason": self.reason
        }

@dataclass
class DestroyEvent(BaseEvent):
    body_id: int
    reason: str = "unknown"
    
    def __post_init__(self):
        self.a_id = self.body_id

    def to_payload_dict(self) -> dict:
        payload = super().to_payload_dict()
        payload.update({
            "reason": self.reason
        })
        return payload
