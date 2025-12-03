# src/satisfying_sims/core/events.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass
class BaseEvent:
    """Marker base class so you can type on 'list[BaseEvent]'."""
    t: float  # simulation time when this event occurred


@dataclass
class CollisionEvent(BaseEvent):
    a_id: int
    b_id: int
    pos: np.ndarray        # collision point (2,)
    impulse: float         # scalar impulse magnitude
    relative_speed: float  # |v_b - v_a| along normal


@dataclass
class HitWallEvent(BaseEvent):
    shape_id: int
    norm_vec: np.ndarray    # (2,) unit vector pointing into the world from wall
    impulse: float


@dataclass
class SpawnEvent(BaseEvent):
    shape_id: int


@dataclass
class DestroyEvent(BaseEvent):
    shape_id: int
    reason: str = "unknown"
