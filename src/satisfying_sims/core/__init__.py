# src/satisfying_sims/core/__init__.py

from .config import SimConfig, make_world
from .world import World
from .physics import step_physics
from .shapes import Body, CircleCollider, create_circle_body
from .boundary import Boundary, BoxBoundary
from .rules import Rule, SpawnRandomBodies, LifetimeDecay, SplitOnHardCollision, SpawnOnCollision
from .recording import FrameSnapshot, SimulationRecording
from .events import (
    BaseEvent,
    CollisionEvent,
    HitWallEvent,
    SpawnEvent,
    DestroyEvent,
)

__all__ = [
    "SimConfig",
    "make_world",
    "World",
    "step_physics",
    "Body",
    "CircleCollider",
    "FrameSnapshot",
    "SimulationRecording",
    "Boundary",
    "BoxBoundary",
    "Rule",
    "SpawnRandomBodies",
    "SpawnOnCollision",
    "LifetimeDecay",
    "SplitOnHardCollision",
    "BaseEvent",
    "CollisionEvent",
    "HitWallEvent",
    "SpawnEvent",
    "DestroyEvent",
]
