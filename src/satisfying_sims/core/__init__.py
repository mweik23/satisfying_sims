# src/satisfying_sims/core/__init__.py

from .config import SimConfig, make_world
from .world import World
from .physics import step_physics
from .shapes import Body, CircleCollider, create_circle_body
from .boundary import Boundary, BoxBoundary
from .rules import Rule, SpawnRandomShapes, LifetimeDecay, SplitOnHardCollision
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
    "Boundary",
    "BoxBoundary",
    "Rule",
    "SpawnRandomShapes",
    "LifetimeDecay",
    "SplitOnHardCollision",
    "BaseEvent",
    "CollisionEvent",
    "HitWallEvent",
    "SpawnEvent",
    "DestroyEvent",
]
