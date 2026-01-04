# src/satisfying_sims/core/__init__.py

from .config import SimConfig, make_world
from .sim_decisions import SimStopRestartPolicy, SimAction
from .world import World, run_simulation
from .physics import step_physics
from .shapes import Body, CircleCollider, create_circle_body
from .boundary import Boundary, BoxBoundary, EllipseBoundary, WallBoundary
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
    "create_circle_body",
    "run_simulation",
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
    "WallBoundary",
    "EllipseBoundary",
    "make_world",
    "SimStopRestartPolicy",
    "SimAction"
]
