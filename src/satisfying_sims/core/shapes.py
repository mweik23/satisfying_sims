# src/satisfying_sims/core/shapes.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod
import numpy as np
from typing import Union
Color = Tuple[int, int, int]
from satisfying_sims.utils.plotting import color_to_uint8
from .recording import ColliderSnapshot
from satisfying_sims.themes.base import BodyTheme

# --------- Colliders (geometry only) ---------

class Collider(ABC):
    """Geometric shape in local space. No position/velocity/mass here."""

    @abstractmethod
    def bounding_radius(self) -> float:
        """Radius of sphere that encloses this shape (for cheap broad-phase tests)."""
        ...

    def to_snapshot(self) -> ColliderSnapshot:
        return ColliderSnapshot(
            kind=type(self).__name__,
            attrs={"bounding_radius": float(self.bounding_radius())},
        )

@dataclass
class CircleCollider(Collider):
    radius: float

    def bounding_radius(self) -> float:
        return self.radius
    
    def to_snapshot(self) -> ColliderSnapshot:
        return ColliderSnapshot(
            kind="CircleCollider",
            attrs={"radius": float(self.radius)},
        )
# Placeholder for future expansion
# @dataclass
# class PolygonCollider(Collider):
#     vertices: np.ndarray
#     def bounding_radius(self) -> float:
#         return float(np.linalg.norm(self.vertices, axis=1).max())


# --------- Bodies (physics state) ---------

@dataclass
class Body:
    """
    A physical object in the world.

    - pos / vel: world-space position & velocity of the body's origin
    - collider: underlying geometric shape (currently just CircleCollider)
    - mass: used for impulse calculations
    """
    id: int
    pos: np.ndarray           # shape (2,)
    vel: np.ndarray           # shape (2,)
    mass: float
    color: Color
    collider: Collider
    life: float = float("inf")
    theme: BodyTheme | None = None
    collision_count: int = 0
    state: Dict[str, Any] = field(default_factory=dict)

    # Convenience for current circle-only world
    @property
    def radius(self) -> float:
        if isinstance(self.collider, CircleCollider):
            return self.collider.radius
        return self.collider.bounding_radius()

def create_circle_body(
    pos: np.ndarray,
    vel: np.ndarray,
    color: Union[tuple, str] = (0, 0, 0),
    theme: BodyTheme | None = None,
    radius: float = 1.0,
    mass: float = 1.0,
    life: float = float("inf"),
    state: dict | None = None,
) -> Body:
        body = Body(
            id=None,
            pos=pos.astype(float),
            vel=vel.astype(float),
            mass=float(mass),
            color=color_to_uint8(color) if type(color) is str else color,
            theme = theme,
            collider=CircleCollider(radius=float(radius)),
            life=float(life),
            state={} if state is None else dict(state),
        )
        return body