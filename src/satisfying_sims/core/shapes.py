# src/satisfying_sims/core/shapes.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod
import numpy as np

Color = Tuple[int, int, int]


# --------- Colliders (geometry only) ---------

class Collider(ABC):
    """Geometric shape in local space. No position/velocity/mass here."""

    @abstractmethod
    def bounding_radius(self) -> float:
        """Radius of sphere that encloses this shape (for cheap broad-phase tests)."""
        ...


@dataclass
class CircleCollider(Collider):
    radius: float

    def bounding_radius(self) -> float:
        return self.radius


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
    state: Dict[str, Any] = field(default_factory=dict)

    # Convenience for current circle-only world
    @property
    def radius(self) -> float:
        if isinstance(self.collider, CircleCollider):
            return self.collider.radius
        return self.collider.bounding_radius()
