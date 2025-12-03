# src/satisfying_sims/core/world.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np

from .shapes import Body, CircleCollider
from .boundary import Boundary


@dataclass
class World:
    boundary: Boundary
    gravity: np.ndarray       # (2,)
    drag: float
    restitution: float
    bodies: List[Body] = field(default_factory=list)
    time: float = 0.0
    _next_id: int = 0

    def new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def add_body(self, body: Body) -> None:
        self.bodies.append(body)

    def create_circle_body(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        color: tuple,
        radius: float = 1.0,
        mass: float = 1.0,
        life: float = float("inf"),
        state: dict | None = None,
    ) -> Body:
        body = Body(
            id=self.new_id(),
            pos=pos.astype(float),
            vel=vel.astype(float),
            mass=float(mass),
            color=color,
            collider=CircleCollider(radius=float(radius)),
            life=float(life),
            state={} if state is None else dict(state),
        )
        self.bodies.append(body)
        return body

    def remove_body_by_id(self, body_id: int) -> None:
        self.bodies = [b for b in self.bodies if b.id != body_id]
