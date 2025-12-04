# src/satisfying_sims/core/world.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .shapes import Body, CircleCollider
from .boundary import Boundary


@dataclass
class World:
    boundary: Boundary
    gravity: np.ndarray       # (2,)
    drag: float
    restitution: float
    bodies: dict[int, Body] = field(default_factory=dict)
    time: float = 0.0
    _next_id: int = 0

    def new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def add_body(self, body: Body) -> None:
        if body.id is None:
            body.id = self.new_id()
        elif body.id in self.bodies:
            # either raise or reassign — your choice
            raise ValueError(f"Body id {body.id} already exists in world")
        self.bodies[body.id] = body
        return None

    def remove_body_by_id(self, body_id: int) -> None:
        self.bodies = [b for b in self.bodies if b.id != body_id]
        
    def plot(self, ax=None, delta=0, include_boundary=True, color_override=None):
        """
        Plot the world: boundary in gray, bodies in their assigned colors.
        Coordinate system: x-right, y-down.
        """
        if include_boundary:
            fig, ax = self.boundary.plot(delta=delta, edgecolor="gray", linewidth=2, linestyle="--", ax=ax)
        else:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure

        # --- Draw bodies ---
        for id, body in self.bodies.items():
            col = body.collider
            if hasattr(col, "radius"):  # CircleCollider
                circ = Circle(
                    (body.pos[0], body.pos[1]),
                    col.radius,
                    edgecolor="black",
                    facecolor=color_override[id] if color_override is not None else body.color,
                    linewidth=1.0,
                )
                ax.add_patch(circ)
            else:
                raise NotImplementedError("Plotting only supports CircleCollider for now.")

        # y increases downward
        ax.invert_yaxis()

        return fig, ax
