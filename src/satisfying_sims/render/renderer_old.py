# src/satisfying_sims/render/renderer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

if TYPE_CHECKING:
    # Adjust these imports to match your actual core API:
    # FrameSnapshot should contain BodySnapshot objects in .bodies.values()
    from satisfying_sims.core import FrameSnapshot, World, Body  # Body = BodySnapshot


@dataclass
class RendererConfig:
    figsize: tuple[float, float] = (6.0, 6.0)
    dpi: int = 200          # bump dpi for video quality
    background_color: str = "white"
    boundary_color: str = "black"
    show_axes: bool = False
    equal_aspect: bool = True
    frame_on: bool = False  # usually off for “satisfying” clips


class MatplotlibRenderer:
    def __init__(self, config: RendererConfig | None = None):
        self.config = config or RendererConfig()

    def render_snapshot(
        self,
        snapshot: FrameSnapshot,
        *,
        ax: Axes,
        world_for_boundary: World | None = None,
    ) -> None:
        """
        Draw a single frame snapshot onto the given Axes.
        """
        ax.clear()
        self._setup_axes(ax)

        if world_for_boundary is not None:
            self._draw_boundary(world_for_boundary, ax)

        for body in snapshot.bodies.values():
            self._draw_body(body, ax)

    # --- helpers ---

    def _setup_axes(self, ax: Axes) -> None:
        ax.set_facecolor(self.config.background_color)
        if self.config.equal_aspect:
            ax.set_aspect("equal", adjustable="box")
        if not self.config.show_axes:
            ax.set_axis_off()
        for spine in ax.spines.values():
            spine.set_visible(self.config.frame_on)

    def _draw_boundary(self, world: World, ax: Axes) -> None:
        boundary = getattr(world, "boundary", None)
        if boundary is None:
            return
        plot_fn = getattr(boundary, "plot", None)
        if callable(plot_fn):
            plot_fn(ax=ax, edgecolor=self.config.boundary_color)

    def _draw_body(self, body: "Body", ax: Axes) -> None:
        """
        Draw a body snapshot.

        Expects `body` to have:
          - pos: array-like shape (2,)
          - color: something Matplotlib can interpret as a color
          - collider: an object with:
                kind: str  (e.g. "CircleCollider")
                attrs: dict with shape parameters, e.g. {"radius": 0.3}
        """
        pos = np.asarray(body.pos, dtype=float)
        fc = getattr(body, "color", "C0")

        collider = getattr(body, "collider", None)
        if collider is None:
            # Fallback: tiny dot if no collider info
            circle = plt.Circle((pos[0], pos[1]), 0.02, fc=fc, ec=None)
            ax.add_patch(circle)
            return

        kind = getattr(collider, "kind", None)
        attrs = getattr(collider, "attrs", {}) or {}

        if kind == "CircleCollider":
            # The collider snapshot should have attrs["radius"]
            radius = float(attrs.get("radius", 0.02))
            circle = plt.Circle((pos[0], pos[1]), radius, fc=fc, ec=None)
            ax.add_patch(circle)

        else:
            # Generic fallback: use bounding_radius if present, otherwise a small dot
            radius = float(
                attrs.get(
                    "bounding_radius",
                    attrs.get("radius", 0.02),
                )
            )
            circle = plt.Circle((pos[0], pos[1]), radius, fc=fc, ec=None)
            ax.add_patch(circle)
