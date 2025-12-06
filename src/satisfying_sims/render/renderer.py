# src/satisfying_sims/render/renderer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .colors import color_for_index, build_time_cmap

if TYPE_CHECKING:
    from satisfying_sims.core import FrameSnapshot, Body, World


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

    # --- helpers (same as before, just Snapshot / Body-agnostic) ---

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
        pos = np.asarray(body.pos, dtype=float)
        radius = float(body.radius)
        fc = getattr(body, "color", "C0")
        circle = plt.Circle((pos[0], pos[1]), radius, fc=fc, ec=None)
        ax.add_patch(circle)
