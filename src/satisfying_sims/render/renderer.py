# src/satisfying_sims/render/renderer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import importlib
from satisfying_sims.themes import BodyTheme

if TYPE_CHECKING:
    from satisfying_sims.core.recording import FrameSnapshot, BodyStaticSnapshot, BodyStateSnapshot
    from satisfying_sims.core.world import World


@dataclass
class RendererConfig:
    figsize: tuple[float, float] = (6.0, 6.0)
    dpi: int = 200          # bump dpi for video quality
    background_color: str = "white"
    world_color: str = "lightgray"
    boundary_color: str = "black"
    show_axes: bool = False
    equal_aspect: bool = True
    frame_on: bool = False  # usually off for “satisfying” clips
    padding: float = 0.1   # fraction of figure size to pad around content
    


class MatplotlibRenderer:
    def __init__(self, config: RendererConfig | None = None, body_static: dict[int, BodyStaticSnapshot] | None = None):
        self.config = config or RendererConfig()
        self.body_themes = {}#{'BodyTheme': BodyTheme}
        if body_static is not None:
            for _, b in body_static.items():
                if b.theme is not None and b.theme not in self.body_themes.keys():
                    Mod = importlib.import_module('satisfying_sims.themes', package=__package__)
                    self.body_themes[b.theme] = getattr(Mod, b.theme)()
            for theme in self.body_themes.values():
                theme.prepare_for_recording(body_static=body_static)
        #print('body themes: ', list(self.body_themes.keys()))
        
    def render_snapshot(
        self,
        snapshot: "FrameSnapshot",
        body_static: dict[int, BodyStaticSnapshot] | None = None,
        *,
        ax: Axes,
        world_for_boundary: "World" | None = None,
    ) -> None:
        """
        Draw a single frame snapshot onto the given Axes.
        """
        ax.clear()
        self._setup_axes(ax)

        if world_for_boundary is not None:
            self._draw_boundary(world_for_boundary, ax)

        for body_id, state in snapshot.bodies.items():
            theme = self.body_themes[body_static[body_id].theme]
            theme.draw_body(
                ax=ax,
                body_id=body_id,
                state=state,
                static=body_static[body_id] or None,
            )
    # --- helpers ---

    def _setup_axes(self, ax: Axes, pad: float = 0.05) -> None:
        # Fill entire figure
        ax.set_position([pad, pad, 1-2*pad, 1-2*pad])
        ax.set_facecolor(self.config.background_color)
        fig = ax.get_figure()
        fig.patch.set_facecolor(self.config.background_color)
        if self.config.equal_aspect:
            ax.set_aspect("equal", adjustable="box")
        if not self.config.show_axes:
            ax.set_axis_off()
        for spine in ax.spines.values():
            spine.set_visible(self.config.frame_on)

    def _draw_boundary(self, world: "World", ax: Axes) -> None:
        boundary = getattr(world, "boundary", None)
        if boundary is None:
            return
        plot_fn = getattr(boundary, "plot", None)
        if callable(plot_fn):
            pad = self.config.padding * max(boundary.width, boundary.height)
            plot_fn(ax=ax, delta=pad, facecolor=self.config.world_color, edgecolor=self.config.boundary_color)

    def _draw_body(
        self,
        state: "BodyStateSnapshot",
        body_static: "BodyStaticSnapshot",
        ax: Axes,
    ) -> None:
        """
        Draw a body using the new snapshot structure.

        Parameters
        ----------
        state : BodyStateSnapshot
            Per-frame dynamic state (pos, vel).
        body_static : BodyStaticSnapshot
            Global static info stored once per recording.
        ax : matplotlib.axes.Axes
            Target axes.
        """

        # --- dynamic state ---
        pos = np.asarray(state.pos, dtype=float)

        # --- static visual attributes ---
        fc = body_static.color                   # already normalized (0–1) tuple
        collider = body_static.collider          # ColliderSnapshot

        if collider is None:
            # Fallback dot
            circle = plt.Circle((pos[0], pos[1]), 0.02, fc=fc, ec=None)
            ax.add_patch(circle)
            return

        kind = collider.kind
        attrs = collider.attrs or {}

        # --- Circle collider ---
        if kind == "CircleCollider":
            radius = float(attrs.get("radius", 0.02))
            circle = plt.Circle((pos[0], pos[1]), radius, fc=fc, ec=None)
            ax.add_patch(circle)
            return

        # --- Fallback for unknown collider types ---
        radius = float(attrs.get("bounding_radius", attrs.get("radius", 0.02)))
        circle = plt.Circle((pos[0], pos[1]), radius, fc=fc, ec=None)
        ax.add_patch(circle)

    def _draw_body_old(self, body: "BodyStateSnapshot", ax: Axes) -> None:
        """
        Draw a body snapshot.

        Expects `body` to have:
          - pos: array-like shape (2,)
          - color: Matplotlib-understood color
          - collider: ColliderSnapshot with:
                kind: str
                attrs: dict, e.g. {"radius": 0.3}
        """
        pos = np.asarray(body.pos, dtype=float)
        fc = getattr(body, "color", "C0")

        collider = getattr(body, "collider", None)
        if collider is None:
            circle = plt.Circle((pos[0], pos[1]), 0.02, fc=fc, ec=None)
            ax.add_patch(circle)
            return

        kind = getattr(collider, "kind", None)
        attrs = getattr(collider, "attrs", {}) or {}

        if kind == "CircleCollider":
            radius = float(attrs.get("radius", 0.02))
            circle = plt.Circle((pos[0], pos[1]), radius, fc=fc, ec=None)
            ax.add_patch(circle)
        else:
            # Fallback: bounding_radius if present, else a tiny dot
            radius = float(
                attrs.get("bounding_radius", attrs.get("radius", 0.02))
            )
            circle = plt.Circle((pos[0], pos[1]), radius, fc=fc, ec=None)
            ax.add_patch(circle)
