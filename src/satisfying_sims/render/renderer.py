# src/satisfying_sims/render/renderer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import importlib
from satisfying_sims.themes import BodyTheme
from satisfying_sims.utils.render_utils import fig_inches_from_pixels

if TYPE_CHECKING:
    from satisfying_sims.core.recording import FrameSnapshot, BodyStaticSnapshot, BodyStateSnapshot
    from satisfying_sims.core.world import World


@dataclass
class RendererConfig:
    figsize: tuple[float, float] = (6.0, 6.0)
    dpi: int = 200          # bump dpi for video quality
    width_px: int | None = None
    height_px: int | None = None
    background_color: str = "white"
    world_color: str = "lightgray"
    boundary_color: str = "turquoise"
    body_color_override: str | None = None
    show_axes: bool = False
    equal_aspect: bool = True
    frame_on: bool = False  # usually off for “satisfying” clips
    padding: float = 0.1   # fraction of figure size to pad around content
    show_debug: bool = False
    


class MatplotlibRenderer:
    def __init__(self, config: RendererConfig | None = None, body_static: dict[int, BodyStaticSnapshot] | None = None):
        self.config = config or RendererConfig()
        self.body_themes = {}#{'BodyTheme': BodyTheme}
        if body_static is not None:
            for _, b in body_static.items():
                if b.theme is not None and b.theme not in self.body_themes.keys():
                    Mod = importlib.import_module('satisfying_sims.themes', package=__package__)
                    self.body_themes[b.theme] = getattr(Mod, b.theme)(facecolor=self.config.body_color_override)
            for theme in self.body_themes.values():
                theme.prepare_for_recording(body_static=body_static)
        self.fig = None
        self.ax = None
        self._axes_rect = None
        self._hud_text = None
        self._caption_text = None
        self.world_text_pad = 0.01
        self.line_gap = 0.07
    
    def _init_figure(self, world_aspect: float):
        fig, ax = plt.subplots(
        figsize=fig_inches_from_pixels(width_px=self.config.width_px, 
                           height_px=self.config.height_px, 
                           dpi=self.config.dpi, 
                           figsize_default=self.config.figsize),
        dpi=self.config.dpi,
    )

        bg = self.config.background_color if self.config.background_color is not None else "none"
        fig.patch.set_facecolor(bg)

        self._axes_rect = self._compute_axes_rect(fig, pad=self.config.padding, world_aspect=world_aspect)
        ax.set_position(self._axes_rect)
        top = self._axes_rect[1] + self._axes_rect[3]
        hud_text_y = top + self.world_text_pad
        self._hud_text = fig.text(0.5, hud_text_y, 
                                  "", 
                                  ha="center", va="bottom", 
                                  size=14, 
                                  color="white")
        self._caption_text = fig.text(0.5, self.line_gap+hud_text_y, 
                                     "Each time two bodies collide,\n" + "a new one spawns.", 
                                     ha="center", va="bottom", 
                                     size=18, 
                                     color="white")
        
        self._debug_text = fig.text(0.5, 0.2, 
                                  "", 
                                  ha="center", va="top", 
                                  size=14, 
                                  color="white")

        self.fig, self.ax = fig, ax
    
    def _compute_axes_rect(self, fig, pad: float, world_aspect: float) -> list[float]:
        fig_aspect = fig.get_figwidth() / fig.get_figheight()
        return [
            pad,
            (1 - (1 - 2*pad) / (world_aspect / fig_aspect)) / 2,
            1 - 2*pad,
            (1 - 2*pad) / (world_aspect / fig_aspect),
        ]
    
    def _update_debug_overlay(self, frame):
        dbg = frame.rates or {}

        lines = [
            f"t = {frame.t:7.3f} s",
            f"λ_coll = {dbg.get('CollisionEvent', 0.0):6.1f} / s",
            f"λ_wall = {dbg.get('HitWallEvent', 0.0):6.1f} / s",
        ]

        self._debug_text.set_text("\n".join(lines))
    
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
        self._setup_axes(ax) #TODO: compute aspect from world boundary
        self._hud_text.set_text(f"Body Count: {len(snapshot.bodies)}")
        if self.config.show_debug:
            self._update_debug_overlay(snapshot)
        else:
            self._debug_text.set_text("")
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

    def _setup_axes(self, ax: Axes) -> None:
        ax.set_position(self._axes_rect)
        ax.set_facecolor(self.config.background_color if self.config.background_color is not None else "none")
        
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
            plot_fn(ax=ax, 
                    facecolor=self.config.world_color if self.config.world_color is not None else "none", 
                    edgecolor=self.config.boundary_color if self.config.boundary_color is not None else "none", 
                    linewidth=2)

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
