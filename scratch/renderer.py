# src/satisfying_sims/render/renderer.py

from __future__ import annotations

from dataclasses import dataclass, field
from PIL import Image
from typing import TYPE_CHECKING, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import importlib
from satisfying_sims.themes import BodyTheme
from satisfying_sims.utils.render_utils import fig_inches_from_pixels
from satisfying_sims.themes import THEME_REGISTRY
from satisfying_sims.utils.render_utils import compute_axes_rect
if TYPE_CHECKING:
    from satisfying_sims.core.recording import FrameSnapshot, BodyStaticSnapshot, BodyStateSnapshot, BoundaryStaticSnapshot
    from satisfying_sims.core.world import World
    from satisfying_sims.themes.base import BodyThemeConfig
    from satisfying_sims.utils.render_utils import BoxGeometry
    from .collision_effects import CollisionEffectRouter


@dataclass
class RendererConfig:
    figsize: tuple[float, float] = (6.0, 6.0)
    dpi: int = 200          # bump dpi for video quality
    width_px: int | None = None
    height_px: int | None = None
    background_color: str = None
    world_color: str = None
    background_png: str | None = None
    boundary_color: str = None
    body_color_override: str | None = None
    theme_configs: dict[str, BodyThemeConfig] = field(default_factory=dict)
    show_axes: bool = False
    equal_aspect: bool = True
    frame_on: bool = False  # usually off for “satisfying” clips
    padding: float = 0.1   # fraction of figure size to pad around content
    show_debug: bool = False
    fps: int = 30
    


class MatplotlibRenderer:
    def __init__(self, 
                 config: RendererConfig | None = None, 
                 body_static: dict[int, BodyStaticSnapshot] | None = None,
                 background_geom: BoxGeometry | None = None,
                 collision_effects: CollisionEffectRouter | None = None):
        self.config = config or RendererConfig()
        self.body_themes = {}#{'BodyTheme': BodyTheme}
        self.background_geom = background_geom
        self.collision_effects = collision_effects
        self._boundary_drawn = False
        self._inner_wall_lines: list[Any] = []  # matplotlib Line2D
        self._inner_wall_closed: list[bool] = []  # parallel list, one per inner wall
        if body_static is None:
            return

        needed = {b.theme_id for b in body_static.values() if b.theme_id is not None}

        # import once
        mod = importlib.import_module("satisfying_sims.themes", package=__package__)

        for theme_id in needed:
            theme_name = theme_id.split(".")[0]
            ThemeCls = THEME_REGISTRY.get(theme_name, None)
            if ThemeCls is None:
                raise ValueError(f"Unknown theme '{theme_name}'")

            theme_cfg = self.config.theme_configs.get(theme_id, None)
            if theme_cfg is None:
                # either create a default, or raise with a clear error
                #theme_cfg = ThemeCls.default_config()  # if you implement this
                raise ValueError(f"Missing theme config for theme_id {theme_id}")

            self.body_themes[theme_id] = ThemeCls(
                config=theme_cfg
            )

        for theme in self.body_themes.values():
            theme.prepare_for_recording(body_static=body_static)
        self.fig = None
        self.ax = None
        self._axes_rect = None
        self._hud_text = None
        self._caption_text = None
        self.world_text_pad = 0.015
        self.line_gap = 0.07
        self.default_caption = "Each time two of the same object collides,\n" + "a new one spawns."
    
    def _init_figure(self, world: World | None = None):
        world_aspect = world.boundary.get_aspect_ratio() if world is not None else None
        fig, ax = plt.subplots(
        figsize=fig_inches_from_pixels(
            width_px=self.config.width_px, 
            height_px=self.config.height_px, 
            dpi=self.config.dpi, 
            figsize_default=self.config.figsize),
            dpi=self.config.dpi,
        )
        if self.collision_effects is not None:
            self.collision_effects.clear_cache()
        if self.config.background_png is not None:
            img = Image.open(f'{self.config.background_png}/raw.png').convert("RGBA")
            arr = np.asarray(img)
            # Draw at (0, 0) in figure pixel coordinates
            self._axes_rect = self.background_geom.axes_rect()
            fig.figimage(arr, xo=0, yo=0, zorder=-10)
            self._axes_rect = self.background_geom.axes_rect()
        else:
            bg = self.config.background_color if self.config.background_color is not None else "none"
            fig.patch.set_facecolor(bg)
            self._axes_rect = compute_axes_rect(fig, pad=self.config.padding, world_aspect=world_aspect)
        
        self._setup_axes(ax) #TODO: compute aspect from world boundary
        top = self._axes_rect[1] + self._axes_rect[3]
        bottom = self._axes_rect[1]
        self._hud_text = fig.text(0.5, bottom-self.world_text_pad, 
                                  "", 
                                  ha="center", va="top",    
                                  size=14, 
                                  color="white")
        self._caption_text = fig.text(0.5, top+self.world_text_pad,
                                     self.default_caption, 
                                     ha="center", va="bottom", 
                                     size=16, 
                                     color="white")
        
        self._debug_text = fig.text(0.5, 0.2, 
                                  "", 
                                  ha="center", va="top", 
                                  size=14, 
                                  color="white")
        if world is not None:
            self._draw_boundary(world, ax)


        self.fig, self.ax = fig, ax
    
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
        frame_idx: int = 0,
    ) -> None:
        """
        Draw a single frame snapshot onto the given Axes.
        """
        for theme in self.body_themes.values():
            theme.begin_frame() #TODO only call on themes in the frame
        self._draw_boundary_from_snapshot(snapshot=snapshot, ax=ax)
        hud_text = ""
        for theme_id, theme in self.body_themes.items():
            hud_text += theme.HUD_text + str(snapshot.body_counts.get(theme_id, 0)) + "\n"
        self._hud_text.set_text(hud_text[:-1]) # remove last newline
                    
        if self.collision_effects is not None:
            self.collision_effects.begin_frame(frame_idx)
            self.collision_effects.ingest_events(snapshot.events, snapshot, body_static)
            self.collision_effects.draw(ax, frame_idx)
            self.collision_effects.end_frame()
        if self.config.show_debug:
            self._update_debug_overlay(snapshot)
        else:
            self._debug_text.set_text("")
        
        for body_id, state in snapshot.bodies.items():
            theme = self.body_themes[body_static[body_id].theme_id]
            theme.draw_body(
                ax=ax,
                body_id=body_id,
                state=state,
                static=body_static[body_id],
            )
        for theme in self.body_themes.values():
            theme.end_frame()
        
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
            
    def _draw_outer_boundary_once(self, boundary: Any, ax: Axes) -> None:
        if self._boundary_drawn:
            return
        plot_fn = getattr(boundary, "plot", None)
        if callable(plot_fn):
            plot_fn(
                ax=ax,
                facecolor=self.config.world_color if self.config.world_color is not None else "none",
                edgecolor=self.config.boundary_color if self.config.boundary_color is not None else "none",
                linewidth=2,
            )
        self._boundary_drawn = True
    def _draw_boundary_from_snapshot(self, snapshot: "FrameSnapshot", ax: Axes) -> None:
        """
        Draw/update boundary during replay/render-from-recording.

        Expected:
        snapshot.boundary is BoundaryStateSnapshot with wall_points: list[list[(x,y)]]
        """
        bstate = getattr(snapshot, "boundary", None)
        if bstate is None:
            return

        wall_points = bstate.wall_points
        if wall_points is None:
            return

        # Ensure we have the right number of Line2D artists
        while len(self._inner_wall_lines) < len(wall_points):
            (line,) = ax.plot(
                [],
                [],
                linewidth=2,
                color=self.config.boundary_color if self.config.boundary_color is not None else "gray",
                zorder=1,  # behind bodies; bump if you want it on top
            )
            self._inner_wall_lines.append(line)

        # Update each wall line
        for i, pts in enumerate(wall_points):
            line = self._inner_wall_lines[i]
            if not pts:
                line.set_data([], [])
                continue

            xy = np.asarray(pts, dtype=float)  # (N,2)

            # If you have closed flags available, close the loop for plotting
            closed = False
            if i < len(self._inner_wall_closed):
                closed = bool(self._inner_wall_closed[i])

            if closed and len(xy) >= 2:
                xy = np.vstack([xy, xy[0]])

            line.set_data(xy[:, 0], xy[:, 1])
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
