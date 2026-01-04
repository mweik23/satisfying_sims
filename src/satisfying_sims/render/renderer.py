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
    overlay_png: str | None = None
    overlay_size: tuple[float, float] = (12, 12)  # in WORLD units (width, height)
    
    


class MatplotlibRenderer:
    def __init__(
        self,
        config: RendererConfig | None = None,
        body_static: dict[int, "BodyStaticSnapshot"] | None = None,
        background_geom: "BoxGeometry | None" = None,
        collision_effects: "CollisionEffectRouter | None" = None,
        *,
        boundary_static: "BoundaryStaticSnapshot | None" = None,
    ):
        self.config = config or RendererConfig()
        self.body_themes: dict[str, BodyTheme] = {}
        self.background_geom = background_geom
        self.collision_effects = collision_effects
        
        self._overlay_img = None  # AxesImage
        self._overlay_w = None
        self._overlay_h = None
        self._overlay_wall_i = 0                         # which inner wall
        self._overlay_point_i = 1                        # which vertex on that wall
        self._overlay_offset = (0, 0.5)                # (dx, dy) world units

        # --- boundary rendering cache / replay support ---
        self._boundary_drawn = False
        self._inner_wall_lines: list[Any] = []       # matplotlib Line2D artists (inner walls)
        self._inner_wall_closed: list[bool] = []     # closed flags for inner walls
        self._boundary_static = boundary_static      # BoundaryStaticSnapshot saved on recording
        self._outer_boundary_obj = None              # reconstructed outer boundary for replay (lazy)

        # If we have boundary static info, preload "closed" flags for inner walls
        if boundary_static is not None:
            inner_walls = getattr(boundary_static, "inner_walls", None)
            if inner_walls is not None:
                self._inner_wall_closed = [bool(w.closed) for w in inner_walls]

        # --- theme setup ---
        if body_static is None:
            # renderer can still be used for boundary-only drawing/replay init later
            self.fig = None
            self.ax = None
            self._axes_rect = None
            self._hud_text = None
            self._caption_text = None
            self._debug_text = None
            self.world_text_pad = 0.015
            self.line_gap = 0.07
            self.default_caption = "Each time two of the same object collides,\n" + "a new one spawns."
            return

        needed = {b.theme_id for b in body_static.values() if b.theme_id is not None}

        # import once (kept from your code even though not used directly)
        importlib.import_module("satisfying_sims.themes", package=__package__)

        for theme_id in needed:
            theme_name = theme_id.split(".")[0]
            ThemeCls = THEME_REGISTRY.get(theme_name, None)
            if ThemeCls is None:
                raise ValueError(f"Unknown theme '{theme_name}'")

            theme_cfg = self.config.theme_configs.get(theme_id, None)
            if theme_cfg is None:
                raise ValueError(f"Missing theme config for theme_id {theme_id}")

            self.body_themes[theme_id] = ThemeCls(config=theme_cfg)

        for theme in self.body_themes.values():
            theme.prepare_for_recording(body_static=body_static)

        # --- figure state ---
        self.fig = None
        self.ax = None
        self._axes_rect = None
        self._hud_text = None
        self._caption_text = None
        self._debug_text = None

        self.world_text_pad = 0.015
        self.line_gap = 0.07
        self.default_caption = ""
        self.use_hud_text = False
        

    def _init_figure(self, world: "World | None" = None):
        world_aspect = world.boundary.get_aspect_ratio() if world is not None else None
        fig, ax = plt.subplots(
            figsize=fig_inches_from_pixels(
                width_px=self.config.width_px,
                height_px=self.config.height_px,
                dpi=self.config.dpi,
                figsize_default=self.config.figsize,
            ),
            dpi=self.config.dpi,
        )
        if self.collision_effects is not None:
            self.collision_effects.clear_cache()

        if self.config.background_png is not None:
            img = Image.open(f"{self.config.background_png}/raw.png").convert("RGBA")
            arr = np.asarray(img)
            self._axes_rect = self.background_geom.axes_rect()
            fig.figimage(arr, xo=0, yo=0, zorder=-10)
            self._axes_rect = self.background_geom.axes_rect()
        else:
            bg = self.config.background_color if self.config.background_color is not None else "none"
            fig.patch.set_facecolor(bg)
            self._axes_rect = compute_axes_rect(fig, pad=self.config.padding, world_aspect=world_aspect)

        self._setup_axes(ax)
        
        if self.config.overlay_png is not None:
            img = Image.open(self.config.overlay_png).convert("RGBA")
            arr = np.asarray(img)

            self._overlay_w, self._overlay_h = self.config.overlay_size

            # initial placement; will be updated each frame
            x0, y0 = 0.0, 0.0
            extent = (x0, x0 + self._overlay_w, y0, y0 + self._overlay_h)

            self._overlay_img = ax.imshow(
                arr,
                extent=extent,     # world coords
                origin="lower",
                zorder=50,         # above bodies/walls; lower if you want it behind
                interpolation="bilinear",
            )
            self._overlay_img.set_clip_on(False)

        #------
        
        top = self._axes_rect[1] + self._axes_rect[3]
        bottom = self._axes_rect[1]
        self._hud_text = fig.text(
            0.5,
            bottom - self.world_text_pad,
            "",
            ha="center",
            va="top",
            size=14,
            color="white",
        )
        self._caption_text = fig.text(
            0.5,
            top + self.world_text_pad,
            self.default_caption,
            ha="center",
            va="bottom",
            size=16,
            color="white",
        )

        self._debug_text = fig.text(
            0.5,
            0.2,
            "",
            ha="center",
            va="top",
            size=14,
            color="white",
        )

        # Draw OUTER boundary once (live if world provided, else replay via boundary_static)
        self._ensure_outer_boundary_drawn(ax=ax, world=world)

        self.fig, self.ax = fig, ax


    def render_snapshot(
        self,
        snapshot: "FrameSnapshot",
        body_static: dict[int, "BodyStaticSnapshot"] | None = None,
        *,
        ax: Axes,
        frame_idx: int = 0,
    ) -> None:
        """
        Draw a single frame snapshot onto the given Axes.
        """
        if self._overlay_img is not None:
            bstate = getattr(snapshot, "boundary", None)
            if bstate is not None and getattr(bstate, "wall_points", None) is not None:
                wall_points = bstate.wall_points

                wi = self._overlay_wall_i
                pi = self._overlay_point_i

                if 0 <= wi < len(wall_points) and len(wall_points[wi]) > 0:
                    pts = wall_points[wi]

                    # clamp point index (safe if wall has variable vertex count)
                    pi = max(0, min(pi, len(pts) - 1))

                    x, y = pts[pi]
                    dx, dy = self._overlay_offset

                    x0 = float(x) + float(dx) - 0.5*self._overlay_w + self._overlay_offset[0]
                    y0 = float(y) + float(dy) - self._overlay_h + self._overlay_offset[1]
                    self._overlay_img.set_extent((x0, x0 + self._overlay_w, y0, y0 + self._overlay_h))
            
        for theme in self.body_themes.values():
            theme.begin_frame()  # TODO only call on themes in the frame

        # Make robust: ensure outer boundary is drawn even in replay mode
        self._ensure_outer_boundary_drawn(ax=ax, world=None)

        # Update inner walls from recorded snapshot state (if present)
        self._draw_boundary_from_snapshot(snapshot=snapshot, ax=ax)
        if self.use_hud_text:
            hud_text = ""
            for theme_id, theme in self.body_themes.items():
                hud_text += theme.HUD_text + str(snapshot.body_counts.get(theme_id, 0)) + "\n"
            
            self._hud_text.set_text(hud_text[:-1])  # remove last newline

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


    def _draw_boundary(self, world: "World", ax: Axes) -> None:
        """
        Backwards-compatible entry point (outer boundary only).
        Now uses the 'draw once' path.
        """
        self._ensure_outer_boundary_drawn(ax=ax, world=world)


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
        Draw/update INNER walls from the recording.

        Expected:
        snapshot.boundary is BoundaryStateSnapshot with:
            - wall_points: list[list[(x,y)]]
        """
        bstate = getattr(snapshot, "boundary", None)
        if bstate is None:
            return

        wall_points = getattr(bstate, "wall_points", None)
        if wall_points is None:
            return

        # Ensure we have enough Line2D artists
        while len(self._inner_wall_lines) < len(wall_points):
            (line,) = ax.plot(
                [],
                [],
                linewidth=2,
                color=self.config.boundary_color if self.config.boundary_color is not None else "gray",
                zorder=1,
            )
            self._inner_wall_lines.append(line)

        # Update each wall line
        for i, pts in enumerate(wall_points):
            line = self._inner_wall_lines[i]

            if not pts:
                line.set_data([], [])
                continue

            xy = np.asarray(pts, dtype=float)  # (N,2)

            closed = False
            if i < len(self._inner_wall_closed):
                closed = bool(self._inner_wall_closed[i])

            if closed and len(xy) >= 2:
                xy = np.vstack([xy, xy[0]])

            line.set_data(xy[:, 0], xy[:, 1])


    def _build_outer_boundary_from_static(self) -> Any | None:
        """
        Build a concrete boundary instance for plotting from `self._boundary_static`.
        Extend this as you add boundary types.
        """
        bs = getattr(self, "_boundary_static", None)
        if bs is None:
            return None

        outer_kind = getattr(bs, "outer_kind", None)
        outer_attrs = getattr(bs, "outer_attrs", None) or {}
        if outer_kind is None:
            return None

        if outer_kind == "BoxBoundary":
            from satisfying_sims.core.boundary import BoxBoundary
            return BoxBoundary(**outer_attrs)

        if outer_kind == "EllipseBoundary":
            from satisfying_sims.core.boundary import EllipseBoundary
            return EllipseBoundary(**outer_attrs)

        raise ValueError(f"Unsupported outer_kind in boundary_static: {outer_kind}")


    def _ensure_outer_boundary_drawn(self, ax: Axes, world: "World | None" = None) -> None:
        """
        Draw the OUTER boundary once.

        Priority:
        1) Live mode: draw world.boundary if provided
        2) Replay mode: draw a reconstructed boundary from self._boundary_static
        """
        if self._boundary_drawn:
            return

        # Live mode
        if world is not None:
            boundary = getattr(world, "boundary", None)
            if boundary is not None:
                self._draw_outer_boundary_once(boundary, ax)
                return

        # Replay mode
        if getattr(self, "_outer_boundary_obj", None) is None:
            self._outer_boundary_obj = self._build_outer_boundary_from_static()

        if self._outer_boundary_obj is not None:
            self._draw_outer_boundary_once(self._outer_boundary_obj, ax)

    def _setup_axes(self, ax: Axes) -> None:
        ax.set_position(self._axes_rect)
        ax.set_facecolor(self.config.background_color if self.config.background_color is not None else "none")
        
        if self.config.equal_aspect:
            ax.set_aspect("equal", adjustable="box")
        if not self.config.show_axes:
            ax.set_axis_off()
        for spine in ax.spines.values():
            spine.set_visible(self.config.frame_on)
        
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

   