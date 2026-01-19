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
from satisfying_sims.utils.render_utils import fig_inches_from_pixels, get_pix_per_world
from .overlay import create_overlay_artist, OverlayConfig
from .text import TextConfig
from .theme_config_factory import BodyThemeRouter
from satisfying_sims.themes import THEME_REGISTRY
from satisfying_sims.utils.render_utils import compute_axes_rect, set_background
from .artists import build_wall_artist
if TYPE_CHECKING:
    from satisfying_sims.core.recording import FrameSnapshot, BodyStaticSnapshot, BodyStateSnapshot, BoundaryStaticSnapshot
    from satisfying_sims.core.world import World
    from satisfying_sims.themes.base import BodyThemeConfig
    from satisfying_sims.utils.render_utils import BoxGeometry
    from .collision_effects import CollisionEffectRouter

def make_renderers(
    config: RendererConfig | None = None,
    body_static: dict[int, BodyStaticSnapshot] | None = None,
    background_geom: BoxGeometry | None = None,
    collision_effects: CollisionEffectRouter | None = None,
    body_themes: BodyThemeRouter | None = None,
    boundary_static: BoundaryStaticSnapshot | None = None,
    overlay_cfg: OverlayConfig | None = None,
    text_cfg: TextConfig | None = None,
):
    plot_layers = set()
    if body_themes is not None:
        plot_layers.update(body_themes.get_layers())
    if collision_effects is not None:
        plot_layers.update(collision_effects.get_layers())
    if text_cfg is not None:
        plot_layers.add(text_cfg.layer)
    if overlay_cfg is not None:
        plot_layers.add(overlay_cfg.layer)
    if boundary_static is not None:
        plot_layers.update(boundary_static.get_layers())
    renderers = []
    for plot_layer in sorted(plot_layers):
        renderers.append(
            MatplotlibRenderer(
                config=config,
                body_static=body_static,
                background_geom=background_geom,
                collision_effects=collision_effects,
                body_themes=body_themes,
                boundary_static=boundary_static,
                overlay_cfg=overlay_cfg,
                text_cfg=text_cfg,
                plot_layer=plot_layer
            )
        )
    return renderers

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
    show_axes: bool = False
    equal_aspect: bool = True
    frame_on: bool = False  # usually off for “satisfying” clips
    padding: float = 0.1   # fraction of figure size to pad around content
    show_debug: bool = False
    fps: int = 30

class MatplotlibRenderer:
    def __init__(
        self,
        config: RendererConfig | None = None,
        body_static: dict[int, "BodyStaticSnapshot"] | None = None,
        background_geom: "BoxGeometry | None" = None,
        collision_effects: "CollisionEffectRouter | None" = None,
        body_themes: BodyThemeRouter | None = None,
        boundary_static: "BoundaryStaticSnapshot | None" = None,
        overlay_cfg: OverlayConfig | None = None,
        text_cfg: TextConfig | None = None,
        plot_layer: int = 0
    ):
        self.config = config or RendererConfig()
        self.plot_layer = plot_layer
        self.boundary_color = self.config.boundary_color if self.config.boundary_color is not None else "none"
        self.boundary_lw = self.config.boundary_lw if hasattr(self.config, "boundary_lw") and self.config.boundary_lw is not None else 1.0
        self.body_themes = body_themes
        self.background_geom = background_geom
        self.collision_effects = collision_effects
        self.body_static = body_static
        self.text_cfg = text_cfg or TextConfig()
        
        self.overlay_cfg = overlay_cfg
        self._overlay_artist = None  # AxesImage

        # --- boundary rendering cache / replay support ---
        self._boundary_drawn = False
        self._boundary_static = boundary_static      # BoundaryStaticSnapshot saved on recording
        self._outer_boundary_obj = None              # reconstructed outer boundary for replay (lazy)

        # --- theme setup ---
        if body_static is None:
            # renderer can still be used for boundary-only drawing/replay init later
            self.fig = None
            self.ax = None
            self._axes_rect = None
            self._hud_text = None
            self._caption_text = None
            self._debug_text = None
            return

        # --- figure state ---
        self.fig = None
        self.ax = None
        self._axes_rect = None
        self._hud_text = None
        self._caption_text = None
        self._debug_text = None
        self.px_per_world = None

    def _init_figure(self, world: "World | None" = None):
        #create figure sized to be the correct resolution and extract axes
        fig, ax = plt.subplots(
            figsize=fig_inches_from_pixels(
                width_px=self.config.width_px,
                height_px=self.config.height_px,
                dpi=self.config.dpi,
                figsize_default=self.config.figsize,
            ),
            dpi=self.config.dpi,
        )
        
        #set background as an image or solid color
        set_background(
            fig, 
            image_path=f"{self.config.background_png}/raw.png" if self.config.background_png is not None else None, 
            color=self.config.background_color,
            plot_layer=self.plot_layer
        )
        
        # get axes rect from background geometry if available, 
        # else comput based on the figure, world boundary, and padding
        self._axes_rect = compute_axes_rect(
            fig, 
            pad=self.config.padding, 
            boundary=world.boundary if world is not None else None,
            background_geom=self.background_geom if hasattr(self, "background_geom") else None,
        )
  
        self._setup_axes(ax)
        
        # Draw OUTER boundary once (live if world provided, else replay via boundary_static)
        self._ensure_outer_boundary_drawn(ax=ax, world=world)
        
        self._setup_camera(ax, bbox=self._outer_boundary_obj.outer.get_bounding_box())

        #clear cache for collision effects. For collision effects with a default, plot it now
        if self.collision_effects is not None:
            self.collision_effects.clear_cache(plot_layer=self.plot_layer)
            self.collision_effects.plot_default(ax, plot_layer=self.plot_layer)

        # prepare wall artists for inner walls. If they only change by affine transformations 
        # then create a standard artist that can be transformed for each frame.
        self._wall_artists = {
            wid: build_wall_artist(
                ax, 
                wall_static, 
                default_style={
                    "color": self.boundary_color, 
                    "linewidth": self.boundary_lw
                },
                plot_layer=self.plot_layer
            ) for wid, wall_static in self._boundary_static.inner_walls.items()
        }
        
        #create overlay artist if needed
        self._overlay_artist = create_overlay_artist(
            ax,
            overlay_cfg=self.overlay_cfg,
            plot_layer=self.plot_layer
        )
        
        #create and position text artists
        self.create_text_artists(fig)
        
        #load theme assets and resize them based on px_per_world to reduce overhead during rendering
        self.body_themes.prepare_for_recording(body_static=self.body_static, px_per_world=get_pix_per_world(ax), plot_layer=self.plot_layer)
        self.fig, self.ax = fig, ax
        """
        print("fig.patch alpha:", fig.patch.get_alpha(), "facecolor:", fig.patch.get_facecolor())
        print("ax.patch  alpha:", ax.patch.get_alpha(), "facecolor:", ax.patch.get_facecolor())
        print("num fig images:", len(fig.images))
        """
        
    def _setup_camera(self, ax: Axes, bbox: tuple[float, float, float, float] | None = None, pad_frac: float = 0.0) -> None:
        xmin, xmax, ymin, ymax = bbox
        dx = (xmax - xmin) * pad_frac
        dy = (ymax - ymin) * pad_frac
        ax.set_xlim(xmin - dx, xmax + dx)
        ax.set_ylim(ymin - dy, ymax + dy)
        ax.set_autoscale_on(False)
        
    def create_text_artists(self, fig: plt.Figure):
        if self.text_cfg.layer != self.plot_layer:
            return # Don't create text artists if they're not on the current plot layer
        top = self._axes_rect[1] + self._axes_rect[3]
        bottom = self._axes_rect[1]
        if self.text_cfg.use_hud_text:
            self._hud_text = fig.text(
                0.5,
                bottom - self.text_cfg.world_text_pad,
                "",
                ha="center",
                va="top",
                size=self.text_cfg.hud_size,
                color="white",
            )
        if self.text_cfg.use_caption_text:
            self._caption_text = fig.text(
                0.5,
                top + self.text_cfg.world_text_pad,
                self.text_cfg.caption_content,
                ha="center",
                va="bottom",
            size=self.text_cfg.caption_size,
            color="white",
        )
        if self.text_cfg.use_debug_text:
            self._debug_text = fig.text(
                0.5,
                0.2,
                "",
                ha="center",
                va="top",
                size=self.text_cfg.debug_size,
                color="white",
        )
    def update_HUD_text(self, snapshot: "FrameSnapshot") -> None:
        if self.text_cfg.layer != self.plot_layer:
            return # Don't update HUD text if it's not on the current plot layer
        if self.text_cfg.use_hud_text:
            hud_text = ""
            for theme_id, theme in self.body_themes.themes.items():
                hud_text += theme.HUD_text + str(snapshot.body_counts.get(theme_id, 0)) + "\n"
            
            self._hud_text.set_text(hud_text[:-1])  # remove last newline

        
    def draw_overlay_img(self, snapshot: "FrameSnapshot") -> None:
        if self._overlay_artist is not None:
            bstate = getattr(snapshot, "boundary", None)
            if bstate is not None and getattr(bstate, "wall_points", None) is not None:
                wall_points = bstate.wall_points

                wi = self.overlay_cfg.wall_idx if self.overlay_cfg is not None else 0
                pi = self.overlay_cfg.point_idx if self.overlay_cfg is not None else 0

                if 0 <= wi < len(wall_points) and len(wall_points[wi]) > 0:
                    pts = wall_points[wi]

                    # clamp point index (safe if wall has variable vertex count)
                    pi = max(0, min(pi, len(pts) - 1))

                    x, y = pts[pi]
                    dx, dy = self.overlay_cfg.offset

                    x0 = float(x) + float(dx) - 0.5*self.overlay_cfg.size[0] + self.overlay_cfg.offset[0]
                    y0 = float(y) + float(dy) - self.overlay_cfg.size[1] + self.overlay_cfg.offset[1]
                    self._overlay_artist.set_extent((x0, x0 + self.overlay_cfg.size[0], y0, y0 + self.overlay_cfg.size[1]))
                    
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
        
        #if there is an overlay artist draw it. #TODO: this is will not work with the new boundary state structure. Next time we want to have an overlay, I will fix
        self.draw_overlay_img(snapshot)
        
        #right now this only clears seen bodies from previous frames
        self.body_themes.begin_frame(plot_layer=self.plot_layer)  # TODO only call on themes in the frame

        # should do nothing if the boundary is static and already drawn. TODO: make it handle dynamic boundaries in the future
        self._ensure_outer_boundary_drawn(ax=ax, world=None)

        # Update inner walls from recorded snapshot state (if present)
        self._draw_boundary_from_snapshot(snapshot=snapshot, ax=ax)
        
        #update HUD text based on theme info and snapshot body counts
        self.update_HUD_text(snapshot)
        
        # manage and draw collision effects if configured
        if self.collision_effects is not None:
            self.collision_effects.begin_frame(frame_idx, plot_layer=self.plot_layer)
            self.collision_effects.check_windows(frame_idx, plot_layer=self.plot_layer)
            self.collision_effects.ingest_events(snapshot.events, snapshot, body_static, plot_layer=self.plot_layer)
            self.collision_effects.draw(ax, frame_idx, plot_layer=self.plot_layer)
            self.collision_effects.end_frame(plot_layer=self.plot_layer)
        self.body_themes.draw_bodies(ax=ax, body_states=snapshot.bodies, body_static=body_static, plot_layer=self.plot_layer)

        self.body_themes.end_frame(plot_layer=self.plot_layer)


    def _draw_boundary(self, world: "World", ax: Axes) -> None:
        """
        Backwards-compatible entry point (outer boundary only).
        Now uses the 'draw once' path.
        """
        self._ensure_outer_boundary_drawn(ax=ax, world=world)


    def _draw_outer_boundary_once(self, boundary: Any, ax: Axes) -> None:
        if self._boundary_drawn:
            return
        if boundary.layer != self.plot_layer:
            return # Don't draw this boundary if it's not on the current plot layer
        plot_fn = getattr(boundary, "plot", None)
        edgecolor = getattr(boundary, "color", None) #TODO: get style info from boundary_static if not present on the boundary object itself
        if edgecolor is None:
            edgecolor = self.boundary_color if self.boundary_color is not None else "none"
        if callable(plot_fn):
            plot_fn(
                ax=ax,
                facecolor=self.config.world_color if self.config.world_color is not None else "none",
                edgecolor=edgecolor,
                linewidth=1,
            )
        self._boundary_drawn = True

    def _draw_boundary_from_snapshot(self, snapshot: "FrameSnapshot", ax: Axes) -> None:
        """
        Draw/update INNER walls from the recording.

        Expected:
        snapshot.boundary is BoundaryStateSnapshot with:
            - inner_wall_snapshots: list[WallStateSnapshot] (per-frame state)
        """
        for state in snapshot.boundary.inner_wall_snapshots:
            if getattr(self._boundary_static.inner_walls[state.id], "layer", 0) != self.plot_layer:
                continue # Don't draw this wall if it's not on the current plot layer
            self._wall_artists[state.id].update(state)


    def _build_outer_boundary_from_static(self) -> Any | None:
        """
        Build a concrete boundary instance for plotting from `self._boundary_static`.
        Extend this as you add boundary types.
        """
        bs = getattr(self, "_boundary_static", None)
        if bs is None:
            return None
        outer_layer = getattr(bs, "outer_layer", 0)

        outer_kind = getattr(bs, "outer_kind", None)
        outer_attrs = dict(getattr(bs, "outer_attrs", None) or {})
        outer_attrs.pop("style", None)
        if outer_kind is None:
            return None

        if outer_kind == "BoxBoundary":
            from satisfying_sims.core.boundary import BoxBoundary
            return BoxBoundary(**outer_attrs, layer=outer_layer)

        if outer_kind == "EllipseBoundary":
            from satisfying_sims.core.boundary import EllipseBoundary
            return EllipseBoundary(**outer_attrs, layer=outer_layer)

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
            self._outer_boundary_obj = boundary  # cache for replay
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
        #ax.set_facecolor(self.config.background_color if self.config.background_color is not None else "none")
        
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

   