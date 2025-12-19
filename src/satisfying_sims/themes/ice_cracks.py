# src/satisfying_sims/themes/ice_cracks.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from satisfying_sims.core.recording import (
    SimulationRecording,
    BodyStaticSnapshot,
    BodyStateSnapshot,
)
from satisfying_sims.themes.base import BodyTheme
from satisfying_sims.utils.random import rng_for_key

@dataclass
class IceCracksTheme(BodyTheme):
    """
    Bodies look like translucent ice cubes with cracks that depend only on
    the collision_count stored in BodyStateSnapshot.
    """
    max_cracks_per_body: int = 10
    facecolor: tuple[float, float, float, float] | None = (0.85, 0.93, 1.0, 0.6)
    edgecolor: tuple[float, float, float, float] = (0.9, 0.98, 1.0, 1.0)
    highlight_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.35)
    shadow_color: tuple[float, float, float, float] = (0.2, 0.35, 0.6, 0.18)     # bluish shadow
    rim_color: tuple[float, float, float, float] = (0.6, 0.85, 1.0, 0.22)        # subtle rim
    line_color: tuple[float, float, float, float] = (0.75, 0.9, 1.0, 1.0)
    three_d_effects: bool = False

    # runtime fields: filled in prepare_for_recording
    crack_geometries: Dict[int, List[np.ndarray]] | None = None

    def prepare_for_recording(self, body_static: dict[int, BodyStaticSnapshot] | None = None) -> None:
        """
        Pre-generate crack geometries for each body id in body_static.

        We don't need collision counts here; we just generate up to
        max_cracks_per_body crack polylines per body and later select how many
        to show based on state.collision_count.
        """
        self.crack_geometries = {}
        #TODO: make rng be determined from base_seed
        
        body_ids = list(body_static.keys())
        for body_id in body_ids:
            crack_rng = rng_for_key("cracks", body_id)
            self.crack_geometries[body_id] = self._generate_cracks_for_body(
                rng=crack_rng,
                n_cracks=self.max_cracks_per_body,
            )

    def draw_body(
        self,
        ax: Axes,
        body_id: int,
        state: BodyStateSnapshot,
        static: BodyStaticSnapshot,
    ) -> None:
        # 1) draw the translucent ice cube
        x, y = state.pos
        collider = static.collider
        kind = collider.kind
        attrs = collider.attrs or {}

        if kind=="CircleCollider":
            radius = attrs.get('radius', 0.02)

            # --- Base sphere ---
            circle = Circle(
                (x, y),
                radius,
                linewidth=radius / 4,
                edgecolor=self.edgecolor,
                facecolor=self.facecolor if self.facecolor is not None else static.color,
                zorder=2,
            )
            ax.add_patch(circle)
            if self.three_d_effects:
                # --- Rim vignette (a slightly smaller ring-ish feel) ---
                rim = Circle(
                    (x, y),
                    radius*0.98,
                    linewidth=radius / 2.8,
                    edgecolor=self.rim_color,
                    facecolor="none",
                    zorder=3,
                )
                ax.add_patch(rim)

                # --- Soft shadow (bottom-right) ---
                sx, sy = x + 0.18 * radius, y + 0.18 * radius
                shadow = Circle(
                    (sx, sy),
                    radius * 0.92,
                    linewidth=0.0,
                    edgecolor="none",
                    facecolor=self.shadow_color,
                    zorder=3,
                )
                ax.add_patch(shadow)

                # --- Specular highlight (top-left) ---
                hx, hy = x - 0.28 * radius, y - 0.28 * radius
                highlight = Circle(
                    (hx, hy),
                    radius * 0.38,
                    linewidth=0.0,
                    edgecolor="none",
                    facecolor=self.highlight_color,
                    zorder=4,
                )
                ax.add_patch(highlight)

                # --- Tiny bright core to make the highlight pop ---
                highlight2 = Circle(
                    (hx - 0.04 * radius, hy - 0.04 * radius),
                    radius * 0.18,
                    linewidth=0.0,
                    edgecolor="none",
                    facecolor=(1.0, 1.0, 1.0, 0.18),
                    zorder=5,
                )
                ax.add_patch(highlight2)

        # 2) cracks based on total collision_count
        if self.crack_geometries is None:
            return

        n_collisions = getattr(state, "collision_count", 0)
        if n_collisions <= 0:
            return

        crack_polylines = self.crack_geometries.get(body_id, [])
        if not crack_polylines:
            return

        n_to_show = min(len(crack_polylines), n_collisions+3)
        lw = (radius/3) * (1.3 + 0.05 * (n_to_show-1))

        for i in range(n_to_show):
            pts = crack_polylines[i]

            # scale from normalized [-1,1] coords to body space
            scaled = np.empty_like(pts)
            scaled[:, 0] = x + (pts[:, 0] * radius)
            scaled[:, 1] = y + (pts[:, 1] * radius)

            # line width grows with collision index; first crack is already visible
            
            line = Line2D(
                scaled[:, 0],
                scaled[:, 1],
                linewidth=lw,
                color=self.line_color,  # softer than edgecolor
                solid_capstyle="butt",
                solid_joinstyle="round",
                zorder=3.5,
            )
            line.set_clip_path(circle)          # <-- hard guarantee: never draws outside
            line.set_clip_on(True)
            ax.add_line(line)


    def _generate_cracks_for_body(
        self,
        rng: np.random.Generator,
        n_cracks: int,
    ) -> list[np.ndarray]:
        """
        Generate `n_cracks` polylines in normalized [-1, 1]^2 coords.
        """
        cracks: list[np.ndarray] = []
        #start = rng.normal(loc=0.0, scale=0.1, size=2)
        
        for i in range(n_cracks):
            n_points = rng.integers(6, 10)
            angle1 = rng.uniform(0, 2 * np.pi)
            angle2 = rng.uniform(0, 2 * np.pi)
            #handle different quadrants
            start = np.array([np.cos(angle1), np.sin(angle1)])
            end = np.array([np.cos(angle2), np.sin(angle2)])
            delta = end - start
            length = float(np.linalg.norm(delta))
            perp = np.array([-delta[1], delta[0]])/length
            t_vals = np.linspace(0.0, 1.0, n_points)
            points = np.zeros((n_points, 2), dtype=float)
            for j, t in enumerate(t_vals):
                base = start + t * delta
                jitter_mag = 0.2 * (1.0 - t)
                jitter = jitter_mag * (rng.random() - 0.5) * perp * 2.0
                points[j] = base + jitter if j>0 and j<n_points-1 else base
                if float(np.linalg.norm(points[j])) > 1.0:
                    angle = np.arctan2(points[j,1], points[j,0])
                    points[j] = np.array([np.cos(angle), np.sin(angle)])
            cracks.append(points)

        return cracks