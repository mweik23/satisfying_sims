# src/satisfying_sims/themes/base.py

from __future__ import annotations

from typing import Protocol
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from satisfying_sims.core.recording import (
    SimulationRecording,
    BodyStaticSnapshot,
    BodyStateSnapshot,
)


class BodyTheme(Protocol):
    """
    A pluggable visual theme for drawing bodies.

    Implementations can:
      - preprocess a recording (e.g. to compute collision stats),
      - override how each body is drawn for a frame.
    """

    def prepare_for_recording(self, recording: SimulationRecording) -> None:
        """Optional preprocessing step. Called once before rendering."""
        ...

    def draw_body(
        self,
        ax: Axes,
        body_id: int,
        state: BodyStateSnapshot,
        static: BodyStaticSnapshot,
    ) -> None:
        """
        Draw a single body for a given frame.

        Implementations can draw everything (body + cracks), or call back into
        a “base” drawing function if you want to layer on top of defaults.

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
        fc = static.color                   # already normalized (0–1) tuple
        collider = static.collider          # ColliderSnapshot

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
