# src/satisfying_sims/core/config.py

from dataclasses import dataclass
import numpy as np
from .world import World
from .boundary import Boundary, BoxBoundary


@dataclass
class SimConfig:
    boundary: Boundary             # <— instead of width/height
    gravity: tuple[float, float] = (0.0, 0.0)
    drag: float = 0.0
    restitution: float = 1.0


def make_world(cfg: SimConfig) -> World:
    return World(
        boundary=cfg.boundary,
        gravity=np.array(cfg.gravity, dtype=float),
        drag=cfg.drag,
        restitution=cfg.restitution,
    )
