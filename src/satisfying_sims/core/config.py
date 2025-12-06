# src/satisfying_sims/core/config.py

from dataclasses import dataclass
import numpy as np
from .world import World
from .boundary import Boundary, BoxBoundary
from typing import List, Optional
from .rules import Rule


@dataclass
class SimConfig:
    boundary: Boundary             # <— instead of width/height
    rules: Optional[List[Rule]] = None
    gravity: tuple[float, float] = (0.0, 0.0)
    drag: float = 0.0
    restitution: float = 1.0


def make_world(cfg: SimConfig) -> World:
    return World(
        boundary=cfg.boundary,
        gravity=np.array(cfg.gravity, dtype=float),
        drag=cfg.drag,
        restitution=cfg.restitution,
        rules=cfg.rules if cfg.rules is not None else [],
    )
