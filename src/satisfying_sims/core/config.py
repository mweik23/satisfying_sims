# src/satisfying_sims/core/config.py

from dataclasses import dataclass, field, fields
import numpy as np
from .world import World
from .boundary import Boundary, BoxBoundary
from typing import List, Optional
from satisfying_sims.themes.base import BodyTheme


@dataclass
class SimConfig:
    boundary: dict = field(default_factory=lambda: {'type': 'BoxBoundary', 'params': {'width': 100.0, 'height': 100.0}})
    rules: dict[str, dict] = field(default_factory=dict) # e.g., {'SpawnOnCollision': {'vel_kick': 2.0}}
    gravity: tuple[float, float] = (0.0, 0.0)
    drag: float = 0.0
    restitution: float = 1.0
    n_bodies: int = 10
    body_color: str = "blue"
    sigma_v: float = 5.0
    radius: float = 1.0
    body_theme: str | None = None
    
    @classmethod
    def from_args(cls, args) -> "SimConfig":
        kwargs={}
        for f in fields(cls):
            name = f.name
            if hasattr(args, name):
                kwargs[name] = getattr(args, name)
        return cls(**kwargs)
