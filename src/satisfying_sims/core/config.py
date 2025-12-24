# src/satisfying_sims/core/config.py

from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import List, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from satisfying_sims.themes.base import BodyTheme


@dataclass
class SimConfig:
    boundary: dict = field(default_factory=lambda: {'type': 'BoxBoundary', 'params': {'width': 100.0, 'height': 100.0}})
    rules: dict[str, dict] = field(default_factory=dict) # e.g., {'SpawnOnCollision': {'vel_kick': 2.0}}
    gravity: tuple[float, float] = (0.0, 0.0)
    drag: float = 0.0
    restitution: float = 1.0
    n_bodies: int = 10
    body_color: str | None = "blue"
    body_cmap: str | None = None
    sigma_v: float = 5.0
    radius: float = 1.0
    body_theme_cfg: Optional[BodyTheme] = None
    
    
    @classmethod
    def from_args(cls, args, body_theme_cfg: Optional[BodyTheme] = None, world_aspect: float | None = None) -> "SimConfig":
        kwargs={}
        for f in fields(cls):
            name = f.name
            if hasattr(args, name):
                #first special cases
                if name == 'gravity':
                    kwargs[name] = (0.0, getattr(args, 'gravity'))
                #all other cases
                else:
                    kwargs[name] = getattr(args, name)
        if body_theme_cfg is not None:
            kwargs['body_theme_cfg'] = body_theme_cfg
        if world_aspect is not None:
            kwargs['boundary']['params']['height'] = kwargs['boundary']['params']['width'] / world_aspect
        return cls(**kwargs)
