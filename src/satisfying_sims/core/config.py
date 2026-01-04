# src/satisfying_sims/core/config.py

from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import List, TYPE_CHECKING


if TYPE_CHECKING:
    from satisfying_sims.themes.base import BodyThemeConfig
import numpy as np
from .world import World
from .shapes import create_circle_body
from . import rules
from . import boundary 
from .boundary import WallBoundary
from satisfying_sims.utils.random import rng
from satisfying_sims.utils.reflection import get_class
from satisfying_sims.core.appearances import SpriteAppearancePolicy


@dataclass
class SimConfig:
    outer_boundary: dict = field(default_factory=lambda: {'type': 'BoxBoundary', 'params': {'width': 100.0, 'height': 100.0}})
    inner_walls: List[dict] | None = None
    rules: dict[str, dict] = field(default_factory=dict) # e.g., {'SpawnOnCollision': {'vel_kick': 2.0}}
    init_groups: dict | None = None
    body_theme_cfgs: dict[str, BodyThemeConfig] | None = None
    gravity: tuple[float, float] = (0.0, 0.0)
    drag: float = 0.0
    
    
    @classmethod
    def from_args(cls, args, world_aspect: float | None = None, **kwargs) -> "SimConfig":
        for f in fields(cls):
            name = f.name
            if hasattr(args, name):
                #first special cases
                if name == 'gravity':
                    kwargs[name] = (0.0, getattr(args, 'gravity'))
                #all other cases
                else:
                    kwargs[name] = getattr(args, name)
        if world_aspect is not None:
            kwargs['outer_boundary']['params']['height'] = kwargs['outer_boundary']['params']['width'] / world_aspect
        return cls(**kwargs)

def sample_pos_vel(world, radius, pos_policy=None, vel_policy=None, **kwargs):
    sample = True
    while sample:
        pos = world.boundary.sample_position(radius=radius, policy=pos_policy, **kwargs.get('pos', {}))
        sample = any(np.linalg.norm(pos - b.pos) < radius + b.collider.bounding_radius() for b in world.bodies.values())
    vel_kwargs = kwargs.get('vel', {})
    if vel_policy is None:
        vel_policy = 'normal'
    if vel_policy == 'normal':
        vel = rng("physics").normal(vel_kwargs.get('mu_v', np.zeros(2)), vel_kwargs.get('sigma_v', 10*np.ones(2)))
    elif vel_policy == 'exact':
        vel = np.array(vel_kwargs.get('coords', np.zeros(2)))
    else:
        raise ValueError(f'vel_policy {vel_policy} not defined')
    return pos, vel

def make_world(sim_config: SimConfig) -> World:
    rule_list = [get_class(name, rules)(**params) for name, params in sim_config.rules.items()]  # etc
    outer = get_class(sim_config.outer_boundary['type'], boundary)(**sim_config.outer_boundary['params'])

    inner_walls = [get_class(inner_wall['type'], boundary)(**inner_wall['params']) for inner_wall in (sim_config.inner_walls or [])]
    boundary_full = WallBoundary(
        outer=outer,
        walls=inner_walls,
    )
    appearance_policies = {name: cfg.make_appearance_policy(theme_id=name) for name, cfg in (sim_config.body_theme_cfgs or {}).items()}
    world = World(boundary=boundary_full,
                  rules=rule_list, 
                  gravity=np.array(sim_config.gravity, dtype=float), 
                  drag=sim_config.drag, 
                  appearance_policies=appearance_policies)
    
    for name, init_cfg in (sim_config.init_groups or {}).items():
        if name not in appearance_policies:
            raise ValueError(f"Appearance policy for '{name}' not found in body_theme_cfgs.")
        appearance_policy = appearance_policies[name]

        keys = init_cfg.get(
            'keys', 
            list(appearance_policy.sprite_keys) if isinstance(appearance_policy, SpriteAppearancePolicy) else []
        )
        num_keys = len(keys)
        
        spawn_kwargs = init_cfg.get('spawn_kwargs', [{}] * init_cfg['count'])
    
        for i in range(init_cfg['count']):
            radius = init_cfg.get('size', 1.0)
            pos, vel = sample_pos_vel(
                world, 
                radius, 
                pos_policy=init_cfg.get('pos_policy', None), 
                vel_policy=init_cfg.get('vel_policy', None), 
                **spawn_kwargs[i]
            )
            body = create_circle_body(
                pos=pos, 
                vel=vel, 
                radius=radius,
                theme_id=name,
                restitution=init_cfg.get('restitution', 1.0),
                gravity_enabled=init_cfg.get('gravity_enabled', True),
                appearance_policy=appearance_policy,
                override=keys[i % num_keys] if isinstance(appearance_policy, SpriteAppearancePolicy) else None,
            )
            world.add_body(body)

    return world