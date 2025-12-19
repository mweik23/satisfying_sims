from __future__ import annotations
import numpy as np
import importlib
from satisfying_sims.core import World, create_circle_body, Boundary, BoxBoundary, Rule, SimConfig, rules, boundary
from satisfying_sims.utils.random import rng
from satisfying_sims.utils.plotting import color_to_uint8
from satisfying_sims.utils.reflection import get_class
from dataclasses import asdict
from satisfying_sims.visual.color_sampler import ColorSampler

def make_world(sim_config: SimConfig) -> World:
    rule_list = [get_class(name, rules)(**params) for name, params in sim_config.rules.items()]  # etc
    print('rule list: ', [vars(rule) for rule in rule_list])
    boundary_obj = get_class(sim_config.boundary['type'], boundary)(**sim_config.boundary['params'])
    if sim_config.body_color is None:
        color_sampler = ColorSampler(cmap=sim_config.body_cmap)
    else:
        color_sampler = None
    world = World(boundary=boundary_obj,
                  rules=rule_list, 
                  gravity=np.array(sim_config.gravity, dtype=float), 
                  drag=sim_config.drag, 
                  restitution=sim_config.restitution,
                  color_sampler=color_sampler)

    for _ in range(sim_config.n_bodies):
        sample = True
        while sample:
            pos = boundary_obj.sample_position(radius=sim_config.radius)
            sample = any(np.linalg.norm(pos - b.pos) < sim_config.radius + b.collider.bounding_radius() for b in world.bodies.values())
        vel = rng("physics").normal(0.0, sim_config.sigma_v, size=2)
        body = create_circle_body(pos=pos, 
                                  vel=vel, 
                                  radius=sim_config.radius,
                                  color_sampler=world.color_sampler,
                                  color=color_to_uint8(sim_config.body_color), 
                                  theme=sim_config.body_theme)
        world.add_body(body)

    return world