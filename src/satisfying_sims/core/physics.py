# src/satisfying_sims/core/physics.py

# src/simproject/core/physics.py

from __future__ import annotations
from typing import List
import numpy as np

from .world import World
from .shapes import Body, CircleCollider
from .events import CollisionEvent, BaseEvent
from .boundary import Boundary


def step_physics(world: World, dt: float) -> List[BaseEvent]:
    """
    Advance physics by dt seconds and return events.
    """
    events: List[BaseEvent] = []

    # Integrate
    for b in world.bodies.values():
        acc = world.gravity - world.drag * b.vel
        b.vel += acc * dt
        b.pos += b.vel * dt

    # Boundary resolution
    for b in world.bodies.values():
        events.extend(world.boundary.resolve_collision(b, world.restitution, world.time + dt))

    # Body-body collisions #TODO: fix
    n = len(world.bodies)
    for i in range(n):
        for j in range(i + 1, n):
            a = world.bodies[i]
            b = world.bodies[j]
            ev = _resolve_body_collision(world, a, b, world.time + dt)
            if ev is not None:
                events.append(ev)

    world.time += dt
    return events


def _resolve_body_collision(world: World, a: Body, b: Body, t: float) -> CollisionEvent | None:
    """
    Dispatch collision logic based on collider types.

    For now, we only implement CircleCollider vs CircleCollider.
    """
    ca, cb = a.collider, b.collider

    if isinstance(ca, CircleCollider) and isinstance(cb, CircleCollider):
        return _circle_circle_collision(world, a, b, t)
    else:
        # future: polygon, etc.
        return None


def _circle_circle_collision(world: World, a: Body, b: Body, t: float) -> CollisionEvent | None:
    delta = b.pos - a.pos
    dist = float(np.linalg.norm(delta))
    ra = a.collider.radius
    rb = b.collider.radius
    radius_sum = ra + rb

    if dist == 0.0:
        n = np.array([1.0, 0.0])
    else:
        n = delta / dist

    if dist >= radius_sum:
        return None

    # relative velocity along normal
    rv = b.vel - a.vel
    vel_along_normal = float(np.dot(rv, n))

    if vel_along_normal > 0:
        _positional_correction(a, b, n, radius_sum - dist)
        return None

    e = world.restitution
    inv_mass_a = 1.0 / a.mass if a.mass > 0 else 0.0
    inv_mass_b = 1.0 / b.mass if b.mass > 0 else 0.0

    j = -(1 + e) * vel_along_normal
    denom = inv_mass_a + inv_mass_b
    if denom == 0:
        return None
    j /= denom

    impulse_vec = j * n
    a.vel -= inv_mass_a * impulse_vec
    b.vel += inv_mass_b * impulse_vec

    penetration = radius_sum - dist
    _positional_correction(a, b, n, penetration)

    contact_point = a.pos + n * ra

    return CollisionEvent(
        t=t,
        a_id=a.id,
        b_id=b.id,
        pos=contact_point,
        impulse=abs(j),
        relative_speed=abs(vel_along_normal),
    )


def _positional_correction(a: Body, b: Body, n: np.ndarray, penetration: float) -> None:
    if penetration <= 0:
        return

    percent = 0.8
    slop = 0.01
    correction_mag = max(penetration - slop, 0.0) * percent

    inv_mass_a = 1.0 / a.mass if a.mass > 0 else 0.0
    inv_mass_b = 1.0 / b.mass if b.mass > 0 else 0.0
    denom = inv_mass_a + inv_mass_b
    if denom == 0:
        return

    correction = (correction_mag / denom) * n
    a.pos -= inv_mass_a * correction
    b.pos += inv_mass_b * correction
