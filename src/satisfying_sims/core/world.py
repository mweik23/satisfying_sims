# src/satisfying_sims/core/world.py

from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
from .recording import SimulationRecording, FrameSnapshot
from dataclasses import dataclass, field
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from .rules import Rule
from satisfying_sims.utils.plotting import get_color

from .shapes import Body, CircleCollider
from .boundary import Boundary
from .physics import step_physics
from .events import BaseEvent
from .recording import snapshot_world, BodyStaticSnapshot
from satisfying_sims.audio.estimate_rate import EventRateEstimator
from .appearances import AppearancePolicy

if TYPE_CHECKING:
    from .rules import Rule

@dataclass
class World:
    boundary: Boundary
    gravity: np.ndarray       # (2,)
    drag: float
    restitution: float
    bodies: dict[int, Body] = field(default_factory=dict)
    appearance_policy: AppearancePolicy | None = None
    time: float = 0.0
    _next_id: int = 0
    rules: List[Rule] = field(default_factory=list)
    
    @property
    def n_bodies(self) -> int:
        return len(self.bodies)

    def new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def add_body(self, body: Body) -> None:
        if not self.boundary.contains(body.pos, radius=body.collider.bounding_radius()):
            raise ValueError("Body placed outside boundary")
        if body.id is None:
            body.id = self.new_id()
        elif body.id in self.bodies:
            # either raise or reassign — your choice
            raise ValueError(f"Body id {body.id} already exists in world")
        self.bodies[body.id] = body
        return None

    def remove_body_by_id(self, body_id: int) -> None:
        self.bodies = [b for b in self.bodies if b.id != body_id]
    
    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)
    
    def step(self, dt: float, max_rule_passes: int = 100) -> List[BaseEvent]:
        """
        Advance the world by dt using an event-queue system:
        - Run physics to generate initial events.
        - Repeatedly apply rules to *new* events only.
        - Each event is processed by rules once.
        """
        physics_events = step_physics(self, dt=dt)

        pending_events: List[BaseEvent] = list(physics_events)
        all_events: List[BaseEvent] = []

        passes = 0
        while pending_events and passes < max_rule_passes:
            passes += 1

            # Copy pending events and clear the queue
            current_batch = list(pending_events)
            pending_events.clear()

            # Apply rules to the new events
            for rule in self.rules:
                produced = rule.apply(self, current_batch, dt)
                if produced:
                    pending_events.extend(produced)

            # Mark current batch as processed
            all_events.extend(current_batch)

        return all_events
        
    def plot(self, ax=None, delta=0, include_boundary=True, gamma=0):
        """
        Plot the world: boundary in gray, bodies in their assigned colors.
        Coordinate system: x-right, y-down.
        """
        if include_boundary:
            fig, ax = self.boundary.plot(delta=delta, edgecolor="gray", linewidth=2, linestyle="--", ax=ax)
        else:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure

        # --- Draw bodies ---
        for id, body in self.bodies.items():
            col = body.collider
            if hasattr(col, "radius"):  # CircleCollider
                circ = Circle(
                    (body.pos[0], body.pos[1]),
                    col.radius,
                    edgecolor="black",
                    facecolor=get_color(tuple(c/255 for c in body.color), gamma),
                    linewidth=.35,
                )
                ax.add_patch(circ)
            else:
                raise NotImplementedError("Plotting only supports CircleCollider for now.")

        # y increases downward
        ax.invert_yaxis()

        return fig, ax
    
def run_simulation(
    world: World,
    n_steps: int,
    dt: float,
    log_interval: int = 600,
    *,
    record_events: bool = True,
) -> SimulationRecording:
    """
    Step the world forward n_steps and record snapshots for video/audio.
    """
    recording = SimulationRecording()
    body_static: dict[int, BodyStaticSnapshot] = {}
    rate_est = EventRateEstimator()
    for step in range(n_steps):
        all_events = world.step(dt)  # or whatever your integrator API is
        frame_events = all_events if record_events else []
        snapshot = snapshot_world(world, 
                                  t=world.time, 
                                  events=frame_events,
                                  body_static_registry=body_static)
        for e in snapshot.events:
            rate_est.update(e.type, snapshot.t)
        lam = rate_est.rates(snapshot.t)
        snapshot.rates = lam
        recording.add_frame(snapshot)
        recording.body_static.update(body_static)
        if (step + 1) % log_interval == 0:
            print(f"Simulated {world.time:.3f} seconds / {n_steps*dt:.3f} seconds...")
            print(f"Number of bodies: {len(world.bodies)}")

    return recording
