# src/satisfying_sims/core/recording.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence, Literal, Any, Iterator
from pathlib import Path
import pickle
import lzma
import numpy as np
from typing import Dict, Tuple
if TYPE_CHECKING:
    from .world import World
    from .events import Event
    from .shapes import Body

@dataclass
class ColliderSnapshot:
    kind: str                 # e.g. "CircleCollider"
    attrs: dict[str, Any]     # serializable parameters, e.g. {"radius": 0.3}
'''
@dataclass
class BodySnapshot:
    id: int
    collider: ColliderSnapshot
    pos: np.ndarray         # shape (2,)
    vel: np.ndarray         # shape (2,)
    mass: float
    color: tuple[float, float, float] | None = None
    # add more fields if you need them for rendering/audio
'''

@dataclass
class BodyStaticSnapshot:
    """Static properties of a body, stored once per recording."""
    id: int
    mass: float
    color: Tuple[float, float, float]  # normalized 0–1 for matplotlib
    collider: "ColliderSnapshot"       # your existing ColliderSnapshot
    # add any other truly-static fields here (e.g., radius for circles)
    
@dataclass
class BodyStateSnapshot:
    """Per-frame dynamic state of a body."""
    pos: Tuple[float, float]
    vel: Tuple[float, float]
    # If you have other evolving fields (e.g. life, state), add them here.


@dataclass
class EventSnapshot:
    t: float
    type: str               # e.g. "collision", "hit_wall", "spawn", ...
    a_id: int | None = None
    b_id: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    # payload can carry extra info like impulse, wall normal, etc.

@dataclass
class FrameSnapshot:
    t: float
    bodies: dict[int, BodyStateSnapshot]
    events: list[EventSnapshot] = field(default_factory=list)


@dataclass
class SimulationRecording:
    """
    Frozen record of a full simulation run.

    `meta` holds config, version, seed, etc.
    """
    frames: list[FrameSnapshot] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    body_static: Dict[int, BodyStaticSnapshot] = field(default_factory=dict)

    def add_frame(self, frame: FrameSnapshot) -> None:
        self.frames.append(frame)

    @property
    def times(self) -> list[float]:
        return [f.t for f in self.frames]
    
    @property
    def t_end(self) -> float | None:
        """Time of the last frame, or None if no frames."""
        if not self.frames:
            return None
        return self.frames[-1].t

    def iter_events(self) -> Iterator[EventSnapshot]:
        """Iterate over all EventSnapshots in time order."""
        for frame in self.frames:
            for ev in frame.events:
                yield ev
    
    def save(self, path: str | Path) -> None:
        path = Path(path)
        with lzma.open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "SimulationRecording":
        path = Path(path)
        with lzma.open(path, "rb") as f:
            rec = pickle.load(f)
        # optional: migrate older versions here
        return rec

def make_body_static_snapshot(body: "Body") -> BodyStaticSnapshot:
    """Create static snapshot from a live Body."""
    
    return BodyStaticSnapshot(
        id=body.id,
        mass=float(body.mass),
        color=tuple(c / 255 for c in body.color) if getattr(body, "color", None) is not None else None,
        collider=body.collider.to_snapshot(),  # your existing method
    )

def make_body_state_snapshot(body: "Body") -> BodyStateSnapshot:
    """Create dynamic (per-frame) state from a live Body."""
    pos = np.asarray(body.pos, dtype=float)
    vel = np.asarray(body.vel, dtype=float)
    return BodyStateSnapshot(
        pos=(float(pos[0]), float(pos[1])),
        vel=(float(vel[0]), float(vel[1])),
    )


def snapshot_world(world, t: float, 
    events: list[Event],
    *,
    body_static_registry: Dict[int, BodyStaticSnapshot]
) -> FrameSnapshot:
    
    bodies_state: Dict[int, BodyStateSnapshot] = {}

    for body in world.bodies.values():
        if body.id is None:
            raise ValueError("All bodies must have an id before snapshotting")

        # Ensure static snapshot exists exactly once per body id
        if body.id not in body_static_registry:
            body_static_registry[body.id] = make_body_static_snapshot(body)

        # Per-frame dynamic state
        bodies_state[body.id] = make_body_state_snapshot(body)

    event_snaps: list[EventSnapshot] = []
    for e in events:
        # adapt to your Event hierarchy
        event_snaps.append(
            EventSnapshot(
                t=t,
                type=type(e).__name__,
                a_id=getattr(e, "a_id", None),
                b_id=getattr(e, "b_id", None),
                payload=e.to_payload_dict()
            )
        )
    return FrameSnapshot(t=t, bodies=bodies_state, events=event_snaps)
