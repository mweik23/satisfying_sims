# src/satisfying_sims/core/recording.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence, Literal, Any
from pathlib import Path
import pickle
import numpy as np
if TYPE_CHECKING:
    from .world import World
    from .events import Event
    from .shapes import Body

@dataclass
class ColliderSnapshot:
    kind: str                 # e.g. "CircleCollider"
    attrs: dict[str, Any]     # serializable parameters, e.g. {"radius": 0.3}

@dataclass
class BodySnapshot:
    id: int
    collider: ColliderSnapshot
    pos: np.ndarray         # shape (2,)
    vel: np.ndarray         # shape (2,)
    mass: float
    color: tuple[float, float, float] | None = None
    # add more fields if you need them for rendering/audio


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
    bodies: dict[int, BodySnapshot]
    events: list[EventSnapshot] = field(default_factory=list)


@dataclass
class SimulationRecording:
    """
    Frozen record of a full simulation run.

    `meta` holds config, version, seed, etc.
    """
    frames: list[FrameSnapshot] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def add_frame(self, frame: FrameSnapshot) -> None:
        self.frames.append(frame)

    @property
    def times(self) -> list[float]:
        return [f.t for f in self.frames]
    
    def save(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "SimulationRecording":
        path = Path(path)
        with path.open("rb") as f:
            rec = pickle.load(f)
        # optional: migrate older versions here
        return rec

def snapshot_world(world, t: float, events: list[Event]) -> FrameSnapshot:
    bodies = {}
    for b in world.bodies.values():
        bodies[b.id] = BodySnapshot(
            id=b.id,
            collider=b.collider.to_snapshot(),
            pos=np.array(b.pos, dtype=float),
            vel=np.array(b.vel, dtype=float),
            mass=float(b.mass),
            color=tuple(c / 255 for c in b.color) if getattr(b, "color", None) is not None else None,
        )

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
    return FrameSnapshot(t=t, bodies=bodies, events=event_snaps)
