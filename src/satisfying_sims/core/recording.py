# src/satisfying_sims/core/recording.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence, Literal, Any, Iterator, Union, List, Dict, Optional, Callable  
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
    theme_id: str | None = None
    sprite_key: str | None = None
    tags: Mapping[str, Any] = field(default_factory=dict)
    # add any other truly-static fields here (e.g., radius for circles)
    
@dataclass
class BodyStateSnapshot:
    """Per-frame dynamic state of a body."""
    pos: Tuple[float, float]
    vel: Tuple[float, float]
    collision_count: int
    angle: float = 0.0
    angular_velocity: float = 0.0
    # If you have other evolving fields (e.g. life, state), add them here.

@dataclass
class RuleStateSnapshot:
    """Snapshot of a Rule's state at a given time."""
    name: str
    activated: bool
    t_last_trigger: float | None = None

@dataclass
class EventSnapshot:
    t: float
    type: str               # e.g. "collision", "hit_wall", "spawn", ...
    a_id: int | None = None
    b_id: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    # payload can carry extra info like impulse, wall normal, etc.

@dataclass(frozen=True)
class EventContext:
    ev: EventSnapshot
    rates: dict[str, float]   # or Mapping[str, float]
    frame_index: int | None = None

# --- Boundary recording snapshots ---

@dataclass
class WallStaticSnapshot:
    """Static wall metadata (no points)."""
    id: str
    kind: str                 # e.g. "PolylineWall"
    closed: bool
    one_sided: bool
    normal_sign: float
    constrains_domain: bool
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class BoundaryStaticSnapshot:
    """
    Static boundary definition saved separately from frames.
    - outer: static domain definition (box/ellipse/etc.)
    - inner_walls: static wall metadata (ids, flags), but not time-dependent points
    """
    outer_kind: str
    outer_attrs: dict[str, Any]
    inner_walls: list[WallStaticSnapshot] = field(default_factory=list)


@dataclass
class BoundaryStateSnapshot:
    """
    Per-frame time-dependent boundary state.
    Records only the wall points for inner walls, in the same order as BoundaryStaticSnapshot.inner_walls.
    """
    wall_points: list[list[tuple[float, float]]]  # [wall_index][vertex_index](x,y)

def make_boundary_static_snapshot(boundary: Any) -> BoundaryStaticSnapshot:
    """
    Create the static boundary snapshot.
    Expects something like a WallBoundary with .outer and .walls (inner walls).
    If you store inner/outer walls differently, tweak here.
    """
    outer = getattr(boundary, "outer", boundary)  # allow passing outer directly

    outer_kind = type(outer).__name__
    if outer_kind == "BoxBoundary":
        outer_attrs = {"width": float(outer.width), "height": float(outer.height)}
    elif outer_kind == "EllipseBoundary":
        outer_attrs = {"a": float(outer.a), "b": float(outer.b)}
    else:
        # fallback: try a method if you add one later
        if hasattr(outer, "to_state"):
            outer_attrs = dict(outer.to_state())
        else:
            raise TypeError(f"Unsupported outer boundary type: {outer_kind}")

    inner_walls = []
    for i, w in enumerate(getattr(boundary, "walls", [])):
        wid = getattr(w, "id", None)
        if wid is None:
            wid = f"inner_wall_{i}"

        inner_walls.append(
            WallStaticSnapshot(
                id=str(wid),
                kind=type(w).__name__,
                closed=bool(getattr(w, "closed", False)),
                one_sided=bool(getattr(w, "one_sided", False)),
                normal_sign=float(getattr(w, "normal_sign", 1.0)),
                constrains_domain=bool(getattr(w, "constrains_domain", False)),
                tags=dict(getattr(w, "tags", {})) if getattr(w, "tags", None) is not None else {},
            )
        )

    return BoundaryStaticSnapshot(
        outer_kind=outer_kind,
        outer_attrs=outer_attrs,
        inner_walls=inner_walls,
    )


def make_boundary_state_snapshot(boundary: Any) -> BoundaryStateSnapshot:
    """
    Create the per-frame boundary snapshot (time dependent part only).
    Records w.points for each inner wall in boundary.walls order.
    """
    wall_points: list[list[tuple[float, float]]] = []

    for w in getattr(boundary, "walls", []):
        pts = np.asarray(getattr(w, "points"), dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"Wall points must be (N,2); got {pts.shape} for {type(w).__name__}")
        wall_points.append([(float(x), float(y)) for x, y in pts])

    return BoundaryStateSnapshot(wall_points=wall_points)


@dataclass
class FrameSnapshot:
    t: float
    bodies: dict[int, BodyStateSnapshot]
    events: list[EventSnapshot] = field(default_factory=list)
    rule_state: list[RuleStateSnapshot] | None = None
    body_counts: dict[str, int] | None = None
    rates: dict[str, float] | None = None
    boundary: BoundaryStateSnapshot | None = None


@dataclass
class SimulationRecording:
    """
    Frozen record of a full simulation run.

    `meta` holds config, version, seed, etc.
    """
    frames: list[FrameSnapshot] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    body_static: Dict[int, BodyStaticSnapshot] = field(default_factory=dict)
    boundary_static: BoundaryStaticSnapshot | None = None

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
    
    def iter_event_context(self):
        for i, frame in enumerate(self.frames):
            for ev in frame.events:
                yield EventContext(ev=ev, rates=frame.rates, frame_index=i)
        
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
        theme_id=body.theme_id,
        sprite_key=body.sprite_key,
        tags=body.tags,
        collider=body.collider.to_snapshot(),  # your existing method
    )

def make_body_state_snapshot(body: "Body") -> BodyStateSnapshot:
    """Create dynamic (per-frame) state from a live Body."""
    pos = np.asarray(body.pos, dtype=float)
    vel = np.asarray(body.vel, dtype=float)
    angle = float(getattr(body, "angle", 0.0))
    return BodyStateSnapshot(
        pos=(float(pos[0]), float(pos[1])),
        vel=(float(vel[0]), float(vel[1])),
        angle=angle,
        collision_count=body.collision_count
    )


def snapshot_world(world, t: float, 
    events: list[Event],
    *,
    body_static_registry: Dict[int, BodyStaticSnapshot],
    boundary: Any | None = None,
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
    rule_state = []
    for rule in world.rules:
        rule_state.append(
            RuleStateSnapshot(
                name=type(rule).__name__,
                activated=rule.activated,
            )
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
    boundary_state = make_boundary_state_snapshot(boundary) if boundary is not None else None
    return FrameSnapshot(
        t=t, 
        bodies=bodies_state, 
        events=event_snaps, 
        rule_state=rule_state, 
        body_counts=dict(world.body_counter), 
        rates={},
        boundary=boundary_state
)