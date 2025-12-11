# src/satisfying_sims/audio/mapping.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List

import numpy as np

from satisfying_sims.core.recording import EventSnapshot
from .engine import SoundTrigger

# Functions map EventSnapshot -> float
GainFn = Callable[[EventSnapshot], float]
PitchFn = Callable[[EventSnapshot], float]


@dataclass
class EventSoundRule:
    """
    Configure sound for a given event type (EventSnapshot.type).

    EventSnapshot.type is set as type(e).__name__ in snapshot_world,
    e.g. "CollisionEvent", "BoundaryCollisionEvent", etc.
    """
    sample_name: str
    gain_fn: GainFn
    pitch_fn: PitchFn
    enabled: bool = True


class EventSoundMapper:
    """
    Maps EventSnapshot objects to SoundTrigger objects.

    Rules are keyed by EventSnapshot.type, e.g. "CollisionEvent".
    """

    def __init__(self, rules: dict[str, EventSoundRule]):
        self.rules = rules

    def snapshot_to_trigger(self, snap: EventSnapshot) -> SoundTrigger | None:
        rule = self.rules.get(snap.type)
        if rule is None or not rule.enabled:
            return None

        gain = float(rule.gain_fn(snap))
        pitch_ratio = float(rule.pitch_fn(snap))

        return SoundTrigger(
            t=float(snap.t),
            sample_name=rule.sample_name,
            gain=gain,
            pitch_ratio=pitch_ratio,
        )

    def triggers_from_snapshots(
        self, snapshots: Iterable[EventSnapshot]
    ) -> List[SoundTrigger]:
        out: List[SoundTrigger] = []
        for snap in snapshots:
            trig = self.snapshot_to_trigger(snap)
            if trig is not None:
                out.append(trig)
        return out


# ---------- Example gain / pitch functions for EventSnapshot ---------- #

def gain_from_impulse(
    snap: EventSnapshot,
    scale: float = 0.05,
    max_gain: float = 1.0,
) -> float:
    """
    Example: gain ∝ impulse (from payload["impulse"]), clipped at max_gain.
    """
    impulse = float(snap.payload.get("impulse", 1.0))
    return min(max_gain, scale * abs(impulse))


def gain_constant(
    snap: EventSnapshot,
    value: float = 0.7,
) -> float:
    """Constant gain, ignores payload."""
    return value


def pitch_from_relative_speed(
    snap: EventSnapshot,
    base: float = 1.0,
    spread: float = 0.2,
    v_scale: float = 0.1,
) -> float:
    """
    Example: pitch slightly increases with relative_speed in payload.

    Maps relative_speed via tanh to [base - spread, base + spread].
    """
    v = float(snap.payload.get("relative_speed", 1.0))
    offset = spread * np.tanh(v_scale * v)
    return base + offset
