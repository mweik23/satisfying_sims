# src/satisfying_sims/audio/mapping.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List

import numpy as np

from satisfying_sims.utils.random import rng
from satisfying_sims.core.recording import EventSnapshot, EventContext
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

    def __init__(self, rules: dict[str, EventSoundRule], keep_prob: Callable | None = None):
        self.rules = rules
        self.keep_prob = keep_prob or (lambda snap, rule_events: 1.0)

    def snapshot_to_trigger(self, snap: EventContext) -> SoundTrigger | None:
        u = rng('audio').random()
        p_accept = self.keep_prob(snap, list(self.rules.keys()))
        if u<=p_accept:
            rule = self.rules.get(snap.ev.type)
        else:
            rule = None
        if rule is None or not rule.enabled:
            return None

        gain = float(rule.gain_fn(snap))
        pitch_ratio = float(rule.pitch_fn(snap))

        return SoundTrigger(
            t=float(snap.ev.t),
            sample_name=rule.sample_name,
            gain=gain,
            pitch_ratio=pitch_ratio,
        )

    def triggers_from_snapshots(
        self, snapshots: Iterable[EventContext]
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
    i0: float = 50,
    sigma_i: float = 25,
    max_factor_log: float = 7/12,
    base : float = 1.0
) -> float:
    """
    Example: gain ∝ impulse (from payload["impulse"]), clipped at max_gain.
    """
    impulse = float(snap.ev.payload.get("impulse", 1.0))
    return base * 2**(max_factor_log * np.tanh((impulse - i0) / sigma_i))


def gain_constant(
    snap: EventSnapshot,
    value: float = 0.7,
) -> float:
    """Constant gain, ignores payload."""
    return value


def pitch_from_relative_speed(
    snap: EventSnapshot,
    v0: float = 50,
    sigma_v: float = 25,
    max_factor_log: float = 7/12,
    base : float = 1.0
) -> float:
    """
    Example: pitch slightly increases with relative_speed in payload.

    Maps relative_speed via tanh to [base - spread, base + spread].
    """
    v = float(snap.ev.payload.get("relative_speed", 1.0))
    
    return  base * 2**(max_factor_log * np.tanh((v - v0) / sigma_v))

def pitch_from_impulse(
    snap: EventSnapshot,
    i0: float = 50,
    sigma_i: float = 25,
    max_factor_log: float = 7/12,
    base : float = 1.0
) -> float:
    """
    Example: pitch slightly increases with impulse in payload.

    Maps impulse via a clipped linear function to [base - 2*spread, base + 2*spread].
    """
    impulse = float(snap.ev.payload.get("impulse", 1.0))
    return base * 2**(max_factor_log * np.tanh((impulse - i0) / sigma_i))
