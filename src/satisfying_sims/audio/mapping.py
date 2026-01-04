# src/satisfying_sims/audio/mapping.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Mapping, Sequence, Any

import numpy as np
from functools import partial

from satisfying_sims.utils.random import rng
from satisfying_sims.utils.reflection import get_path
from satisfying_sims.core.recording import EventSnapshot, EventContext
from .engine import SoundTrigger

# Functions map EventSnapshot -> float
GainFn = Callable[[EventSnapshot], float]
PitchFn = Callable[[EventSnapshot], float]

@dataclass(frozen=True)
class EventSoundRule:
    """
    A rule that can match an event snapshot (EventContext) and produce a SoundTrigger.

    - event_type: matches snap.ev.type (e.g. "CollisionEvent")
    - event_filter: dict of dotted paths -> expected values (exact match)
    - priority: higher wins if multiple rules match
    """
    event_type: str
    sample_name: str
    gain_fn: GainFn
    pitch_fn: PitchFn
    event_filter: Mapping[str, Any] = field(default_factory=dict)
    priority: int = 0
    enabled: bool = True

    @property
    def specificity(self) -> int:
        # simple measure: more filter keys => more specific
        return len(self.event_filter)


class EventSoundMapper:
    """
    Maps EventContext snapshots to SoundTrigger objects using a list of rules.
    """

    def __init__(
        self,
        rules: Mapping[str, EventSoundRule],
        keep_prob: Callable[[Any, Sequence[EventSoundRule]], float] | None = None,
    ):
        self.rules = list(rules.values())
        self.keep_prob = keep_prob or (lambda snap, rules: 1.0)

        # Optional: pre-group by event_type for speed
        self._rules_by_type: dict[str, list[EventSoundRule]] = {}
        for r in self.rules:
            self._rules_by_type.setdefault(r.event_type, []).append(r)

    def _matches_filter(self, snap: Any, rule: EventSoundRule) -> bool:
        # `snap` is EventContext; event object is snap.ev
        ev = snap.ev
        for k, expected in rule.event_filter.items():
            actual = getattr(ev, k, None)
            if actual is None:
                #check payload too
                actual = ev.payload.get(k, None)
                if actual is None:
                    print('WARNING: key not found in event or payload: ', k)
            if actual != expected:
                return False
        return True

    def _select_rule(self, snap: Any) -> EventSoundRule | None:
        # rejection sampling gate (your existing behavior)
        if not snap.ev.type == "CollisionEvent" or snap.ev.payload.get('same_type', True):
            u = rng("audio").random()
            event_types = list(self._rules_by_type.keys())
            p_accept = float(self.keep_prob(snap, event_types))
            if u > p_accept:
                return None

        candidates = [
            r for r in self._rules_by_type.get(snap.ev.type, [])
            if r.enabled and self._matches_filter(snap, r)
        ]
        if not candidates:
            return None

        # tie-breaking:
        # 1) highest priority
        # 2) most specific (more filter keys)
        # 3) random among exact ties (so you can have multiple variants)
        candidates.sort(key=lambda r: (r.priority, r.specificity), reverse=True)

        best_priority = candidates[0].priority
        best_specificity = candidates[0].specificity
        best = [r for r in candidates if r.priority == best_priority and r.specificity == best_specificity]

        if len(best) == 1:
            return best[0]
        return rng("audio").choice(best)

    def snapshot_to_trigger(self, snap: Any):  # -> SoundTrigger | None
        rule = self._select_rule(snap)
        if rule is None:
            return None

        gain = float(rule.gain_fn(snap))
        pitch_ratio = float(rule.pitch_fn(snap))

        return SoundTrigger(
            t=float(snap.ev.t),
            sample_name=rule.sample_name,
            gain=gain,
            pitch_ratio=pitch_ratio,
        )

    def triggers_from_snapshots(self, snapshots: Iterable[Any]) -> List[Any]:  # List[SoundTrigger]
        out: List[Any] = []
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

GAIN_FNS: dict[str, Callable[..., float]] = {
    "gain_from_impulse": gain_from_impulse,
    "gain_constant": gain_constant,
}

PITCH_FNS: dict[str, Callable[..., float]] = {
    "pitch_from_relative_speed": pitch_from_relative_speed,
    "pitch_from_impulse": pitch_from_impulse,
}

def build_fn(
    spec: Mapping[str, Any] | None,
    *,
    registry: dict[str, Callable[..., float]],
    default: Callable[[Any], float],
) -> Callable[[Any], float]:
    """
    spec:
      {"fn": "gain_from_impulse", "kwargs": {...}}
    Returns a function snap -> float.
    """
    if not spec:
        return default

    name = spec.get("fn")
    if not isinstance(name, str) or name not in registry:
        raise ValueError(f"Unknown fn {name!r}. Allowed: {sorted(registry)}")

    kwargs = spec.get("kwargs", {}) or {}
    if not isinstance(kwargs, dict):
        raise ValueError(f"kwargs must be a dict for fn {name!r}")

    # Create snap -> float function with those kwargs bound.
    # Assumes the underlying function signature is like f(snap, **kwargs).
    return partial(registry[name], **kwargs)

def rules_from_config(
    cfg: dict[str, dict[str, Any]],
    *,
    default_gain_fn: GainFn,
    default_pitch_fn: PitchFn,
) -> list[EventSoundRule]:
    rules: dict[str, EventSoundRule] = {}
    for key, item in cfg.items():
        event_type = item["event_type"]
        sample_name = item["asset"]
        event_filter = item.get("event_filter", {}) or {}
        priority = int(item.get("priority", 0))
        enabled = bool(item.get("enabled", True))

        gain_fn = build_fn(item.get("gain"), registry=GAIN_FNS, default=default_gain_fn)
        pitch_fn = build_fn(item.get("pitch"), registry=PITCH_FNS, default=default_pitch_fn)

        rules[key] = EventSoundRule(
            event_type=event_type,
            sample_name=sample_name,
            gain_fn=gain_fn,
            pitch_fn=pitch_fn,
            event_filter=event_filter,
            priority=priority,
            enabled=enabled,
        )
    return rules