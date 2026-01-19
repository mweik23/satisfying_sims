from typing import Tuple, Sequence, List, Any
from dataclasses import dataclass, field, fields
from collections import deque

from satisfying_sims.core.events import matches_filter
from satisfying_sims.core.recording import EventContext


def compute_windows(event_times: Sequence[float], *, bpm, grace, gate_mode) -> List[Tuple[float, float]]:
    if gate_mode == "one_beat_after_last":
        return _compute_decay_windows(event_times, bpm)
    elif gate_mode == "to_next_beat":
        return _compute_to_next_beat_windows(event_times, bpm, grace=grace)
    else:
        raise ValueError(f"Unknown gate_mode {gate_mode!r}. Expected 'one_beat_after_last' or 'to_next_beat'.")


def _compute_decay_windows(event_times: Sequence[float], bpm: float) -> List[Tuple[float, float]]:
    if not event_times:
        return []
    P = 60.0 / bpm
    ts = sorted(event_times)
    print("event times:", ts)
    windows: List[Tuple[float, float]] = []
    t0 = ts[0]
    t_end = t0 + P
    
    for t in ts[1:]:
        if t <= t_end:
            t_end = t + P
        else:
            windows.append((t0, t_end))
            t0 = t
            t_end = t + P

    windows.append((t0, t_end))
    print("decay windows:", windows)
    return windows


def _compute_to_next_beat_windows(event_times: Sequence[float], bpm: float, grace: float = 0.0) -> List[Tuple[float, float]]:
    if not event_times:
        return []
    P = 60.0 / bpm
    ts = sorted(event_times)

    windows: List[Tuple[float, float]] = []
    i = 0
    n = len(ts)

    while i < n:
        t0 = ts[i]
        k = 0
        while True:
            interval_start = t0 + k * P
            interval_end   = t0 + (k + 1) * P + grace

            while i < n and ts[i] < interval_start:
                i += 1

            if i < n and ts[i] < interval_end:
                while i < n and ts[i] < interval_end:
                    i += 1
                k += 1
                continue

            t_end = t0 + (k + 1) * P
            windows.append((t0, t_end))
            break

    return windows

@dataclass
class BeatGatedConfig:
    bpm: float
    gate_mode: str = "to_next_beat"   # or "one_beat_after_last"
    grace: float = 0.0
    loop: bool = True
    fade_out: float = 0.01           # seconds
    gain: float = 1.0
    
@dataclass
class BeatGatedRule:
    config: BeatGatedConfig
    event_type: str
    sample_name: str
    overlay_name: str | None = None
    background_name: str | None = None
    event_filter: dict[str, object] = field(default_factory=dict)
    size_world: float = 0.3
    layer: int = 0
    default_frame: int | None = None
    position_override: list[float, float] | None = None
    enabled: bool = True
    windows: list[Tuple[float, float]] | None = None
    
    def set_windows(self, snaps: List[EventContext]):
        # For each beat rule, collect matching event times and compute windows
        type_snaps = [s for s in snaps if s.ev.type == self.event_type]

        times = [
            float(s.ev.t)
            for s in type_snaps
            if matches_filter(s.ev, self.event_filter or {})
        ]
        self.windows = compute_windows(
            times, 
            bpm=self.config.bpm, 
            grace=self.config.grace, 
            gate_mode=self.config.gate_mode
        )
        
def make_beat_gated_rules(rules: dict, snaps: List[EventContext]) -> dict[str, BeatGatedRule]:
    cfg_field_names = {f.name for f in fields(BeatGatedConfig)}
    cfg_kwargs = {}
    for name in cfg_field_names:
        if name in rules:
            cfg_kwargs[name] = rules.pop(name)
    rule = BeatGatedRule(config=BeatGatedConfig(**cfg_kwargs), **rules)
    rule.set_windows(snaps)
    return rule

    