from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict

@dataclass
class EventRateEstimator:
    window_s: float = 0.05
    _times: Dict[str, Deque[float]] = field(default_factory=dict)

    def update(self, event_type: str, t: float) -> None:
        q = self._times.setdefault(event_type, deque())
        q.append(float(t))
        self._evict_old(q, float(t))

    def rate(self, event_type: str, t: float) -> float:
        q = self._times.setdefault(event_type, deque())
        self._evict_old(q, float(t))
        return len(q) / self.window_s

    def rates(self, t: float) -> Dict[str, float]:
        """Rates for all event types at time t."""
        t = float(t)
        out: Dict[str, float] = {}
        for etype, q in self._times.items():
            self._evict_old(q, t)
            out[etype] = len(q) / self.window_s
        return out

    def _evict_old(self, q: Deque[float], t: float) -> None:
        cutoff = t - self.window_s
        while q and q[0] <= cutoff:
            q.popleft()
