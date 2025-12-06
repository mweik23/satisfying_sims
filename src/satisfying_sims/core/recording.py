# src/satisfying_sims/core/recording.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from .world import World
    from .events import Event
    from .shapes import Body


@dataclass
class FrameSnapshot:
    t: float
    bodies: dict[int, "Body"]
    events: list["Event"] = field(default_factory=list)


@dataclass
class SimulationRecording:
    """Simple container for a simulation run."""
    frames: list[FrameSnapshot] = field(default_factory=list)

    def add_frame(self, frame: FrameSnapshot) -> None:
        self.frames.append(frame)

    @property
    def times(self) -> list[float]:
        return [f.t for f in self.frames]
