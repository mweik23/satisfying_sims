from typing import Protocol, Any
from dataclasses import dataclass
from satisfying_sims.visual.color_sampler import ColorSampler
from satisfying_sims.utils.plotting import rgba_float_to_u8
from satisfying_sims.utils.random import rng

class AppearancePolicy(Protocol):
    def sample(self, rng, **overrides) -> dict[str, Any]:
        ...

@dataclass(frozen=True)
class ColorAppearancePolicy:
    theme_id: str
    theme: str
    color_sampler: ColorSampler

    def sample(self, *, override=None, **_):
        color = override if override is not None else rgba_float_to_u8(self.color_sampler.sample())
        return {"theme": self.theme, "color": color}

@dataclass(frozen=True)
class SpriteAppearancePolicy:
    theme_id: str
    sprite_type: str
    sprite_keys: tuple[str, ...]
    rotation_policy: str = "random"  # "point_forward" or "random"

    def sample(self, *, override=None, **_):
        key = override if override is not None else rng('color').choice(self.sprite_keys)
        return {"theme": "sprite", "sprite_type": self.sprite_type, "sprite_key": key}

@dataclass(frozen=True)
class BodyKind:
    theme: str                 # e.g. "sprite", "solid_color", "ice"
    type: str                  # your coarse bucket (see below)
    subtype: str | None = None # optional refinement
    key: str | None = None     # e.g. sprite_key if you want
    tags: frozenset[str] = frozenset()  # optional labels ("glass", "small", ...)