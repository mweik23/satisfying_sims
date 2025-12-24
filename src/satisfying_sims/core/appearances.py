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
    theme: str
    color_sampler: ColorSampler

    def sample(self, *, override=None, **_):
        color = override if override is not None else rgba_float_to_u8(self.color_sampler.sample())
        return {"theme": self.theme, "color": color}

@dataclass(frozen=True)
class SpriteAppearancePolicy:
    sprite_type: str
    sprite_keys: tuple[str, ...]

    def sample(self, *, override=None, **_):
        key = override if override is not None else rng('color').choice(self.sprite_keys)
        return {"theme": "sprite", "sprite_type": self.sprite_type, "sprite_key": key}
