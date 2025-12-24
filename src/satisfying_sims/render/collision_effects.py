from dataclasses import dataclass, fields
from typing import Any

from satisfying_sims.themes.collision_effect import CollisionEffectTheme, CollisionEffectConfig
from satisfying_sims.visual.color_sampler import ColorSampler

@dataclass
class CollisionEffectRule:
    event_type: str                 # "CollisionEvent", "HitWallEvent"
    theme: CollisionEffectTheme

class CollisionEffectRouter:
    def __init__(self, rules: dict[str, CollisionEffectTheme]):
        self.rules = rules  # event_type -> theme

    def clear_cache(self) -> None:
        for t in self.rules.values():
            t.clear_cache()

    def begin_frame(self, frame_idx: int) -> None:
        for t in self.rules.values():
            t.begin_frame(frame_idx)

    def ingest_events(self, event_snaps, snapshot, body_static) -> None:
        # Let each theme pick only the types it cares about
        for t in self.rules.values():
            t.ingest_events(event_snaps, snapshot, body_static)

    def draw(self, ax, frame_idx: int) -> None:
        for t in self.rules.values():
            t.draw(ax, frame_idx)

    def end_frame(self) -> None:
        for t in self.rules.values():
            t.end_frame()

def build_collision_effect_router(
    rules_dict: dict[str, Any],
    *,
    asset_dir: str | None = None,
) -> CollisionEffectRouter | None:
    """
    rules_dict example:
    {
      "CollisionEvent": {"asset": "firework.npz", "size_world": 0.35, "cmap": "viridis"},
      "HitWallEvent":   {"asset": "spark.npz",    "size_scale_with_radius": 3.0}
    }
    """
    if not rules_dict:
        return None
    if not isinstance(rules_dict, dict):
        raise TypeError("collision_effects must be a dict mapping event_type -> config dict")

    # Which keys are legal to pass straight into CollisionEffectConfig(...)
    cfg_field_names = {f.name for f in fields(CollisionEffectConfig)}

    themes: dict[str, CollisionEffectTheme] = {}

    for event_type, raw in rules_dict.items():
        if not isinstance(raw, dict):
            raise TypeError(f"Rule for {event_type} must be a dict, got {type(raw)}")

        raw = dict(raw)  # copy so we can pop

        # Accept either "asset" (relative) and combine with asset_dir
        asset = raw.pop("asset", None)

        if asset is None:
            raise ValueError(f"Rule for {event_type} needs 'asset'")
        effect_path = (
            f"{asset_dir}/{asset}"
            if asset_dir and not str(asset).startswith("/")
            else str(asset)
        )

        # Optional colormap -> ColorSampler
        cmap = raw.pop("cmap", None)
        color_sampler = raw.pop("color_sampler", None)  # allow advanced use
        if color_sampler is None and cmap is not None:
            color_sampler = ColorSampler(cmap)

        # Filter kwargs down to CollisionEffectConfig fields only #TODO: change name mp4_path throughout
        cfg_kwargs = {"effect_path": effect_path, "color_sampler": color_sampler}
        for k, v in raw.items():
            if k in cfg_field_names:
                cfg_kwargs[k] = v
            else:
                raise ValueError(
                    f"Unknown key in collision effect rule for {event_type}: '{k}'. "
                    f"Allowed: asset/effect_path/cmap/color_sampler plus {sorted(cfg_field_names)}"
                )

        theme_cfg = CollisionEffectConfig(**cfg_kwargs)
        theme = CollisionEffectTheme(theme_cfg)

        # Make the theme only respond to that event type
        theme.allowed_event_type = event_type  # simplest; or add to config properly

        themes[event_type] = theme

    return CollisionEffectRouter(themes)