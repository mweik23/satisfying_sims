from dataclasses import dataclass, fields
from typing import Any

from satisfying_sims.themes.collision_effect import CollisionEffectTheme, CollisionEffectConfig
from satisfying_sims.visual.color_sampler import ColorSampler

@dataclass
class CollisionEffectRule:
    event_type: str                 # "CollisionEvent", "HitWallEvent"
    theme: CollisionEffectTheme
    
class CollisionEffectRouter:
    def __init__(self, rules: list[CollisionEffectTheme]):
        self.rules = rules 

    def clear_cache(self) -> None:
        for t in self.rules:
            t.clear_cache()

    def begin_frame(self, frame_idx: int) -> None:
        for t in self.rules:
            t.begin_frame(frame_idx)

    def ingest_events(self, event_snaps, snapshot, body_static) -> None:
        # Let each theme pick only the types it cares about
        for t in self.rules:
            t.ingest_events(event_snaps, snapshot, body_static)

    def draw(self, ax, frame_idx: int) -> None:
        for t in self.rules:
            t.draw(ax, frame_idx)

    def end_frame(self) -> None:
        for t in self.rules:
            t.end_frame()

def build_collision_effect_router(
    rules: list[dict[str, Any]] | None,
    *,
    asset_dir: str | None = None,
) -> CollisionEffectRouter | None:
    '''
    rules example:
    [{"event_type": "CollisionEvent", "asset": "firework.npz", "size_world": 0.35, "cmap": "viridis"}]
    '''
    if not rules:
        return None
    if not isinstance(rules, list):
        raise TypeError("collision_effects must be a list of dicts")

    # Which keys are legal to pass straight into CollisionEffectConfig(...)
    cfg_field_names = {f.name for f in fields(CollisionEffectConfig)}

    themes: list[CollisionEffectTheme] = []

    for rule in rules:
        if not isinstance(rule, dict):
            raise TypeError(f"Rule must be a dict, got {type(rule)}")

        rule = dict(rule)  # copy so we can pop
        # Accept either "asset" (relative) and combine with asset_dir
        asset = rule.pop("asset", None)

        if asset is None:
            raise ValueError(f"Rule for {rule.get('event_type', '<unknown>')} needs 'asset'")
        effect_path = (
            f"{asset_dir}/{asset}"
            if asset_dir and not str(asset).startswith("/")
            else str(asset)
        )

        # Optional colormap -> ColorSampler
        cmap = rule.pop("cmap", None)
        color_sampler = rule.pop("color_sampler", None)  # allow advanced use
        if color_sampler is None and cmap is not None:
            color_sampler = ColorSampler(cmap)

        # Filter kwargs down to CollisionEffectConfig fields only
        cfg_kwargs = {"effect_path": effect_path, "color_sampler": color_sampler, "event_type": rule.pop("event_type")}
        for k, v in rule.items():
            if k in cfg_field_names:
                cfg_kwargs[k] = v
            else:
                raise ValueError(
                    f"Unknown key in collision effect rule for {rule.get('event_type', '<unknown>')}: '{k}'. "
                    f"Allowed: asset/effect_path/cmap/color_sampler plus {sorted(cfg_field_names)}"
                )

        theme_cfg = CollisionEffectConfig(**cfg_kwargs)
        theme = CollisionEffectTheme(theme_cfg)
        themes.append(theme)

    return CollisionEffectRouter(themes)