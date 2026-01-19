from collections import deque
from dataclasses import dataclass, fields
from typing import Any

from satisfying_sims.core.recording import SimulationRecording
from satisfying_sims.themes.collision_effect import CollisionEffectTheme, CollisionEffectConfig
from satisfying_sims.visual.color_sampler import ColorSampler
from satisfying_sims.utils.beat_gated_utils import BeatGatedRule
    
class CollisionEffectRouter:
    def __init__(self, rules: list[CollisionEffectTheme]):
        self.rules = rules
    def get_layers(self) -> set[int]:
        return set(t.config.layer for t in self.rules)
    def clear_cache(self, plot_layer=0) -> None:
        for t in self.rules:
            if t.config.layer == plot_layer:
                t.clear_cache()
    def check_windows(self, frame_idx: int, plot_layer=0) -> None:
        for t in self.rules:
            if t.config.effect_type == "segment":
                if t.config.layer == plot_layer:
                    t.check_windows(frame_idx)
                    
    def plot_default(self, ax, plot_layer=0) -> None:
        for t in self.rules:
            if t.config.default_frame is not None and t.config.layer == plot_layer:
                t.plot_default(ax)
    def begin_frame(self, frame_idx: int, plot_layer=0) -> None:
        for t in self.rules:
            if t.config.layer == plot_layer:
                t.begin_frame(frame_idx)

    def ingest_events(self, event_snaps, snapshot, body_static, plot_layer=0) -> None:
        # Let each theme pick only the types it cares about
        for t in self.rules:
            if t.config.effect_type != "segment" and t.config.layer == plot_layer:
                t.ingest_events(event_snaps, snapshot, body_static)

    def draw(self, ax, frame_idx: int, plot_layer=0) -> None:
        for t in self.rules:
            if t.config.layer == plot_layer:
                t.draw(ax, frame_idx)

    def end_frame(self, plot_layer=0) -> None:
        for t in self.rules:
            if t.config.layer == plot_layer:
                t.end_frame()

def build_collision_effect_router(
    one_shot_effects: dict[str, dict[str, Any]] | None,
    beat_gated_effects: dict[str, BeatGatedRule] | None = None,
    *,
    asset_dir: str | None = None,
    fps=60
) -> CollisionEffectRouter | None:
    '''
    rules example:
    [{"event_type": "CollisionEvent", "asset": "firework.npz", "size_world": 0.35, "cmap": "viridis"}]
    '''
    if not one_shot_effects:
        return None
    if not isinstance(one_shot_effects, dict):
        raise TypeError("collision_effects must be a dict")
    if len(one_shot_effects)>0 and any(key not in {"one_shot", "segment"} for key in one_shot_effects.keys()):
        raise ValueError("collision_effects keys must be 'one_shot' and/or 'segment'")
    if any(not isinstance(one_shot_effects[key], list) for key in one_shot_effects.keys()):
        raise TypeError("collision_effects values must be lists of one_shot_effects")
        

    # Which keys are legal to pass straight into CollisionEffectConfig(...)
    cfg_field_names = {f.name for f in fields(CollisionEffectConfig)}

    themes: list[CollisionEffectTheme] = []
    for rule_type, rule_list in one_shot_effects.items():
        for rule in rule_list:
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
            color_arg_names = ['cmap', 'color_override']
            color_arg_vals = [rule.pop(name, None) for name in color_arg_names]
            color_sampler = rule.pop("color_sampler", None)
            if color_sampler is None and any(x is not None for x in color_arg_vals):
                color_kwargs = {name: val for name, val in zip(color_arg_names, color_arg_vals) if val is not None}
                color_sampler = ColorSampler(**color_kwargs)

            # Filter kwargs down to CollisionEffectConfig fields only
            cfg_kwargs = {"effect_path": effect_path, "effect_type": rule_type, "color_sampler": color_sampler, "event_type": rule.pop("event_type")}
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
        
    for name, rule in beat_gated_effects.items():

        # Accept either "asset" (relative) and combine with asset_dir
        overlay_name = rule.overlay_name

        if overlay_name is None:
            raise ValueError(f"Rule for {rule.get('event_type', '<unknown>')} needs 'overlay_name'")
        effect_path = (
            f"{asset_dir}/{overlay_name}"
            if asset_dir and not str(overlay_name).startswith("/")
            else str(overlay_name)
        )
        cfg_kwargs = {"effect_path": effect_path, 'beat_gated_config': rule.config, "effect_type": 'segment'}
        for field_name in cfg_field_names:
            if hasattr(rule, field_name):
                cfg_kwargs[field_name] = getattr(rule, field_name)
        theme_cfg = CollisionEffectConfig(**cfg_kwargs)
        theme = CollisionEffectTheme(theme_cfg)
        if rule.windows is not None and theme_cfg.enabled:
            theme.windows = deque((start*fps, end*fps) for start, end in rule.windows)
        themes.append(theme)
    return CollisionEffectRouter(themes)