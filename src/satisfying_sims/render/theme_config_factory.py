# satisfying_sims/rendering/theme_factory.py
from pathlib import Path
from satisfying_sims.themes import THEME_REGISTRY
from satisfying_sims.themes.base import BodyTheme, BodyThemeConfig
from satisfying_sims.themes.sprite import SpriteThemeConfig
from satisfying_sims.themes.ice_cracks import IceThemeConfig
from .sprites import build_sprite_paths
from satisfying_sims.core.recording import BodyStaticSnapshot
import importlib

class BodyThemeRouter:
    def __init__(self, themes: dict[str, BodyTheme]):
        self.themes = themes 
    def get_layers(self) -> set[int]:
        return set(t.config.layer for t in self.themes.values())
    def clear_cache(self, plot_layer=0) -> None:
        for t in self.themes.values():
            if t.config.layer == plot_layer:
                t.clear_cache()
    def prepare_for_recording(self, body_static: dict[int, BodyStaticSnapshot], px_per_world, plot_layer=0) -> None:
        for t in self.themes.values():
            if t.config.layer == plot_layer:
                t.prepare_for_recording(body_static=body_static, px_per_world=px_per_world)
    def begin_frame(self, plot_layer=0) -> None:
        for t in self.themes.values():
            if t.config.layer == plot_layer:
                t.begin_frame()

    def draw_bodies(self, ax, body_states, body_static, plot_layer=0) -> None:
        for body_id, state in body_states.items():
            theme = self.themes[body_static[body_id].theme_id]
            if theme.config.layer == plot_layer:
                theme.draw_body(
                    ax=ax,
                    body_id=body_id,
                    state=state,
                    static=body_static[body_id],
                )

    def end_frame(self, plot_layer=0) -> None:
        for t in self.themes.values():
            if t.config.layer == plot_layer:
                t.end_frame()

def make_body_theme_router(theme_configs: dict[str, BodyThemeConfig], body_static: dict[int, BodyStaticSnapshot], ) -> BodyThemeRouter:
    body_themes = {}
    needed = {b.theme_id for b in body_static.values() if b.theme_id is not None}

    # import once (kept from your code even though not used directly)
    importlib.import_module("satisfying_sims.themes", package=__package__)

    for theme_id in needed:
        theme_name = theme_id.split(".")[0]
        ThemeCls = THEME_REGISTRY.get(theme_name, None)
        if ThemeCls is None:
            raise ValueError(f"Unknown theme '{theme_name}'")

        theme_cfg = theme_configs.get(theme_id, None)
        if theme_cfg is None:
            raise ValueError(f"Missing theme config for theme_id {theme_id}")

        body_themes[theme_id] = ThemeCls(config=theme_cfg)
    return BodyThemeRouter(body_themes)


#TODO: decide whether to update to allow multiple sprite types or else require an additional theme for each sprite type
def make_body_theme_cfgs(body_theme_registry, sprite_dir: Path | None = None) -> BodyThemeConfig:
    cfgs = {}
    for name, opts in body_theme_registry.items():
        if "." in name:
            parts = name.split(".")
            theme_name = parts[0]
            kind = parts[1] if len(parts) > 1 else None
        else:
            theme_name = name
            kind = None
        if theme_name == "sprite":
            sprite_paths = build_sprite_paths(sprite_dir, kind, keys=opts.pop("keys", None))
            cfgs[name] = SpriteThemeConfig(
                sprite_paths=sprite_paths,
                sprite_type=kind,
                theme_id=name, 
                **opts
            )   
            
        elif theme_name == "ice_cracks":
            print("WARNING: IceCracksTheme is not supported yet.")
            cfgs[name] = IceThemeConfig(**opts)
        else:
            raise ValueError(f"Unknown theme name: {theme_name}")
    return cfgs
    