# satisfying_sims/rendering/theme_factory.py
from pathlib import Path
from satisfying_sims.themes.base import BodyThemeConfig
from satisfying_sims.themes.sprite import SpriteThemeConfig
from satisfying_sims.themes.ice_cracks import IceThemeConfig
from .sprites import build_sprite_paths

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