# satisfying_sims/rendering/theme_factory.py
from pathlib import Path
from satisfying_sims.themes.base import BodyThemeConfig
from satisfying_sims.themes.sprite import SpriteTheme, SpriteThemeConfig
from satisfying_sims.themes.ice_cracks import IceCracksTheme, IceThemeConfig
from .sprites import build_sprite_paths

def make_body_theme_cfg(args, project_root: Path) -> BodyThemeConfig:
    if args.body_theme == "sprite":
        sprite_dir = project_root / args.sprite_dir
        sprite_paths = build_sprite_paths(sprite_dir, args.sprite_type)
        return SpriteThemeConfig(sprite_paths=sprite_paths, sprite_type=args.sprite_type)
    elif args.body_theme == "ice_cracks":
        print("WARNING: IceCracksTheme is not supported yet.")
        return IceThemeConfig(body_cmap=args.body_cmap)
    else:
        raise ValueError(...)