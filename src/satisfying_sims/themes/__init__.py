from .base import BodyTheme
from .ice_cracks import IceCracksTheme, IceThemeConfig
from .sprite import SpriteTheme, SpriteThemeConfig

THEME_REGISTRY = {
    "sprite": SpriteTheme,
    "ice_cracks": IceCracksTheme,
}