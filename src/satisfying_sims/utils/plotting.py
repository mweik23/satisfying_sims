from matplotlib import colors
import numpy as np

def color_to_uint8(color_str: str | None) -> tuple[int, int, int]:
    if color_str is None:
        return None
    rgb_float = colors.to_rgb(color_str)       # (0.0–1.0 floats)
    rgb_uint8 = tuple(int(255 * c) for c in rgb_float)
    return rgb_uint8

def get_color(base_color, gamma, light_factor=0.7):
    base = np.array(base_color, dtype=float)
    white = np.ones(3, dtype=float)
    light = white * light_factor + base * (1 - light_factor)
    color = light * (1 - gamma) + base * gamma
    return tuple(color)

def rgba_float_to_u8(rgba):
    return tuple(int(round(255 * c)) for c in rgba)
