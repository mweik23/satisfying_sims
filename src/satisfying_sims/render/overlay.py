from dataclasses import dataclass, field
from PIL import Image
import numpy as np
import matplotlib as mpl
from pathlib import Path

@dataclass
class OverlayConfig:
    png: str | None = None
    size: tuple[float, float] | None = None  # (width, height) in world units
    wall_idx: int = 0                         # which inner wall
    point_idx: int = 1                        # which vertex on that wall
    offset: tuple[float, float] = (0, 0.5)  
    zorder: int = 50
    layer: int = 0
    
def make_overlay_config(overlay_resolved: dict, sprite_dir: Path) -> OverlayConfig:
    overlay_resolved = overlay_resolved or {}
    png = str(sprite_dir / overlay_resolved.get("png", None)) if overlay_resolved.get("png", None) is not None else None
    overlay_resolved["png"] = png
    return OverlayConfig(**overlay_resolved)
    
def create_overlay_artist(ax: mpl.axes.Axes, overlay_cfg: OverlayConfig, plot_layer: int=0) -> mpl.image.AxesImage | None:
    if overlay_cfg.layer != plot_layer:
        return None
    if overlay_cfg.png is None:
        return None
    img = Image.open(overlay_cfg.png).convert("RGBA")
    arr = np.asarray(img)

    overlay_w, overlay_h = overlay_cfg.size
    # initial placement; will be updated each frame
    x0, y0 = 0.0, 0.0
    extent = (x0, x0 + overlay_w, y0, y0 + overlay_h) 
    overlay_artist = ax.imshow(
        arr,
        extent=extent,     # world coords
        origin="lower",
        zorder=overlay_cfg.zorder,         # above bodies/walls; lower if you want it behind
        interpolation="bilinear",
    )
    overlay_artist.set_clip_on(False)
    return overlay_artist