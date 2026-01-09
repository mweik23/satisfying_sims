from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def build_sprite_paths(asset_root: Path, sprite_type: str, keys=None) -> dict[str, Path]:
    # e.g. assets/sprites/ornament
    sprite_dir = asset_root / f"{sprite_type}"
    paths = sorted(sprite_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNGs found in {sprite_dir}")

    # key = filename stem (or implement your own mapping)
    if keys is not None:
        return {p.stem: p for p in paths if p.stem in keys}
    
    return {p.stem: p for p in paths}

def load_rgba(path: str | Path) -> np.ndarray:
    """Load an image as RGBA uint8 array of shape (H, W, 4)."""
    return np.asarray(Image.open(path).convert("RGBA"))


def crop_to_alpha(img: np.ndarray, alpha_thresh: int = 1) -> np.ndarray:
    """
    Crop an RGBA image to the tight bounding box of pixels with alpha >= alpha_thresh.
    If the image is fully transparent, returns the original image.
    """
    if img.ndim != 3 or img.shape[-1] != 4:
        raise ValueError(f"Expected RGBA image (H, W, 4), got shape {img.shape}")

    a = img[..., 3]
    ys, xs = np.where(a >= alpha_thresh)
    if len(xs) == 0:
        return img

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return img[y0:y1, x0:x1, :]


def preprocess_sprite(path: str | Path, *, alpha_thresh: int = 1, max_render_px: int | None = None, factor=1.5) -> np.ndarray:
    """
    Load and preprocess a sprite for rendering:
      - load RGBA
      - crop to non-transparent region

    NOTE: We intentionally do NOT pad to square, because the pivot is not the bbox center
    for the ornament use-case (knob above circle).
    """
    img = load_rgba(path)
    img = crop_to_alpha(img, alpha_thresh=alpha_thresh)
    return img


@dataclass(frozen=True)
class SpriteGeom:
    """
    Geometry metadata for mapping a sprite image onto world coordinates.

    cx_px, cy_px: pivot (circle center) in pixel coords within the cropped image.
    r_px: circle radius in pixels.
    """
    cx_px: float
    cy_px: float
    r_px: float


def ornament_geom(img: np.ndarray) -> SpriteGeom:
    """
    Ornament assumption:
      - knob is at top of the cropped image
      - circular part touches the bottom of the cropped image
      - image width corresponds to the circle diameter

    Therefore:
      r_px = W/2
      cx_px = W/2
      cy_px = H - r_px  (one radius above the bottom)
    """
    if img.ndim != 3 or img.shape[-1] != 4:
        raise ValueError(f"Expected RGBA image (H, W, 4), got shape {img.shape}")

    h, w = img.shape[:2]
    r_px = w / 2.0
    cx_px = w / 2.0
    cy_px = h - r_px
    return SpriteGeom(cx_px=cx_px, cy_px=cy_px, r_px=r_px)

def discoball_geom(img: np.ndarray) -> SpriteGeom:
    """
    Disco ball assumption:
      - ball is close to a circle
    
    if there is any oblate distrortion we average the width and height for radius
    the center is at the center of the image
    """
    if img.ndim != 3 or img.shape[-1] != 4:
        raise ValueError(f"Expected RGBA image (H, W, 4), got shape {img.shape}")

    h, w = img.shape[:2]
    r_px = (w+h) / 4.0
    cx_px = w / 2.0
    cy_px = h / 2.0
    return SpriteGeom(cx_px=cx_px, cy_px=cy_px, r_px=r_px)

def firework_rocket_geom(img: np.ndarray) -> SpriteGeom:
    """
    Disco ball assumption:
      - ball is close to a circle
    
    if there is any oblate distrortion we average the width and height for radius
    the center is at the center of the image
    """
    if img.ndim != 3 or img.shape[-1] != 4:
        raise ValueError(f"Expected RGBA image (H, W, 4), got shape {img.shape}")

    h, w = img.shape[:2]
    left = w / 3.0
    right = 0.82 * w
    r_px = (right - left) / 2.0
    cx_px = (left + right) / 2.0
    cy_px = h / 2.0
    return SpriteGeom(cx_px=cx_px, cy_px=cy_px, r_px=r_px)

def fireball_geom(img: np.ndarray) -> SpriteGeom:
    """
    Ornament assumption:
      - knob is at top of the cropped image
      - circular part touches the bottom of the cropped image
      - image width corresponds to the circle diameter

    Therefore:
      r_px = W/2
      cx_px = W/2
      cy_px = H - r_px  (one radius above the bottom)
    """
    if img.ndim != 3 or img.shape[-1] != 4:
        raise ValueError(f"Expected RGBA image (H, W, 4), got shape {img.shape}")

    h, w = img.shape[:2]
    r_px = 0.6 * h
    cx_px = w - r_px
    cy_px = h / 2.0
    return SpriteGeom(cx_px=cx_px, cy_px=cy_px, r_px=r_px)

def grenade_geom(img: np.ndarray) -> SpriteGeom:
    """
    Disco ball assumption:
      - ball is close to a circle
    
    if there is any oblate distrortion we average the width and height for radius
    the center is at the center of the image
    """
    if img.ndim != 3 or img.shape[-1] != 4:
        raise ValueError(f"Expected RGBA image (H, W, 4), got shape {img.shape}")

    h, w = img.shape[:2]
    left = 0
    right = 2.0 * w / 3.0
    bottom = 0.0
    top = 2.0 * h / 3.0

    r_px = ((right-left) + (top-bottom)) / 4.0  
    cx_px = (left + right) / 2.0
    cy_px = (bottom + top) / 2.0
    return SpriteGeom(cx_px=cx_px, cy_px=cy_px, r_px=r_px)

def big_aj_geom(img: np.ndarray) -> SpriteGeom:
    """
    Disco ball assumption:
      - ball is close to a circle
    
    if there is any oblate distrortion we average the width and height for radius
    the center is at the center of the image
    """
    if img.ndim != 3 or img.shape[-1] != 4:
        raise ValueError(f"Expected RGBA image (H, W, 4), got shape {img.shape}")

    h, w = img.shape[:2]
    r_px = (w+h) / 4.0
    cx_px = w / 2.0
    cy_px = h / 2.0
    return SpriteGeom(cx_px=cx_px, cy_px=cy_px, r_px=r_px)

GEOM_REGISTRY = {
    "ornament": ornament_geom,
    "discoball": discoball_geom,
    "firework_rocket": firework_rocket_geom,
    "fireball": fireball_geom,
    "grenade": grenade_geom,
    "big_aj": big_aj_geom,
}

def sprite_extent_for_circle_center(
    *,
    x: float,
    y: float,
    R: float,
    img: np.ndarray,
    geom: SpriteGeom,
) -> Tuple[float, float, float, float]:
    """
    Compute imshow extent so that:
      - the sprite's circle center (geom.cx_px, geom.cy_px) maps to (x, y) in world coords
      - the sprite's circle radius geom.r_px maps to collider radius R

    Returns (x_min, x_max, y_min, y_max).
    """
    h, w = img.shape[:2]
    if geom.r_px <= 0:
        raise ValueError(f"Invalid geom.r_px={geom.r_px}")

    s = float(R) / float(geom.r_px)  # world units per pixel

    x_min = x - geom.cx_px * s
    x_max = x + (w - geom.cx_px) * s
    y_min = y - geom.cy_px * s
    y_max = y + (h - geom.cy_px) * s
    return (x_min, x_max, y_min, y_max)
