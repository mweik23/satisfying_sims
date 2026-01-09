from __future__ import annotations

from dataclasses import dataclass
from PIL import Image
from typing import Any, Dict, Mapping, Set

import matplotlib.transforms as mtransforms
import numpy as np

from satisfying_sims.render.sprites import (
    SpriteGeom,
    GEOM_REGISTRY,
)
from satisfying_sims.core.recording import BodyStaticSnapshot
from .base import BodyTheme, BodyThemeConfig
from satisfying_sims.render.sprites import preprocess_sprite
from pathlib import Path
from satisfying_sims.core.appearances import SpriteAppearancePolicy


@dataclass(frozen=True)
class SpriteAsset:
    """
    A sprite image plus geometry metadata for placement/scaling.
    """
    image: np.ndarray
    geom: SpriteGeom

@dataclass(frozen=True)
class SpriteThemeConfig(BodyThemeConfig):
    sprite_paths: dict[str, Path]
    sprite_type: str = "ornament"  # future-proof
    alpha_thresh: int = 25
    theme_id: str = "sprite."
    zorder: int = 5
    interpolation: str = "bilinear"
    origin: str = "upper"
    rotation_policy: str = "random"  # "point_forward" or "random" set in theme_config_factory.py
    HUD_text: str = "Body Count: "
    
    def make_appearance_policy(self, theme_id) -> "SpriteAppearancePolicy":
        return SpriteAppearancePolicy(
            theme_id=theme_id,
            sprite_type=self.sprite_type,
            sprite_keys=tuple(self.sprite_paths.keys()),
            rotation_policy=self.rotation_policy,
        )

class SpriteTheme(BodyTheme):
    def __init__(self, config: SpriteThemeConfig):
        self.geom_fn = GEOM_REGISTRY.get(config.sprite_type, None)
        if self.geom_fn is None:
            raise ValueError(f"Unknown sprite_type: {config.sprite_type}")
        self.config = config
        self.HUD_text = config.HUD_text

        #could base preprocessing on sprite geometry, and then rebuild self.sprites with updated prepocessed images
        self.sprites = self.build_sprite_assets(
            sprite_paths=config.sprite_paths,
        )

        self._extent_by_key: dict[str, tuple[float,float,float,float]] = {}
        for key, asset in self.sprites.items():
            geom = asset.geom
            # Map sprite pixel coordinates so that (0,0) is the intended rotation center.
            # If image array coords are (col=x, row=y), extent units here are "pixels".
            img = asset.image
            h, w = img.shape[0], img.shape[1]
            self._extent_by_key[key] = (-geom.cx_px, w - geom.cx_px, - geom.cy_px, h - geom.cy_px) 
        
        self.zorder = config.zorder
        self.interpolation = config.interpolation
        self.origin = config.origin

        # Bodies that were drawn this frame (used to 
        self._artists: dict[int, Any] = {}
        self._seen_this_frame: set[int] = set()
        self._last_pose: dict[int, tuple[float,float,float,float,str]] = {} # (x, y, angle, radius, sprite_key) for change detection

        # NEW: per-body cached transform + last sprite_key
        self._xf: dict[int, mtransforms.Affine2D] = {}
        self._last_key: dict[int, str] = {}

        # OPTIONAL: keep artists instead of removing (pooling)
        self._pooling = True
        self._max_cached = 5000  # tune for your worst case
    
    def build_sprite_assets(
        self,
        sprite_paths: dict[str, Path],
        *,
        max_render_px: dict[str, float] | None = None,
        oversample: float = 1.5,
        long_cap: float | None = None,  # optional guard
    ) -> dict[str, SpriteAsset]:
        assets: dict[str, SpriteAsset] = {}
        if max_render_px is None:
            max_render_px = {}
        for key, path in sprite_paths.items():
            img = preprocess_sprite(path, alpha_thresh=self.config.alpha_thresh)

            # 1) compute geom on the cropped original
            geom0 = self.geom_fn(img)
            # 2) choose downsample scale based on geom radius (best anchor)
            mrp = max_render_px.get(key, None)
            if mrp is not None:
                target_d = oversample * mrp               # target collider diameter in pixels
                current_d = 2.0 * float(geom0.r_px)
                scale = target_d / current_d

                # downsample only (never upscale)
                if scale < 1.0:
                    # optional: cap long side to avoid super tall/skinny assets staying huge
                    if long_cap is not None:
                        h, w = img.shape[:2]
                        scale2 = (long_cap * mrp) / max(h, w)
                        scale = min(scale, scale2)

                    new_w = max(1, int(round(img.shape[1] * scale)))
                    new_h = max(1, int(round(img.shape[0] * scale)))
                    img = np.asarray(
                        Image.fromarray(img).resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
                    )

            # 3) recompute geom after resize (important)
            geom = self.geom_fn(img)
            assets[key] = SpriteAsset(image=img, geom=geom)

        return assets

    def begin_frame(self) -> None:
        self._seen_this_frame.clear()

    def end_frame(self) -> None:
        if not self._pooling:
            # your original remove behavior
            for body_id in list(self._artists.keys()):
                if body_id not in self._seen_this_frame:
                    artist = self._artists.pop(body_id)
                    self._xf.pop(body_id, None)
                    self._last_key.pop(body_id, None)
                    self._last_pose.pop(body_id, None)
                    try:
                        artist.remove()
                    except Exception:
                        pass
            return

        # pooling: just hide missing
        for body_id, artist in self._artists.items():
            if body_id not in self._seen_this_frame:
                if artist.get_visible():
                    artist.set_visible(False)

        # prevent unbounded growth if IDs churn a lot
        if len(self._artists) > self._max_cached:
            # prune invisible ones first
            dead = [bid for bid, a in self._artists.items() if not a.get_visible()]
            for bid in dead[: len(self._artists) - self._max_cached]:
                a = self._artists.pop(bid)
                self._xf.pop(bid, None)
                self._last_key.pop(bid, None)
                self._last_pose.pop(bid, None)
                try:
                    a.remove()
                except Exception:
                    pass
    def clear_cache(self) -> None:
        """Forget cached artists (e.g., between recordings)."""
        for artist in list(self._artists.values()):
            try:
                artist.remove()
            except Exception:
                # If it's already detached/removed, ignore.
                pass
        self._artists.clear()
        self._xf.clear()
        self._last_key.clear()
        self._last_pose.clear()

    def prepare_for_recording(
        self, body_static: dict[int, BodyStaticSnapshot] | None = None,
        px_per_world: float | None = None,
    ) -> None:
        theme_id = self.config.theme_id
        sprite_keys = list(self.config.sprite_paths.keys())
        #find max radius for this theme_id
        max_render_px = {}
        if px_per_world is not None and body_static is not None:
            for key in sprite_keys:
                rmax = max((bs.collider.attrs["radius"] for bs in body_static.values() if bs.theme_id==theme_id and bs.sprite_key==key), default=None)
                if rmax is not None:
                    max_render_px[key] = 2.0 * px_per_world * rmax
 
        else:
            max_render_px = None
        self.sprites = self.build_sprite_assets(
            sprite_paths=self.config.sprite_paths,
            max_render_px=max_render_px
        )
        self.clear_cache()
        self._extent_by_key = {}
        for key, asset in self.sprites.items():
            geom = asset.geom
            # Map sprite pixel coordinates so that (0,0) is the intended rotation center.
            # If image array coords are (col=x, row=y), extent units here are "pixels".
            img = asset.image
            h, w = img.shape[0], img.shape[1]
            self._extent_by_key[key] = (-geom.cx_px, w - geom.cx_px, - geom.cy_px, h - geom.cy_px)  # note origin='upper' quir
        return 
    
    def draw_body(self, ax, body_id: int, state, static) -> None:
        pos = state.pos
        x, y = float(pos[0]), float(pos[1])
        angle = float(getattr(state, "angle", 0.0))
        R = float(static.collider.attrs["radius"])

        sprite_key = static.sprite_key  # rely on it existing
        asset = self.sprites[sprite_key]
        img, geom = asset.image, asset.geom
        # debug print shape of img
        
        artist = self._artists.get(body_id)    
        pose = (x, y, angle, R, sprite_key)
        if (
            self._last_pose.get(body_id) == pose
            and artist is not None
            and getattr(artist, "axes", None) is not None
            and artist.get_visible()
        ):
            self._seen_this_frame.add(body_id)
            return
        self._last_pose[body_id] = pose

        if artist is None or getattr(artist, "axes", None) is None:
            extent = self._extent_by_key[sprite_key]
            artist = ax.imshow(
                img,
                extent=extent,
                origin=self.origin,
                interpolation=self.interpolation,
                zorder=self.zorder,
            )
            self._artists[body_id] = artist
            self._last_key[body_id] = sprite_key

            xf = mtransforms.Affine2D()
            self._xf[body_id] = xf
            # set visible in case pooled id was reused
            if not artist.get_visible():
                artist.set_visible(True)
        else:
            # only update pixels if the sprite changed
            if self._last_key.get(body_id) != sprite_key:
                artist.set_data(img)
                artist.set_extent(self._extent_by_key[sprite_key])
                self._last_key[body_id] = sprite_key
                
            if not artist.get_visible():
                artist.set_visible(True)
            xf = self._xf[body_id]

        # mutate cached transform instead of allocating a new one
        xf.clear()
        s = R / geom.r_px

        # Compose: base recenter -> scale -> rotate -> translate to world position
        # (Order matters; Affine2D applies in the order you call them.)
        xf.scale(s, s)
        xf.rotate(angle)
        xf.translate(x, y)                           # move to world position

        artist.set_transform(xf + ax.transData)

        self._seen_this_frame.add(body_id)

