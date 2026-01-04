from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Set

import matplotlib.transforms as mtransforms
import numpy as np

from satisfying_sims.render.sprites import (
    SpriteGeom,
    ornament_geom,
    discoball_geom,
    firework_rocket_geom,
    fireball_geom,
    sprite_extent_for_circle_center,
)
from satisfying_sims.core.recording import BodyStaticSnapshot
from .base import BodyTheme, BodyThemeConfig
from satisfying_sims.render.sprites import preprocess_sprite
from pathlib import Path
from satisfying_sims.core.appearances import AppearancePolicy, SpriteAppearancePolicy


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
        if config.sprite_type == "ornament":
            geom_fn = ornament_geom
        elif config.sprite_type == "disco_ball":
            geom_fn = discoball_geom
        elif config.sprite_type == "firework_rocket":
            geom_fn = firework_rocket_geom
        elif config.sprite_type == "fireball":
            geom_fn = fireball_geom
        else:
            raise ValueError(f"Unknown sprite_type: {config.sprite_type}")
        self.config = config
        self.HUD_text = config.HUD_text

        # Preprocess once
        self.preprocessed_sprites = {
            name: preprocess_sprite(path, alpha_thresh=config.alpha_thresh)
            for name, path in config.sprite_paths.items()
        }
        self.sprites = {
            name: SpriteAsset(
                image=img,
                geom=geom_fn(img),
            )
            for name, img in self.preprocessed_sprites.items()
        }

        self.zorder = config.zorder
        self.interpolation = config.interpolation
        self.origin = config.origin

        # Cache of matplotlib AxesImage artists by body id
        self._artists: Dict[int, Any] = {}

        # Bodies that were drawn this frame (used to 
        self._artists: dict[int, Any] = {}
        self._seen_this_frame: set[int] = set()

        # NEW: per-body cached transform + last sprite_key
        self._xf: dict[int, mtransforms.Affine2D] = {}
        self._last_key: dict[int, str] = {}

        # OPTIONAL: keep artists instead of removing (pooling)
        self._pooling = True
        self._max_cached = 5000  # tune for your worst case

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
                    try:
                        artist.remove()
                    except Exception:
                        pass
            return

        # pooling: just hide missing
        for body_id, artist in self._artists.items():
            if body_id not in self._seen_this_frame:
                artist.set_visible(False)

        # prevent unbounded growth if IDs churn a lot
        if len(self._artists) > self._max_cached:
            # prune invisible ones first
            dead = [bid for bid, a in self._artists.items() if not a.get_visible()]
            for bid in dead[: len(self._artists) - self._max_cached]:
                a = self._artists.pop(bid)
                self._xf.pop(bid, None)
                self._last_key.pop(bid, None)
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

    def prepare_for_recording(
        self, body_static: dict[int, BodyStaticSnapshot] | None = None
    ) -> None:
        # No-op for now; sprites already preprocessed in __init__
        pass
    
    def draw_body(self, ax, body_id: int, state, static) -> None:
        pos = state.pos
        x, y = float(pos[0]), float(pos[1])
        angle = float(getattr(state, "angle", 0.0))
        R = float(static.collider.attrs["radius"])

        sprite_key = static.sprite_key  # rely on it existing
        asset = self.sprites[sprite_key]
        img, geom = asset.image, asset.geom
        # debug print shape of img
        extent = sprite_extent_for_circle_center(x=x, y=y, R=R, img=img, geom=geom)

        artist = self._artists.get(body_id)

        if artist is None or getattr(artist, "axes", None) is None:
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
            artist.set_visible(True)
        else:
            # only update pixels if the sprite changed
            if self._last_key.get(body_id) != sprite_key:
                artist.set_data(img)
                self._last_key[body_id] = sprite_key

            artist.set_extent(extent)
            artist.set_visible(True)
            xf = self._xf[body_id]

        # mutate cached transform instead of allocating a new one
        xf.clear()
        xf.rotate_around(x, y, angle)
        artist.set_transform(xf + ax.transData)

        self._seen_this_frame.add(body_id)

'''
class SpriteTheme(BodyTheme):
    """
    Render bodies as PNG "sprites" (RGBA images) placed in world coordinates.

    Option 2 lifecycle: persist artists across frames and update them in-place.
      - begin_frame(): mark no bodies as "seen"
      - draw_body(): create if missing / detached, else update in-place
      - end_frame(): remove artists for bodies not seen this frame
    """

    def __init__(self, config: SpriteThemeConfig):
        if config.sprite_type == "ornament":
            geom_fn = ornament_geom
        else:
            raise ValueError(f"Unknown sprite_type: {config.sprite_type}")

        # Preprocess once
        self.preprocessed_sprites = {
            name: preprocess_sprite(path, alpha_thresh=config.alpha_thresh)
            for name, path in config.sprite_paths.items()
        }
        self.sprites = {
            name: SpriteAsset(
                image=img,
                geom=geom_fn(img),
            )
            for name, img in self.preprocessed_sprites.items()
        }

        self.zorder = config.zorder
        self.interpolation = config.interpolation
        self.origin = config.origin

        # Cache of matplotlib AxesImage artists by body id
        self._artists: Dict[int, Any] = {}

        # Bodies that were drawn this frame (used to remove disappeared bodies)
        self._seen_this_frame: Set[int] = set()

    def prepare_for_recording(
        self, body_static: dict[int, BodyStaticSnapshot] | None = None
    ) -> None:
        # No-op for now; sprites already preprocessed in __init__
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

    def begin_frame(self) -> None:
        """Called once per frame before any draw_body calls."""
        self._seen_this_frame.clear()

    def end_frame(self) -> None:
        """Called once per frame after all draw_body calls."""
        # Remove artists for bodies that were not drawn this frame (despawned/offscreen)
        for body_id in list(self._artists.keys()):
            if body_id not in self._seen_this_frame:
                artist = self._artists.pop(body_id)
                try:
                    artist.remove()
                except Exception:
                    pass

    def draw_body(self, ax, body_id: int, state, static) -> None:
        x, y = float(state.pos[0]), float(state.pos[1])
        angle = float(getattr(state, "angle", 0.0))

        # Circle collider radius in world units
        R = float(static.collider.attrs["radius"])

        sprite_key = getattr(static, "sprite_key", None)
        if sprite_key is None:
            raise KeyError(
                "SpriteTheme requires BodyStaticSnapshot.sprite_key (got None)."
            )
        try:
            asset = self.sprites[sprite_key]
        except KeyError as e:
            raise KeyError(
                f"Unknown sprite_key '{sprite_key}'. "
                f"Known keys: {list(self.sprites.keys())}"
            ) from e

        img = asset.image
        geom = asset.geom

        extent = sprite_extent_for_circle_center(x=x, y=y, R=R, img=img, geom=geom)

        artist = self._artists.get(body_id)

        # If missing or detached (e.g. axes cleared), recreate
        if artist is None or getattr(artist, "axes", None) is None:
            artist = ax.imshow(
                img,
                extent=extent,
                origin=self.origin,
                interpolation=self.interpolation,
                zorder=self.zorder,
            )
            self._artists[body_id] = artist
        else:
            # In-place updates are fast
            artist.set_data(img)
            artist.set_extent(extent)

            # Ensure it's visible (in case you ever hide instead of removing)
            if hasattr(artist, "set_visible"):
                artist.set_visible(True)

        # Rotate around the circle center in world coords (x, y)
        artist.set_transform(
            mtransforms.Affine2D().rotate_around(x, y, angle) + ax.transData
        )

        self._seen_this_frame.add(body_id)
'''