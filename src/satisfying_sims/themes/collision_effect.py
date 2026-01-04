from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import imageio.v3 as iio
from collections import Counter

from pathlib import Path

import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from satisfying_sims.visual.color_sampler import ColorSampler

# --------- helpers: video decode + chroma key (green screen) ----------


def _chroma_key_green_to_rgba(
    rgb: np.ndarray,
    *,
    bg_green_thresh: float = 0.55,
    green_dominance: float = 0.20,
    feather: float = 0.08,
    despill: float = 0.5,
    white: bool = False,
    white_mode: str = "alpha",  # "alpha" (Option B) or "sum" (Option A)
) -> np.ndarray:
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]

    mx = np.maximum(r, b)
    score = g - mx

    # background-likelihood mask m in [0,1]
    m = (score - green_dominance) / max(feather, 1e-6)
    m = np.clip(m, 0.0, 1.0)
    m *= (g > bg_green_thresh).astype(np.float32)

    # key alpha (foreground opacity)
    a_key = 1.0 - m

    # optional despill (still helpful even in white mode)
    if despill > 0:
        g2 = g - despill * m * (g - mx)
        g = np.clip(g2, 0.0, 1.0)

    if white:
        # brightness estimate (pick one)
        L = (r + g + b) / 3.0  # smooth
        # L = np.maximum.reduce([r, g, b])  # punchier alternative

        if white_mode == "sum":
            # Option A: conserve r+g+b by forcing grayscale
            gray = L
            r = g = b = gray
            a = a_key
        elif white_mode == "alpha":
            # Option B: pure white RGB, put brightness into alpha
            r = g = b = 1.0
            a = a_key * L
        else:
            raise ValueError("white_mode must be 'sum' or 'alpha'")
    else:
        a = a_key

    rgba = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
    rgba[..., 0] = (np.clip(r, 0, 1) * 255).astype(np.uint8)
    rgba[..., 1] = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    rgba[..., 2] = (np.clip(b, 0, 1) * 255).astype(np.uint8)
    rgba[..., 3] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return rgba



def _tight_crop_box_from_alpha(alpha: np.ndarray, *, alpha_thresh: int = 1) -> tuple[int, int, int, int]:
    # alpha: (T,H,W) uint8
    mask = alpha > int(alpha_thresh)
    any_yx = mask.any(axis=0)  # (H,W)
    if not any_yx.any():
        return (0, 1, 0, 1)
    ys = np.where(any_yx.any(axis=1))[0]
    xs = np.where(any_yx.any(axis=0))[0]
    return (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)


def load_crop_effect_frames(
    path: str | Path,
    *,
    npz_key: str = "frames",
    max_frames: Optional[int] = None,
    # chroma key controls
    chroma_key: bool = True,
    bg_green_thresh: float = 0.55,
    green_dominance: float = 0.20,
    feather: float = 0.08,
    # crop controls
    crop_to_alpha: bool = True,
    alpha_thresh: int = 1,
    white: bool = False,
    white_mode: str = "alpha",  # "alpha" (Option B) or "sum" (Option A)
) -> tuple[List[np.ndarray], tuple[int, int, int, int]]:
    """
    Load frames from .npz/.npy as (T,H,W,3 or 4) uint8, optionally chroma-key (if RGB),
    then crop to the tight bounding box containing any alpha>alpha_thresh across all frames.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suf = path.suffix.lower()
    if suf == ".npz":
        data = np.load(path)
        if npz_key not in data:
            raise KeyError(f"{path} missing key '{npz_key}'. Keys: {list(data.keys())}")
        arr = np.asarray(data[npz_key])
    elif suf == ".npy":
        arr = np.asarray(np.load(path))
    else:
        raise ValueError(f"Unsupported format '{suf}'. Expected .npz or .npy: {path}")

    if arr.ndim != 4:
        raise ValueError(f"Expected (T,H,W,C), got {arr.shape} from {path}")
    if arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8, got {arr.dtype} from {path}")

    if max_frames is not None:
        arr = arr[: int(max_frames)]

    c = arr.shape[-1]
    if c == 4:
        rgba = arr
    elif c == 3:
        if chroma_key:
            rgba = np.stack(
                [
                    _chroma_key_green_to_rgba(
                        arr[i],
                        bg_green_thresh=bg_green_thresh,
                        green_dominance=green_dominance,
                        feather=feather,
                        white=white,
                        white_mode=white_mode,
                    )
                    for i in range(arr.shape[0])
                ],
                axis=0,
            )
        else:
            alpha = np.full(arr.shape[:3] + (1,), 255, dtype=np.uint8)
            rgba = np.concatenate([arr, alpha], axis=-1)
    else:
        raise ValueError(f"Channel dim must be 3 or 4, got {c} from {path}")

    H, W = rgba.shape[1], rgba.shape[2]
    if crop_to_alpha:
        y0, y1, x0, x1 = _tight_crop_box_from_alpha(rgba[..., 3], alpha_thresh=alpha_thresh)
        rgba = rgba[:, y0:y1, x0:x1, :]
        crop_box = (y0, y1, x0, x1)
    else:
        crop_box = (0, H, 0, W)
    return [rgba[i] for i in range(rgba.shape[0])], crop_box



# ------------------- effect theme -------------------

@dataclass
class CollisionEffectConfig:
    effect_path: str
    zorder: int = 50
    interpolation: str = "bilinear"
    origin: str = "upper"
    white_only: bool = True
    white_mode: str = "alpha"
    event_type: str | None = None
    event_filter: Dict[str, Any] | None = None # e.g. {"same_type": False}

    # size in world units: set one of these
    size_world: float = 0.3          # width of effect in world units (default)
    size_scale_with_radius: float = 0.0  # if >0, use size = this * R

    # when spawning multiple effects same frame, jitter a bit to avoid perfect overlap
    jitter_world: float = 0.0

    # chroma key tuning
    chroma_key: bool = True
    # NEW
    color_sampler: ColorSampler | None = None
    tint_strength: float = 0.6          # 0..1
    use_sampler_alpha: bool = False     # if True, modulate effect alpha by sampled alpha
    event_details: dict[str, Any] = None  # optional extra details about the event type


@dataclass
class _EffectInstance:
    start_frame: int
    pos: Tuple[float, float]   # world coords
    size_world: float
    artist_id: int             # index into pooled artists
    tint_key: tuple[int,int,int,int] | None = None


class CollisionEffectTheme:
    """
    Draw a keyed-out MP4 effect at collision positions.

    Call pattern per frame:
      theme.begin_frame(frame_idx)
      theme.ingest_events(snapshot.events, snapshot, body_static)
      theme.draw(ax, frame_idx)
      theme.end_frame()

    - Supports CollisionEvent snapshots with payload['pos'].
    - Supports HitWallEvent snapshots by inferring contact point:
        contact ≈ body_center - norm_vec * R
      where norm_vec points into world from wall.
    """

    def __init__(self, config: CollisionEffectConfig):
        self.config = config
        self.frames_rgba, _ = load_crop_effect_frames(
            config.effect_path, 
            chroma_key=config.chroma_key,
            crop_to_alpha=True,
            white=config.white_only,
            white_mode=config.white_mode,
        )
        self.num_effect_frames = len(self.frames_rgba)

        # pooled artists for speed (AxesImage)
        self._artists: List[Any] = []
        self._xforms: List[mtransforms.Affine2D] = []
        self._artist_visible: List[bool] = []

        # active instances
        self._active: List[_EffectInstance] = []
        self._frame_idx: int = 0
        self._tinted_cache: dict[tuple[int,int,int,int], list[np.ndarray]] = {}
        self._free_artist_ids: list[int] = []  # <-- NEW
    def clear_cache(self) -> None:
        for a in self._artists:
            try:
                a.remove()
            except Exception:
                pass
        self._artists.clear()
        self._xforms.clear()
        self._artist_visible.clear()
        self._active.clear()
        self._tinted_cache.clear()

    def prepare_for_recording(self, body_static=None) -> None:
        pass

    def begin_frame(self, frame_idx: int) -> None:
        self._frame_idx = frame_idx
        for i, a in enumerate(self._artists):
            if a is not None and self._artist_visible[i]:
                a.set_visible(False)
                self._artist_visible[i] = False


    def end_frame(self) -> None:
        # hide all unused artists (we’ll re-enable those in draw())
        pass

    def _build_tinted_stack(self, key: tuple[int,int,int,int]) -> list[np.ndarray]:
        r8,g8,b8,a8 = key
        tint = np.array([r8, g8, b8], dtype=np.float32) / 255.0
        strength = float(self.config.tint_strength)
        use_alpha = bool(self.config.use_sampler_alpha)

        out_stack: list[np.ndarray] = []
        for base in self.frames_rgba:  # each base is uint8 RGBA
            rgba = base.copy()

            # RGB tint
            rgb = rgba[..., :3].astype(np.float32) / 255.0
            tinted = rgb * tint.reshape(1,1,3)
            rgb2 = (1.0 - strength) * rgb + strength * tinted
            rgba[..., :3] = np.clip(rgb2 * 255.0, 0, 255).astype(np.uint8)

            # optional alpha modulation
            if use_alpha:
                a = a8 / 255.0
                rgba[..., 3] = (rgba[..., 3].astype(np.float32) * a).astype(np.uint8)

            out_stack.append(rgba)

        return out_stack

    def _get_tinted_stack(self, rgba_float: tuple[float,float,float,float]) -> list[np.ndarray]:
        key = self._quantize_rgba(rgba_float)
        stack = self._tinted_cache.get(key)
        if stack is None:
            stack = self._build_tinted_stack(key)
            self._tinted_cache[key] = stack
        return stack
    def _quantize_rgba(self, rgba: tuple[float,float,float,float]) -> tuple[int,int,int,int]:
        r,g,b,a = rgba
        def q(x: float) -> int:
            x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))
            return int(round(x * 255))
        return (q(r), q(g), q(b), q(a))
    
    def _get_stack_for_key(self, key: tuple[int,int,int,int]) -> list[np.ndarray]:
        stack = self._tinted_cache.get(key)
        if stack is None:
            stack = self._build_tinted_stack(key)
            self._tinted_cache[key] = stack
        return stack

    # --------- spawn logic ---------

    def ingest_events(
        self,
        event_snaps: List[Any],  # list[EventSnapshot]
        snapshot: Any,           # FrameSnapshot (for body positions)
        body_static: Dict[int, Any],
    ) -> None:
        """
        Look at event snapshots and spawn effect instances for collisions.
        """
        if not event_snaps:
            return

        rng = None
        if self.config.jitter_world > 0:
            rng = np.random.default_rng(12345 + self._frame_idx)  #TODO use one of my rngs
            
        allowed = getattr(self.config, "event_type", None)

        for e in event_snaps:
            if allowed is not None and e.type != allowed:
                continue
            filter_mismatch = False
            for k, v in (self.config.event_filter or {}).items():
                if e.payload.get(k, None) != v:
                    filter_mismatch = True
                    break
            
            if filter_mismatch:
                continue
            
            et = e.type

            if et == "CollisionEvent":
                pos = e.payload.get("pos", None)
                if pos is None:
                    continue
                x, y = float(pos[0]), float(pos[1])
                R = min(
                    float(body_static.get(e.a_id, {}).collider.attrs.get("radius", 0.0)) if e.a_id is not None else 0.0,
                    float(body_static.get(e.b_id, {}).collider.attrs.get("radius", 0.0)) if e.b_id is not None else 0.0,
                )
                self._spawn_at(x, y, R=R, body_static=body_static, rng=rng)

            elif et == "HitWallEvent":
                # Infer contact point from body center and wall normal + radius
                a_id = e.a_id if e.a_id is not None else e.payload.get("body_id", None)
                if a_id is None:
                    continue

                state = snapshot.bodies.get(a_id, None)
                if state is None:
                    continue

                nv = e.payload.get("norm_vec", None)
                if nv is None:
                    continue
                n = np.asarray(nv, dtype=float)
                norm = float(np.linalg.norm(n))
                if norm <= 0:
                    continue
                n = n / norm  # unit vector into world

                static = body_static.get(a_id, None)
                if static is None or static.collider is None:
                    continue
                R = float(static.collider.attrs.get("radius", 0.0))
                if R <= 0:
                    continue

                cx, cy = float(state.pos[0]), float(state.pos[1])
                # contact point on body surface toward the wall (opposite inward normal)
                x, y = cx - n[0] * R, cy - n[1] * R
                self._spawn_at(x, y, R=R, body_static=body_static, rng=rng)

            # else: ignore other event types

    def _spawn_at(
        self,
        x: float,
        y: float,
        *,
        R: float | None = None,
        body_static: Dict[int, Any] | None = None,
        rng: Optional[np.random.Generator],
    ) -> None:
        # size selection
        size = self.config.size_world
        if self.config.size_scale_with_radius > 0 and R is not None:
            if R > 0:
                size = self.config.size_scale_with_radius * R

        if rng is not None and self.config.jitter_world > 0:
            j = self.config.jitter_world
            x += float(rng.uniform(-j, j))
            y += float(rng.uniform(-j, j))
        
        tint_key = None
        if self.config.color_sampler is not None:
            tint_rgba = self.config.color_sampler.sample()  # (r,g,b,a) floats
            tint_key = self._quantize_rgba(tint_rgba)
            # (optional) clamp cache growth: quantization already helps

        artist_id = self._alloc_artist()
        self._active.append(
            _EffectInstance(
                start_frame=self._frame_idx,
                pos=(x, y),
                size_world=size,
                artist_id=artist_id,
                tint_key=tint_key,
            )
        )

    def _alloc_artist(self) -> int:
        # reuse an id that was explicitly freed
        if self._free_artist_ids:
            return self._free_artist_ids.pop()

        # otherwise create a new slot
        self._artists.append(None)
        self._xforms.append(mtransforms.Affine2D())
        self._artist_visible.append(False)
        return len(self._artists) - 1

    # --------- drawing ---------

    def draw(self, ax: Axes, frame_idx: int) -> None:
        """
        Draw all active effects for this frame.
        """
        if not self._active:
            return

        still_active: List[_EffectInstance] = []

        for inst in self._active:
            local = frame_idx - inst.start_frame
            if local < 0 or local >= self.num_effect_frames:
                self._free_artist_ids.append(inst.artist_id)  # <-- NEW
                continue
            if inst.tint_key is None:
                rgba = self.frames_rgba[local]
                
            else:
                rgba = self._get_stack_for_key(inst.tint_key)[local]
            x, y = inst.pos
            s = inst.size_world
            extent = (x - s / 2, x + s / 2, y - s / 2, y + s / 2)

            i = inst.artist_id
            artist = self._artists[i]
            if artist is None or getattr(artist, "axes", None) is None:
                artist = ax.imshow(
                    rgba,
                    extent=extent,
                    origin=self.config.origin,
                    interpolation=self.config.interpolation,
                    zorder=self.config.zorder,
                )
                self._artists[i] = artist
            else:
                # effect is animated, so data changes each frame
                artist.set_data(rgba)
                artist.set_extent(extent)
            artist.set_visible(True)
            self._artist_visible[i] = True

            # (optional) you could rotate/scale via self._xforms[i] if desired

            still_active.append(inst)

        self._active = still_active
