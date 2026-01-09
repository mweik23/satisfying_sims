from __future__ import annotations

from pathlib import Path
from typing import Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image

def get_pix_per_world(ax) -> float:
    fig = ax.figure
    fig_w_px, fig_h_px = fig.get_size_inches() * fig.dpi

    bbox = ax.get_position()
    ax_w_px = fig_w_px * bbox.width
    ax_h_px = fig_h_px * bbox.height

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    world_w = abs(x1 - x0)
    world_h = abs(y1 - y0)

    return min(ax_w_px / world_w, ax_h_px / world_h)

def fig_inches_from_pixels(width_px: int | None = None, 
                           height_px: int | None = None, 
                           dpi: int = 100, 
                           figsize_default: tuple[float, float] = (6.0, 6.0)) -> tuple[float, float]:
    if width_px is not None and height_px is not None:
        return (width_px / dpi, height_px / dpi)
    else:
        return figsize_default

def compute_axes_rect(fig, pad: float, world_aspect: float) -> list[float]:
        fig_aspect = fig.get_figwidth() / fig.get_figheight()
        return [
            pad,
            (1 - (1 - 2*pad) / (world_aspect / fig_aspect)) / 2,
            1 - 2*pad,
            (1 - 2*pad) / (world_aspect / fig_aspect),
        ]

# ----------------------------
# Public API
# ----------------------------

@dataclass(frozen=True)
class BoxGeometry:
    """
    Inner box bounds in image pixel coordinates.

    Coordinates:
      - x increases to the right
      - y increases downward
      - (0, 0) is the top-left pixel of the image
      - x1/y1 are inclusive

    img_w/img_h are the full image dimensions in pixels.
    """
    x0: int
    y0: int
    x1: int  # inclusive
    y1: int  # inclusive
    img_w: int
    img_h: int

    @property
    def inner_w_px(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def inner_h_px(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def world_aspect(self) -> float:
        # width / height
        return self.inner_w_px / self.inner_h_px

    def axes_rect(self) -> list[float]:
        """
        Convert inner pixel bounds -> Matplotlib axes rect [left, bottom, width, height]
        in *figure normalized coordinates*.

        Assumes the PNG maps to the full figure area (pixel-aligned is ideal).
        """
        left = self.x0 / self.img_w
        right = (self.x1 + 1) / self.img_w

        top = self.y0 / self.img_h
        bottom_img = (self.y1 + 1) / self.img_h  # measured from top

        # flip Y axis for Matplotlib rects (origin bottom-left)
        bottom = 1.0 - bottom_img

        return [float(left), float(bottom), float(right - left), float(bottom_img - top)]


def find_box_geometry_from_png(
    png_path: str | Path,
    *,
    bg_rgb: Tuple[int, int, int] | None = None,
    bg_tol: float = 10.0,
    border_sample_px: int = 8,
    alpha_min: float = 0.1,
    # Inner-edge detection knobs
    interior_bg_frac: float = 0.995,
    min_outline_frac: float = 0.01,
    edge_clearance_px: int = 0,
) -> BoxGeometry:
    """
    Detect the *inner* empty rectangle of an outlined box in a PNG with a mostly-uniform background.

    Returns BoxGeometry so you can use:
      - geom.world_aspect  (early, for simulation setup)
      - geom.axes_rect()   (later, for rendering)

    Key assumptions:
      - background is mostly uniform
      - box is axis-aligned
      - outline pixels are non-background
      - interior is background-colored (or close enough under bg_tol)

    Tuning:
      - bg_tol: increase if antialiasing/noise makes bg differ slightly
      - interior_bg_frac: how "pure background" a row/col must be to count as interior
      - min_outline_frac: how much non-bg in a row/col counts as "outline-ish"
      - edge_clearance_px: push the detected inner edge further inward
    """
    png_path = Path(png_path)

    img = Image.open(png_path).convert("RGBA")
    arr = np.asarray(img)  # (H, W, 4) uint8
    H, W, _ = arr.shape

    rgb = arr[:, :, :3].astype(np.float32)
    alpha = (arr[:, :, 3].astype(np.float32) / 255.0)

    bg = _estimate_background_rgb(arr[:, :, :3], border_sample_px, bg_rgb)

    # Outline mask: pixels that differ from background and are not very transparent
    dist = np.linalg.norm(rgb - bg[None, None, :], axis=2)
    non_bg = (dist > bg_tol) & (alpha > alpha_min)

    ys, xs = np.nonzero(non_bg)
    if xs.size == 0:
        raise ValueError(
            "No non-background pixels detected. "
            "Try increasing bg_tol or providing bg_rgb explicitly."
        )

    # Rough bbox around all non-bg pixels (outline + any decorations)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    # Crop to bbox for density scans
    crop = non_bg[y0 : y1 + 1, x0 : x1 + 1]
    h, w = crop.shape

    # Fraction of non-bg pixels in each column/row
    col_frac = crop.mean(axis=0)
    row_frac = crop.mean(axis=1)

    # Convert interior_bg_frac into an equivalent threshold on non-bg fraction
    # interior means: "almost all background" -> non-bg fraction <= (1 - interior_bg_frac)
    max_nonbg_for_interior = 1.0 - interior_bg_frac

    ix0 = _find_inner_edge_1d(
        frac=col_frac,
        side="left",
        min_outline_frac=min_outline_frac,
        max_nonbg_for_interior=max_nonbg_for_interior,
        edge_clearance_px=edge_clearance_px,
    )
    ix1 = _find_inner_edge_1d(
        frac=col_frac,
        side="right",
        min_outline_frac=min_outline_frac,
        max_nonbg_for_interior=max_nonbg_for_interior,
        edge_clearance_px=edge_clearance_px,
    )
    iy0 = _find_inner_edge_1d(
        frac=row_frac,
        side="top",
        min_outline_frac=min_outline_frac,
        max_nonbg_for_interior=max_nonbg_for_interior,
        edge_clearance_px=edge_clearance_px,
    )
    iy1 = _find_inner_edge_1d(
        frac=row_frac,
        side="bottom",
        min_outline_frac=min_outline_frac,
        max_nonbg_for_interior=max_nonbg_for_interior,
        edge_clearance_px=edge_clearance_px,
    )

    if ix1 <= ix0 or iy1 <= iy0:
        raise ValueError(
            "Inner bbox collapsed. "
            "Reduce edge_clearance_px or loosen thresholds (interior_bg_frac/min_outline_frac)."
        )

    # Map inner bbox back into full-image pixel coordinates
    inner_x0 = x0 + ix0
    inner_x1 = x0 + ix1
    inner_y0 = y0 + iy0
    inner_y1 = y0 + iy1

    return BoxGeometry(
        x0=inner_x0,
        y0=inner_y0,
        x1=inner_x1,
        y1=inner_y1,
        img_w=W,
        img_h=H,
    )


def set_figure_background_png(fig, png_path: str | Path, *, zorder: int = -10) -> None:
    """
    Draw a PNG as a true figure background using fig.figimage (pixel coords).
    Best when your figure size/DPI matches the PNG pixels for exact alignment.
    """
    img = Image.open(png_path).convert("RGBA")
    fig.figimage(np.asarray(img), xo=0, yo=0, zorder=zorder)


# ----------------------------
# Internals
# ----------------------------

def _estimate_background_rgb(
    arr_rgb: np.ndarray,
    border_sample_px: int,
    bg_rgb: Tuple[int, int, int] | None,
) -> np.ndarray:
    if bg_rgb is not None:
        return np.array(bg_rgb, dtype=np.float32)

    samples = _sample_border_pixels(arr_rgb, border_sample_px)
    return _mode_color(samples)


def _sample_border_pixels(arr_rgb: np.ndarray, border: int) -> np.ndarray:
    H, W, _ = arr_rgb.shape
    b = max(1, min(border, H // 2, W // 2))
    top = arr_rgb[:b, :, :]
    bottom = arr_rgb[H - b :, :, :]
    left = arr_rgb[:, :b, :]
    right = arr_rgb[:, W - b :, :]
    return np.concatenate(
        [top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)],
        axis=0,
    )


def _mode_color(rgb: np.ndarray) -> np.ndarray:
    """
    rgb: (N, 3) uint8 -> most common RGB triplet as float32[3]
    """
    rgb = np.ascontiguousarray(rgb)
    packed = (
        (rgb[:, 0].astype(np.uint32) << 16)
        | (rgb[:, 1].astype(np.uint32) << 8)
        | (rgb[:, 2].astype(np.uint32))
    )
    vals, counts = np.unique(packed, return_counts=True)
    v = vals[np.argmax(counts)]
    return np.array([(v >> 16) & 255, (v >> 8) & 255, v & 255], dtype=np.float32)


def _smooth_1d(x: np.ndarray, win: int) -> np.ndarray:
    win = int(max(1, win))
    if win == 1:
        return x
    k = np.ones(win, dtype=float) / win
    # pad by edge values to avoid shifting endpoints too much
    xp = np.pad(x, (win // 2, win - 1 - win // 2), mode="edge")
    return np.convolve(xp, k, mode="valid")


def _find_inner_edge_1d(
    *,
    frac: np.ndarray,
    side: str,
    min_outline_frac: float,
    max_nonbg_for_interior: float,
    edge_clearance_px: int,
    # NEW knobs (safe defaults)
    smooth_win: int = 7,
    outline_hi: float | None = None,
    interior_lo: float | None = None,
    min_gap_px: int = 1,
) -> int:
    """
    More robust: find a transition from high non-bg density (outline) to low density (interior)
    with hysteresis thresholds.

    - outline_hi: threshold to consider we're "in outline"
    - interior_lo: threshold to consider we've entered "interior"
    If not provided, they are derived from min_outline_frac / max_nonbg_for_interior,
    but we also clamp to sensible values.
    """
    if side not in {"left", "right", "top", "bottom"}:
        raise ValueError(f"Invalid side={side}")

    n = int(frac.shape[0])
    f = _smooth_1d(frac.astype(float), smooth_win)

    # Derive hysteresis thresholds if not provided
    # outline_hi should be comfortably above interior_lo
    if interior_lo is None:
        interior_lo = float(max_nonbg_for_interior)
    if outline_hi is None:
        outline_hi = float(max(min_outline_frac, interior_lo * 3.0, interior_lo + 0.01))

    # If the image is noisy, interior_lo might be too tiny; adapt using quantiles.
    # This helps when interior isn't "pure background".
    q10 = float(np.quantile(f, 0.10))
    q50 = float(np.quantile(f, 0.50))
    # Use the lower tail as "interior-ish" if it’s higher than the strict threshold
    interior_lo = max(interior_lo, q10 * 1.2)
    # Use something above median as outline-ish, but not insane
    outline_hi = max(outline_hi, min(0.95, q50 * 1.2))

    def scan_forward() -> int:
        in_outline = False
        outline_start = None

        for i in range(n):
            if not in_outline and f[i] >= outline_hi:
                in_outline = True
                outline_start = i
                continue

            if in_outline and f[i] <= interior_lo:
                # require a small gap from where outline started so we don't trigger immediately
                if outline_start is not None and (i - outline_start) >= min_gap_px:
                    return min(n - 1, i + edge_clearance_px)

        raise ValueError(
            f"Could not find interior edge from {side}. "
            f"(outline_hi={outline_hi:.4f}, interior_lo={interior_lo:.4f}, "
            f"q10={q10:.4f}, q50={q50:.4f})"
        )

    def scan_backward() -> int:
        in_outline = False
        outline_start = None

        for j in range(n - 1, -1, -1):
            if not in_outline and f[j] >= outline_hi:
                in_outline = True
                outline_start = j
                continue

            if in_outline and f[j] <= interior_lo:
                if outline_start is not None and (outline_start - j) >= min_gap_px:
                    return max(0, j - edge_clearance_px)

        raise ValueError(
            f"Could not find interior edge from {side}. "
            f"(outline_hi={outline_hi:.4f}, interior_lo={interior_lo:.4f}, "
            f"q10={q10:.4f}, q50={q50:.4f})"
        )

    if side in {"left", "top"}:
        return scan_forward()
    else:
        return scan_backward()
