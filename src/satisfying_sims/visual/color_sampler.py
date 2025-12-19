# satisfying_sims/vis/color_sampler.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Literal

import numpy as np
from satisfying_sims.utils.random import rng

try:
    import matplotlib as mpl
except ImportError as e:  # pragma: no cover
    raise ImportError("ColorSampler requires matplotlib to be installed.") from e


Strategy = Literal["uniform", "avoid_extremes", "discrete"]


def _stable_u(key: Hashable, seed: int) -> float:
    """
    Deterministically map (seed, key) -> u in [0, 1).
    Uses a stable (non-Python-hash) transformation so it won't change across runs.
    """
    # Convert key to bytes in a stable way
    s = f"{seed}|{repr(key)}".encode("utf-8", errors="surrogatepass")
    # FNV-1a 64-bit-ish simple hash (stable, fast, no extra deps)
    h = np.uint64(1469598103934665603)
    fnv_prime = np.uint64(1099511628211)
    for b in s:
        h ^= np.uint64(b)
        h *= fnv_prime
    # Map to [0,1)
    return float(h) / float(np.uint64(2**64 - 1))


@dataclass(frozen=True)
class ColorSampler:
    """
    Sample colors from a Matplotlib colormap.

    Typical usage:
        sampler = ColorSampler("viridis", strategy="avoid_extremes", vmin=0.15, vmax=0.90)
        rgba = sampler.sample(rng)  # (r,g,b,a)

    Deterministic usage:
        rgba = sampler.sample_for_key(body_id, seed=12345)

    Notes:
      - Returned RGBA is a 4-tuple of floats in [0,1], directly usable by Matplotlib.
      - If you want alpha control, set `alpha` here (or override later in the renderer).
    """

    cmap: str | mpl.colors.Colormap = "viridis"
    strategy: Strategy = "avoid_extremes"

    # Range in colormap parameter space
    vmin: float = 0.05
    vmax: float = 0.95

    # For discrete palettes
    n_discrete: int = 16

    # Output alpha (overrides colormap alpha)
    alpha: float = 1.0

    def _get_cmap(self) -> mpl.colors.Colormap:
        if isinstance(self.cmap, str):
            return mpl.colormaps[self.cmap]
        return self.cmap

    def _sample_u(self) -> float:
        if self.strategy == "uniform":
            return float(rng("color").random())
        if self.strategy == "avoid_extremes":
            lo, hi = float(self.vmin), float(self.vmax)
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError(f"Invalid vmin/vmax: {lo=}, {hi=} (need 0<=vmin<vmax<=1)")
            return float(rng("color").uniform(lo, hi))
        if self.strategy == "discrete":
            k = int(self.n_discrete)
            if k <= 0:
                raise ValueError(f"n_discrete must be > 0, got {k}")
            # Evenly spaced values in [vmin, vmax]
            lo, hi = float(self.vmin), float(self.vmax)
            vals = np.linspace(lo, hi, k, dtype=float)
            return float(rng("color").choice(vals))
        raise ValueError(f"Unknown strategy: {self.strategy!r}")

    def sample(self) -> tuple[float, float, float, float]:
        """Random sample using the provided RNG."""
        u = self._sample_u()
        return self.color_at(u)

    def sample_many(
        self, n: int
    ) -> list[tuple[float, float, float, float]]:
        """Sample n colors."""
        n = int(n)
        if n < 0:
            raise ValueError("n must be >= 0")
        return [self.sample() for _ in range(n)]

    def sample_for_key(self, key: Hashable, seed: int) -> tuple[float, float, float, float]:
        """
        Deterministic color for a key (e.g., body_id), stable across runs.
        Uses (seed, key) -> u in [0,1), then maps u through the colormap.

        If strategy is 'avoid_extremes', we clamp u into [vmin, vmax).
        If strategy is 'discrete', we map u to one of the discrete bins.
        """
        u = _stable_u(key=key, seed=int(seed))

        if self.strategy == "avoid_extremes":
            lo, hi = float(self.vmin), float(self.vmax)
            u = lo + (hi - lo) * u  # rescale into [lo, hi)
        elif self.strategy == "discrete":
            k = int(self.n_discrete)
            if k <= 0:
                raise ValueError(f"n_discrete must be > 0, got {k}")
            lo, hi = float(self.vmin), float(self.vmax)
            vals = np.linspace(lo, hi, k, dtype=float)
            idx = min(int(u * k), k - 1)
            u = float(vals[idx])
        # 'uniform' uses u as-is

        return self.color_at(u)

    def color_at(self, u: float) -> tuple[float, float, float, float]:
        """Map u in [0,1] through the colormap and apply alpha."""
        if not (0.0 <= float(u) <= 1.0):
            raise ValueError(f"u must be in [0,1], got {u}")
        cmap = self._get_cmap()
        r, g, b, _a = cmap(float(u))
        a = float(self.alpha)
        a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)
        return (float(r), float(g), float(b), a)


def rgba_to_hex(rgba: Iterable[float], include_alpha: bool = False) -> str:
    """Convert (r,g,b[,a]) in [0,1] to '#RRGGBB' or '#RRGGBBAA'."""
    rgba = list(rgba)
    if len(rgba) not in (3, 4):
        raise ValueError("rgba must have length 3 or 4")
    r, g, b = rgba[:3]
    a = rgba[3] if len(rgba) == 4 else 1.0

    def to_byte(x: float) -> int:
        x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))
        return int(round(x * 255))

    rb, gb, bb = to_byte(r), to_byte(g), to_byte(b)
    if include_alpha:
        ab = to_byte(a)
        return f"#{rb:02x}{gb:02x}{bb:02x}{ab:02x}"
    return f"#{rb:02x}{gb:02x}{bb:02x}"
