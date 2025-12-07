# src/satisfying_sims/utils/randomness.py

from __future__ import annotations
import numpy as np

# The global RNG for the entire simulation framework
_global_rng: np.random.Generator | None = None


def seed_all(seed: int | None) -> None:
    """
    Set the global RNG using the given seed.
    Allows full reproducibility for physics, presets, rendering randomness,
    audio event jitter, etc.
    """
    global _global_rng
    _global_rng = np.random.default_rng(seed)


def rng() -> np.random.Generator:
    """
    Return the global RNG. If not yet seeded, auto-seed with a random entropy source.
    """
    global _global_rng
    if _global_rng is None:
        _global_rng = np.random.default_rng()
    return _global_rng
