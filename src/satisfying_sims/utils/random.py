# satisfying_sims/utils/rng.py

from __future__ import annotations

from typing import Dict, Hashable
import numpy as np

_master_seed: int | None = None
_rngs: Dict[str, np.random.Generator] = {}


def seed_all(seed: int | None) -> None:
    """
    Set the master seed for all RNG usage in the project.

    - If seed is None: RNGs will be entropy-seeded (non-reproducible).
    - Resets cached named RNGs.
    """
    global _master_seed, _rngs
    _master_seed = seed
    _rngs.clear()


def rng(name: str = "physics") -> np.random.Generator:
    """
    Return a named global RNG stream (order-dependent draws within that stream).
    Good for: physics, color sampling, audio jitter, presets, etc.
    """
    global _rngs
    if name not in _rngs:
        if _master_seed is None:
            _rngs[name] = np.random.default_rng()
        else:
            # SeedSequence makes it easy to derive independent streams.
            ss = np.random.SeedSequence([_master_seed, _stable_int(name)])
            _rngs[name] = np.random.default_rng(ss)
    return _rngs[name]


def rng_for_key(name: str, key: Hashable) -> np.random.Generator:
    """
    Return a deterministic RNG for (name, key), e.g. ("cracks", body_id).

    This is *keyed randomness*: independent of call order and other RNG usage.
    Ideal for: per-body crack generation, per-body textures, etc.
    """
    if _master_seed is None:
        # If you truly want non-reproducible keyed RNGs when unseeded, include entropy.
        # Note: this is deterministic per process call, not per key across runs.
        return np.random.default_rng(np.random.SeedSequence())

    ss = np.random.SeedSequence([_master_seed, _stable_int(name), _stable_int(key)])
    return np.random.default_rng(ss)


def _stable_int(x: Hashable) -> int:
    """
    Convert arbitrary key -> stable 32-bit-ish integer without relying on Python's hash().
    """
    s = repr(x).encode("utf-8", errors="surrogatepass")
    # Simple stable folding into 32 bits
    h = 2166136261
    for b in s:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h
