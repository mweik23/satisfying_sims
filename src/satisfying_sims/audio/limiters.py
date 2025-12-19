from __future__ import annotations

import numpy as np

def soft_limit_tanh(x: np.ndarray, max_amplitude: float = 0.95) -> np.ndarray:
    """
    Smooth limiter using tanh. Output is guaranteed in [-max_amplitude, +max_amplitude].

    max_amplitude should be <= 1.0 if you later write float audio or normalize to int16.
    """
    if max_amplitude <= 0:
        raise ValueError("max_amplitude must be > 0")
    x = np.asarray(x, dtype=np.float32)
    return max_amplitude * np.tanh(x / max_amplitude)


def soft_limit_softsign(x: np.ndarray, max_amplitude: float = 0.95) -> np.ndarray:
    """
    Smooth limiter using a rational 'softsign' curve. Output is in [-max_amplitude, +max_amplitude].
    """
    if max_amplitude <= 0:
        raise ValueError("max_amplitude must be > 0")
    x = np.asarray(x, dtype=np.float32)
    return (max_amplitude * x) / (max_amplitude + np.abs(x))


def apply_makeup_gain_then_limit(
    x: np.ndarray,
    *,
    target_peak: float = 0.90,
    limiter_max: float = 0.95,
    limiter: str = "tanh",
    eps: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """
    Convenience helper:
      1) Scales x so its peak becomes `target_peak`
      2) Applies a soft limiter that caps at `limiter_max`

    Returns (y, applied_gain).

    This helps keep overall loudness in a reasonable band even as event density changes.
    """
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x)) + eps)
    gain = target_peak / peak
    xg = x * gain

    if limiter == "tanh":
        y = soft_limit_tanh(xg, max_amplitude=limiter_max)
    elif limiter == "softsign":
        y = soft_limit_softsign(xg, max_amplitude=limiter_max)
    else:
        raise ValueError(f"Unknown limiter: {limiter}")

    return y, float(gain)
