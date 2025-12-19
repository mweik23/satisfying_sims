from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt

def _smooth_gain(g: np.ndarray, sr: int, attack_s: float, release_s: float) -> np.ndarray:
    """One-pole smoothing with separate attack/release on the gain curve."""
    g = np.asarray(g, dtype=np.float32)
    out = np.empty_like(g)

    a_a = np.exp(-1.0 / max(1, int(sr * attack_s)))
    a_r = np.exp(-1.0 / max(1, int(sr * release_s)))

    y = 1.0
    for i, v in enumerate(g):
        a = a_a if v < y else a_r   # gain drops = attack; gain rises = release
        y = a * y + (1 - a) * v
        out[i] = y
    return out

def agc_rms(
    x: np.ndarray,
    *,
    sr: int,
    window_s: float = 0.050,     # 50 ms loudness window
    target_rms: float = 0.12,    # tune
    max_gain: float = 1.0,       # don't boost quiet parts (keeps beginning unchanged)
    max_reduction_db: float = 18.0,
    attack_s: float = 0.010,
    release_s: float = 0.250,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Automatic gain control based on short-time RMS.
    Keeps early quieter parts mostly unchanged (no upward gain by default),
    but reduces gain when the signal gets dense/loud.

    Returns (y, gain_curve).
    """
    x = np.asarray(x, dtype=np.float32)

    win = max(1, int(sr * window_s))
    # RMS via moving average of power
    power = x * x
    kernel = np.ones(win, dtype=np.float32) / win
    mean_power = np.convolve(power, kernel, mode="same")
    rms = np.sqrt(mean_power + eps)

    # Desired gain to hit target RMS
    raw_gain = target_rms / (rms + eps)

    # Don't boost the beginning (only reduce)
    raw_gain = np.minimum(raw_gain, max_gain)

    # Cap maximum reduction
    min_gain = 10 ** (-max_reduction_db / 20.0)
    raw_gain = np.clip(raw_gain, min_gain, 1.0)

    gain = _smooth_gain(raw_gain, sr, attack_s=attack_s, release_s=release_s)
    y = x * gain
    return y, gain

def soft_limit_tanh(x: np.ndarray, max_amplitude: float = 0.95) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return max_amplitude * np.tanh(x / max_amplitude)


def lowpass(x, sr, cutoff_hz=8000, order=3):
    b, a = butter(order, cutoff_hz/(sr/2), btype="low")
    return filtfilt(b, a, x).astype(np.float32)
