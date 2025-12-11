"""
Utilities for loading audio samples from disk into AudioSample objects.

Uses scipy.io.wavfile under the hood.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample  # simple resampler

from .engine import AudioSample


def load_wav_sample(
    path: str | Path,
    name: str | None = None,
    target_sr: int | None = None,
    mono: bool = True,
) -> AudioSample:
    """
    Load a WAV file and return an AudioSample.

    Parameters
    ----------
    path : str or Path
        WAV file path.
    name : str, optional
        Sample name; default is stem of path.
    target_sr : int, optional
        If set and file's sr != target_sr, resample.
    mono : bool
        If True, mixdown multi-channel to mono.

    Returns
    -------
    AudioSample
    """
    path = Path(path)
    sr, data = wavfile.read(path)

    # Convert to float32 in [-1, 1]
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)

    # Optionally mixdown to mono
    if mono and data.ndim == 2:
        data = data.mean(axis=1)

    # Optional resampling
    if target_sr is not None and target_sr != sr:
        factor = target_sr / sr
        new_n = max(1, int(round(data.shape[0] * factor)))
        if data.ndim == 1:
            data = resample(data, new_n)
        else:
            # multi-channel (if mono=False)
            channels = data.shape[1]
            out = np.empty((new_n, channels), dtype=np.float32)
            for ch in range(channels):
                out[:, ch] = resample(data[:, ch], new_n)
            data = out
        sr = target_sr

    return AudioSample(
        name=name or path.stem,
        data=data,
        sr=sr,
    )


def load_sample_bank(
    directory: str | Path,
    pattern: str = "*.wav",
    target_sr: int | None = None,
    mono: bool = True,
) -> Dict[str, AudioSample]:
    """
    Load a directory of WAV files into a dict[name, AudioSample].

    Parameters
    ----------
    directory : str or Path
        Directory to search.
    pattern : str
        Glob pattern for WAV files (default: "*.wav").
    target_sr : int, optional
        If set, resample all to this sample rate.
    mono : bool
        If True, mixdown to mono.

    Returns
    -------
    dict[str, AudioSample]
    """
    directory = Path(directory)
    samples: Dict[str, AudioSample] = {}

    for path in sorted(directory.glob(pattern)):
        sample = load_wav_sample(path, target_sr=target_sr, mono=mono)
        samples[sample.name] = sample

    if not samples:
        raise RuntimeError(f"No samples found in directory {directory} matching {pattern!r}")

    return samples
