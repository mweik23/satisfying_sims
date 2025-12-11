"""
Audio engine for mixing short samples into a soundtrack.

Core ideas:
- AudioSample: loaded WAV/PCM data.
- SoundTrigger: "play sample S at time t with gain and pitch".
- AudioEngine: mixes triggers into a mono or stereo buffer and writes WAV.

All audio is float32 in [-1, 1] internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
from scipy.io import wavfile  # dependency: scipy


@dataclass
class AudioSample:
    """A short audio sample to be triggered by events."""
    name: str
    data: np.ndarray  # shape (n,) for mono or (n, channels)
    sr: int           # sample rate (Hz)


@dataclass
class SoundTrigger:
    """
    A request to play a sample at a specific time with gain and pitch.

    pitch_ratio: 1.0 = original speed/pitch, 2.0 = one octave up, etc.
    """
    t: float
    sample_name: str
    gain: float = 1.0
    pitch_ratio: float = 1.0


class AudioEngine:
    """
    Simple offline audio mixer.

    - Assumes all samples share the same sample rate and number of channels.
    - Supports basic pitch shifting via time-stretch resampling.
    """

    def __init__(self, samples: dict[str, AudioSample], sr: int | None = None):
        if not samples:
            raise ValueError("AudioEngine requires at least one sample.")

        # Determine reference sample rate and channels
        if sr is None:
            any_sample = next(iter(samples.values()))
            sr = any_sample.sr
        self.sr = sr

        # Validate / normalize samples
        self.samples: dict[str, AudioSample] = {}
        ref_channels = self._get_num_channels(next(iter(samples.values())).data)

        for name, s in samples.items():
            if s.sr != self.sr:
                raise ValueError(
                    f"Sample {name!r} has sr={s.sr}, expected {self.sr}. "
                    "Resample on load (audio.io) or pass a consistent target_sr."
                )
            if self._get_num_channels(s.data) != ref_channels:
                raise ValueError(
                    f"Sample {name!r} has different channel count than others."
                )
            self.samples[name] = s

        self.n_channels = ref_channels

    # ------------------ public API ------------------ #

    def mix(
        self,
        triggers: Iterable[SoundTrigger],
        duration: float,
        normalize: bool = True,
        tail: float = 0.0,
    ) -> np.ndarray:
        """
        Mix all triggers into an audio buffer.

        Parameters
        ----------
        triggers : iterable of SoundTrigger
        duration : float
            Base duration in seconds (e.g. length of recording).
        normalize : bool
            If True, scale down if peaks exceed 1.0.
        tail : float
            Extra seconds added after max(duration, last_trigger_time).

        Returns
        -------
        audio : np.ndarray
            Shape (n_samples,) for mono or (n_samples, channels) for stereo.
        """
        triggers = list(triggers)
        if not triggers:
            n_samples = int(np.ceil(duration * self.sr))
            return np.zeros(n_samples, dtype=np.float32)

        last_trigger_t = max(trig.t for trig in triggers)
        total_duration = max(duration, last_trigger_t) + tail
        n_samples = int(np.ceil(total_duration * self.sr))

        if self.n_channels == 1:
            audio = np.zeros(n_samples, dtype=np.float32)
        else:
            audio = np.zeros((n_samples, self.n_channels), dtype=np.float32)

        for trig in triggers:
            sample = self.samples.get(trig.sample_name)
            if sample is None:
                # silently ignore unknown sample names; or raise if you prefer
                continue

            data = sample.data.astype(np.float32, copy=True)
            data = self._apply_pitch_and_gain(data, trig.pitch_ratio, trig.gain)

            start_idx = int(round(trig.t * self.sr))
            if start_idx >= n_samples:
                continue

            end_idx = min(start_idx + data.shape[0], n_samples)
            data = data[: end_idx - start_idx]

            # Mix in-place
            audio[start_idx:end_idx] += data

        if normalize:
            max_abs = np.max(np.abs(audio))
            if max_abs > 1.0e-8 and max_abs > 1.0:
                audio /= max_abs

        return audio

    def write_wav(self, path: str, audio: np.ndarray) -> None:
        """Write a float32 audio buffer to WAV (PCM 16-bit)."""
        # Convert float32 [-1, 1] to int16 for WAV
        clipped = np.clip(audio, -1.0, 1.0)
        int16 = (clipped * 32767.0).astype(np.int16)
        wavfile.write(path, self.sr, int16)

    # ------------------ internal helpers ------------------ #

    @staticmethod
    def _get_num_channels(data: np.ndarray) -> int:
        if data.ndim == 1:
            return 1
        elif data.ndim == 2:
            return data.shape[1]
        else:
            raise ValueError("AudioSample.data must be 1D (mono) or 2D (multi-channel).")

    def _apply_pitch_and_gain(
        self, data: np.ndarray, pitch_ratio: float, gain: float
    ) -> np.ndarray:
        if not np.isclose(pitch_ratio, 1.0):
            data = self._resample_pitch(data, pitch_ratio)
        data *= gain
        return data

    def _resample_pitch(self, data: np.ndarray, pitch_ratio: float) -> np.ndarray:
        """
        Very simple pitch-shift via time-domain resampling.

        pitch_ratio > 1.0 : sample plays faster and at higher pitch.
        pitch_ratio < 1.0 : slower / lower pitch.
        """
        n = data.shape[0]
        new_n = max(1, int(round(n / pitch_ratio)))

        x_old = np.linspace(0.0, 1.0, n, endpoint=False)
        x_new = np.linspace(0.0, 1.0, new_n, endpoint=False)

        if data.ndim == 1:
            return np.interp(x_new, x_old, data)
        else:
            out = np.empty((new_n, data.shape[1]), dtype=data.dtype)
            for ch in range(data.shape[1]):
                out[:, ch] = np.interp(x_new, x_old, data[:, ch])
            return out
