# src/satisfying_sims/build_soundtrack.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Mapping, Sequence
import subprocess

import numpy as np

from satisfying_sims.core.recording import SimulationRecording, EventContext
from satisfying_sims.audio.io import load_sample_bank
from satisfying_sims.audio.engine import AudioEngine
from satisfying_sims.audio.mapping import (
    EventSoundMapper,
    EventSoundRule,
    gain_from_impulse,
    gain_constant,
    pitch_from_relative_speed,
    pitch_from_impulse,
    rules_from_config,
)
from satisfying_sims.utils.event_rejection import RejectConfig, make_keep_prob

# ---------- Core: snapshots -> audio buffer ---------- #

def build_soundtrack_from_snapshots(
    snapshots: Iterable[EventContext],
    mapper: EventSoundMapper,
    engine: AudioEngine,
    base_duration: Optional[float] = None,
    tail: float = 0.3,
    normalize: bool = True,
) -> np.ndarray:
    triggers = mapper.triggers_from_snapshots(snapshots)

    if base_duration is None:
        if snapshots:
            base_duration = max(float(s.ev.t) for s in snapshots)
        else:
            base_duration = 0.0

    audio = engine.mix(
        triggers=triggers,
        duration=base_duration,
        normalize=normalize,
        tail=tail,
    )
    return audio


# ---------- Default rules + full “build & save” helper ---------- #

def make_default_event_sound_rules(
    sample_map: Optional[Mapping[str, str]] = None,
) -> dict[str, EventSoundRule]:
    """
    Default mapping from EventSnapshot.type -> EventSoundRule.

    `sample_names` maps event type names to sample names (filestems).
    For example:
        {
            "CollisionEvent": "metal_clang",
            "HitWallEvent": "wood_thud",
        }
    Missing keys use built-in defaults.
    """
    # Built-in defaults
    default_sample_names: dict[str, str] = {
        "CollisionEvent": "ball_hit_ball",
        "HitWallEvent": "ball_hit_wall",
        # add more event types here as you introduce them
    }

    # Overlay user-provided sample names
    if sample_map:
        for event_type, sample_name in sample_map.items():
            default_sample_names[event_type] = sample_name

    # Build rules from final mapping
    rules: dict[str, EventSoundRule] = {}

    if "CollisionEvent" in default_sample_names:
        rules["CollisionEvent"] = EventSoundRule(
            event_type="CollisionEvent",
            sample_name=default_sample_names["CollisionEvent"],
            gain_fn=lambda s: gain_from_impulse(s, i0=50, sigma_i=25, max_factor_log=0.75, base=1.0),
            pitch_fn=lambda s: pitch_from_relative_speed(s, v0=50, sigma_v=25, max_factor_log=0.75, base=1.0)
        )

    if "HitWallEvent" in default_sample_names:
        rules['HitWallEvent'] = EventSoundRule(
            event_type="HitWallEvent",
            sample_name=default_sample_names["HitWallEvent"],
            gain_fn=lambda s: gain_from_impulse(s, i0=50, sigma_i=25, max_factor_log=0.75, base=1.0),
            pitch_fn=lambda s: pitch_from_impulse(s, i0=75, sigma_i=37.5, max_factor_log=0.3, base=1.0),
        )

    # If you add more event types, configure them here using default_sample_names[...]    

    return rules

def build_and_save_soundtrack(
    recording: SimulationRecording,
    samples_dir: str | Path,
    wav_path: str | Path,
    *,
    sr: int = 44100,
    tail: float = 0.3,
    rules: Optional[Sequence[EventSoundRule]] = None,
    rules_cfg: dict[str, dict] | None = None,   # your JSON-like list input
    sample_map: dict[str, str] | None = None,  # keep if you still want simple overrides for defaults
    reject_cfg: Optional[RejectConfig] = None,
) -> Path:
    samples_dir = Path(samples_dir)
    wav_path = Path(wav_path)

    event_context = list(recording.iter_event_context())
    t_end = recording.t_end
    if t_end is None:
        t_end = max((float(s.ev.t) for s in event_context), default=0.0)

    samples = load_sample_bank(samples_dir, target_sr=sr, mono=True)
    engine = AudioEngine(samples, sr=sr)

    keep_prob = make_keep_prob(reject_cfg) if reject_cfg is not None else None

    if rules is None:
        if rules_cfg is not None:
            # choose some defaults (or make them per-event-type like before)
            default_gain = lambda s: gain_from_impulse(s, i0=50, sigma_i=25, max_factor_log=0.75, base=1.0)
            default_pitch = lambda s: pitch_from_impulse(s, i0=50, sigma_i=25, max_factor_log=0.75, base=1.0)
            rules = rules_from_config(
                rules_cfg,
                default_gain_fn=default_gain,
                default_pitch_fn=default_pitch,
            )
        else:
            rules = make_default_event_sound_rules(sample_map=sample_map)  # update this to return list

    mapper = EventSoundMapper(rules, keep_prob=keep_prob)

    audio = build_soundtrack_from_snapshots(
        snapshots=event_context,
        mapper=mapper,
        engine=engine,
        base_duration=t_end,
        tail=tail,
        normalize=True,
    )
    engine.write_wav(str(wav_path), audio)
    return wav_path



# ---------- ffmpeg mux helper ---------- #

def mux_audio_into_video_ffmpeg(
    video_in: str | Path,
    audio_in: str | Path,
    video_out: str | Path,
    audio_codec: str = "aac",
) -> Path:
    """
    Attach an audio track to an existing video using ffmpeg.

    - Video stream is copied without re-encoding.
    - Audio is encoded as AAC by default.
    """
    video_in = Path(video_in)
    audio_in = Path(audio_in)
    video_out = Path(video_out)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_in),
        "-i", str(audio_in),
        "-c:v", "copy",
        "-c:a", audio_codec,
        "-shortest",
        str(video_out),
    ]
    subprocess.run(cmd, check=True)
    return video_out
