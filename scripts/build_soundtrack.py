#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from satisfying_sims.utils.io import unique_path
from satisfying_sims.core.recording import SimulationRecording
from satisfying_sims.audio.build_soundtrack import (
    build_and_save_soundtrack,
    mux_audio_into_video_ffmpeg,
)
from satisfying_sims.utils.event_rejection import RejectConfig
'''
example usage:
python scripts/build_soundtrack.py --output_wav soundtrack.wav --exp_name my_experiment \
    --sample_map '{"CollisionEvent": "ice_crack", "HitWallEvent": "bloop"}'
'''

def main() -> None:
    parser = argparse.ArgumentParser(description="Build soundtrack from a recording.")
    parser.add_argument(
        "--recording_name", 
        type=str, 
        default="recording.pkl.xz",
        help="Path to recording pickle (possibly .xz)."
    )
    parser.add_argument(
        "--samples_dir",
        type=str,
        default="assets/audio",
        help="Directory containing WAV samples.",
    )
    parser.add_argument(
        "--output_wav",
        type=str,
        default="soundtrack.wav",
        help="Output WAV path for the soundtrack.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=44100,
        help="Target audio sample rate.",
    )
    parser.add_argument(
        "--tail",
        type=float,
        default=0.3,
        help="Extra seconds of tail after recording end.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Existing silent video to attach the audio to (optional).",
    )
    parser.add_argument(
        "--video_out",
        type=str,
        default="video_new.mp4",
        help="Output video path if --video is given.",
    )
    parser.add_argument(
        "--exp_name",
        type=str, default="",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--sample_map",
        type=json.loads,
        default=None,
        help=(
            "JSON dict mapping event types to sample names, "
            'e.g. \'{"CollisionEvent": "metal_clang", "BoundaryCollisionEvent": "wood_thud"}\''
        ),
    )
    parser.add_argument(
        "--lam0",
        type=float,
        default=None,
        help="Base rejection rate for events (None to disable rejection).",
    )
    parser.add_argument(
        "--lam_max",
        type=float,
        default=100.0,
        help="Maximum rejection rate for events.",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=2.0,
        help="Rejection rate growth parameter.",
    )
    args = parser.parse_args()
    project_root = Path(__file__).parent.parent
    results_path = project_root / args.results_dir
    exp_path = results_path / args.exp_name
    rec_path = exp_path / args.recording_name
    output_wav_path = unique_path(exp_path / args.output_wav)
    video_out_path = unique_path(exp_path / args.video_out)
    silent_video_path = exp_path / args.video if args.video is not None else None
    # If you're always using .xz you might want SimulationRecording.load instead:
    if rec_path.suffix == ".xz":
        recording = SimulationRecording.load(rec_path)
    else:
        with rec_path.open("rb") as f:
            recording: SimulationRecording = pickle.load(f)

    reject_config = RejectConfig.from_args(args) if args.lam0 is not None else None
    wav_path = build_and_save_soundtrack(
        recording=recording,
        samples_dir=project_root / args.samples_dir,
        wav_path=output_wav_path,
        sr=args.sr,
        tail=args.tail,
        sample_names=args.sample_map,
        reject_cfg=reject_config,  # You can customize RejectConfig here if desired
    )
    print(f"Wrote soundtrack to {wav_path!r}")

    if silent_video_path is not None:
        video_out = mux_audio_into_video_ffmpeg(
            video_in=silent_video_path,
            audio_in=wav_path,
            video_out=video_out_path,
        )
        print(f"Wrote video with audio to {video_out!r}")


if __name__ == "__main__":
    main()