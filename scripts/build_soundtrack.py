#!/usr/bin/env python

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from satisfying_sims.core.recording import SimulationRecording
from satisfying_sims.audio.build_soundtrack import (
    build_and_save_soundtrack_for_recording,
    mux_audio_into_video_ffmpeg,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build soundtrack from a recording.")
    parser.add_argument("recording", type=str, help="Path to recording pickle (possibly .xz).")
    parser.add_argument(
        "--samples-dir",
        type=str,
        required=True,
        help="Directory containing WAV samples.",
    )
    parser.add_argument(
        "--output-wav",
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
        help="Existing silent video to attach the audio to (optional).",
    )
    parser.add_argument(
        "--video-out",
        type=str,
        default="video_with_audio.mp4",
        help="Output video path if --video is given.",
    )
    args = parser.parse_args()

    rec_path = Path(args.recording)
    # If you're always using .xz you might want SimulationRecording.load instead:
    if rec_path.suffix == ".xz":
        recording = SimulationRecording.load(rec_path)
    else:
        with rec_path.open("rb") as f:
            recording: SimulationRecording = pickle.load(f)

    wav_path = build_and_save_soundtrack_for_recording(
        recording=recording,
        samples_dir=args.samples_dir,
        wav_path=args.output_wav,
        sr=args.sr,
        tail=args.tail,
    )
    print(f"Wrote soundtrack to {wav_path!r}")

    if args.video is not None:
        video_out = mux_audio_into_video_ffmpeg(
            video_in=args.video,
            audio_in=wav_path,
            video_out=args.video_out,
        )
        print(f"Wrote video with audio to {video_out!r}")


if __name__ == "__main__":
    main()
