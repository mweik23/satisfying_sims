# src/simproject/render/video.py

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import shutil
import subprocess

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from .renderer import MatplotlibRenderer, RendererConfig

if TYPE_CHECKING:
    from satisfying_sims.core import SimulationRecording, World

from satisfying_sims.audio.build_soundtrack import (
    build_and_save_soundtrack,
    mux_audio_into_video_ffmpeg,
)
from satisfying_sims.core.recording import SimulationRecording
from pathlib import Path

if shutil.which("ffmpeg") is None:
    raise RuntimeError(
        "ffmpeg not found. Install with: conda install -c conda-forge ffmpeg"
    )
def select_frames_for_fps(recording: SimulationRecording, fps: int) -> list[SimulationRecording]:
    target_dt = 1.0 / fps
    frames = recording.frames
    if not frames:
        return []

    selected = []
    next_t = frames[0].t

    for f in frames:
        if f.t + 1e-9 >= next_t:
            selected.append(f)
            next_t += target_dt

    return selected


def render_video(
    recording: SimulationRecording,
    *,
    output_path: str | Path,
    fps: int = 60,
    renderer: MatplotlibRenderer | None = None,
    world_for_boundary: World | None = None,
    bitrate: int = 8000,
) -> None:
    """
    Render a SimulationRecording to an MP4 using Matplotlib + ffmpeg.

    Requirements:
        - ffmpeg installed and discoverable by Matplotlib.
    """
    if renderer is None:
        renderer = MatplotlibRenderer(RendererConfig())

    output_path = Path(output_path)
    config = renderer.config

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    writer = FFMpegWriter(
        fps=fps,
        metadata={"artist": "satisfying_sims"},
        bitrate=bitrate, 
    )

    frames_to_render = select_frames_for_fps(recording, fps)
    with writer.saving(fig, str(output_path), config.dpi):
        for frame in frames_to_render:
            renderer.render_snapshot(
                frame,
                body_static=recording.body_static,
                ax=ax,
                world_for_boundary=world_for_boundary,
            )
            writer.grab_frame()
    plt.close(fig)
    return output_path

def render_video_with_audio(
    recording: SimulationRecording,
    *,
    output_path: str | Path,
    samples_dir: str | Path,
    fps: int = 60,
    audio_sr: int = 44100,
    audio_tail: float = 0.3,
    world_for_boundary: World | None = None,
    bitrate: int = 8000,
    sample_names: dict[str, str] | None = None,
) -> Path:
    """
    Convenience wrapper:
      1) Render silent MP4.
      2) Build soundtrack via shared helper.
      3) Mux audio into final MP4.
    """
    output_path = Path(output_path)

    # 1) Silent video
    silent_video_path = output_path.with_name(
        output_path.stem + "_silent" + output_path.suffix
    )
    
    render_video(
        recording,
        output_path=silent_video_path,
        fps=fps,
        world_for_boundary=world_for_boundary,
        bitrate=bitrate,
    )

    # 2) Build soundtrack
    wav_path = output_path.with_suffix(".wav")
    build_and_save_soundtrack(
        recording=recording,
        samples_dir=samples_dir,
        wav_path=wav_path,
        sr=audio_sr,
        tail=audio_tail,
        sample_names=sample_names,
    )

    # 3) Mux
    final_video = mux_audio_into_video_ffmpeg(
        video_in=silent_video_path,
        audio_in=wav_path,
        video_out=output_path,
    )
    return final_video
