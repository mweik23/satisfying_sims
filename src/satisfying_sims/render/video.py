# src/simproject/render/video.py

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import shutil
import subprocess
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from .renderer import MatplotlibRenderer, RendererConfig
from satisfying_sims.utils.video_utils import PREVIEW_FFMPEG_ARGS, FINAL_FFMPEG_ARGS


if TYPE_CHECKING:
    from satisfying_sims.core import SimulationRecording, World
    from satisfying_sims.audio.event_rejection import RejectConfig

from satisfying_sims.audio.build_soundtrack import (
    build_and_save_soundtrack,
    mux_audio_into_video_ffmpeg,
)
from satisfying_sims.core.recording import SimulationRecording, FrameSnapshot
from pathlib import Path

if shutil.which("ffmpeg") is None:
    raise RuntimeError(
        "ffmpeg not found. Install with: conda install -c conda-forge ffmpeg"
    )
from copy import copy

def select_frames_for_fps(recording: SimulationRecording, fps: int) -> list[FrameSnapshot]:
    target_dt = 1.0 / fps
    frames = recording.frames
    if not frames:
        return []

    # 1) Select frames (same as before)
    selected: list[FrameSnapshot] = []
    next_t = frames[0].t
    for f in frames:
        if f.t + 1e-9 >= next_t:
            # shallow copy so we can rewrite events without mutating original recording
            f2 = copy(f)
            f2.events = []  # type: ignore[attr-defined]
            selected.append(f2)
            next_t += target_dt

    if not selected:
        return []

    # 2) Gather all events with their times from the *full-rate* frames
    #    (adjust this depending on where events live in your data model)
    all_events = []
    for f in frames:
        for e in getattr(f, "events", []):
            all_events.append((f.t, e))

    # 3) Assign each event to the nearest selected frame
    j = 0  # index into selected
    for t_e, e in all_events:
        # advance j while the next selected frame is closer
        while j + 1 < len(selected):
            t0 = selected[j].t
            t1 = selected[j + 1].t
            if abs(t1 - t_e) <= abs(t0 - t_e):
                j += 1
            else:
                break
        selected[j].events.append(e)  # type: ignore[attr-defined]

    return selected


def render_video(
    recording: SimulationRecording,
    *,
    output_path: str | Path,
    fps: int = 60,
    renderer: MatplotlibRenderer | None = None,
    world_for_boundary: World | None = None,
    bitrate: int | None = None,
    preview: bool = False,
    log_interval: int = 1 #seconds
) -> None:
    """
    Render a SimulationRecording to an MP4 using Matplotlib + ffmpeg.

    Requirements:
        - ffmpeg installed and discoverable by Matplotlib.
    """
    if renderer is None:
        renderer = MatplotlibRenderer(RendererConfig(), body_static=recording.body_static)

    output_path = Path(output_path)
    writer = FFMpegWriter(
        fps=fps,
        metadata={"artist": "satisfying_sims"},
        bitrate=bitrate,
        extra_args=PREVIEW_FFMPEG_ARGS if preview else FINAL_FFMPEG_ARGS,
    )

    renderer._init_figure(world=world_for_boundary)

    frames_to_render = select_frames_for_fps(recording, fps)
    with writer.saving(renderer.fig, str(output_path), renderer.config.dpi):
        for idx, frame in enumerate(frames_to_render):
            renderer.render_snapshot(
                frame,
                body_static=recording.body_static,
                ax=renderer.ax,
                frame_idx = idx
            )
            if (idx+1) % (fps*log_interval) == 0:
                print(f"Rendered {(idx+1)/fps:.1f} seconds of video...")
            writer.grab_frame()
    plt.close(renderer.fig)
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
    bitrate: int | None = None,
    preview: bool = False,
    sample_names: dict[str, str] | None = None,
    renderer: MatplotlibRenderer | None = None,
    reject_cfg: Optional[RejectConfig] = None,
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
        preview=preview,
        renderer=renderer
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
        reject_cfg=reject_cfg,
    )

    # 3) Mux
    final_video = mux_audio_into_video_ffmpeg(
        video_in=silent_video_path,
        audio_in=wav_path,
        video_out=output_path,
    )
    return final_video
