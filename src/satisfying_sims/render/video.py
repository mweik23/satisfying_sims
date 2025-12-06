# src/simproject/render/video.py

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from .renderer import MatplotlibRenderer, RendererConfig

if TYPE_CHECKING:
    from satisfying_sims.core import SimulationRecording, World
    
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
        bitrate=8000,
    )

    frames_to_render = select_frames_for_fps(recording, fps)
    with writer.saving(fig, str(output_path), config.dpi):
        for frame in frames_to_render:
            renderer.render_snapshot(
                frame,
                ax=ax,
                world_for_boundary=world_for_boundary,
            )
            writer.grab_frame()
