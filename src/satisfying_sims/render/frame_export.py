from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from satisfying_sims.utils.render_utils import fig_inches_from_pixels
from .renderer import MatplotlibRenderer, RendererConfig
from satisfying_sims.core import SimulationRecording
from satisfying_sims.core import World
from dataclasses import replace

def export_frame(
    recording: SimulationRecording,
    *,
    out_path: str | Path,
    frame_index: int | None = None,
    t: float | None = None,
    renderer: MatplotlibRenderer | None = None,
    world_for_boundary: World | None = None,
) -> Path:
    """
    Render a single frame from `recording` to a PNG (or whatever extension you provide),
    matching the video renderer's resolution + visuals.

    Provide either:
      - frame_index (index into recording.frames)
      - t (choose the frame closest in time)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if renderer is None:
        renderer = MatplotlibRenderer(RendererConfig(), body_static=recording.body_static)

    if (frame_index is None) == (t is None):
        raise ValueError("Provide exactly one of frame_index or t")

    frames = recording.frames
    if not frames:
        raise ValueError("Recording has no frames")

    if frame_index is not None:
        if frame_index < 0 or frame_index >= len(frames):
            raise IndexError(f"frame_index {frame_index} out of range (0..{len(frames)-1})")
        frame = frames[frame_index]
    else:
        # pick closest by time
        frame = min(frames, key=lambda f: abs(float(f.t) - float(t)))  # type: ignore[arg-type]

    config = renderer.config

    world_aspect = world_for_boundary.boundary.width / world_for_boundary.boundary.height if world_for_boundary is not None else 1.0
    renderer._init_figure(world_aspect=world_aspect)

    renderer.render_snapshot(
        frame,
        body_static=recording.body_static,
        ax=renderer.ax,
        world_for_boundary=world_for_boundary,
    )

    # IMPORTANT: keep export consistent with how frames look in the video.
    # If your renderer hides axes / sets padding internally, this will match.
    renderer.fig.savefig(out_path, dpi=config.dpi, bbox_inches=None, pad_inches=0)
    plt.close(renderer.fig)
    return out_path

def find_base_frame_index(recording: SimulationRecording, n_bodies_thresh: int) -> int:
    min_idx = 0
    last_frame_idx = len(recording.frames) - 1
    max_idx = last_frame_idx
    search = True
    while search:
        base_frame_index = (max_idx + min_idx) // 2 
        #check if there are greater than n bodies
        if base_frame_index >= last_frame_idx-1:
            break
        if len(recording.frames[base_frame_index].bodies) < n_bodies_thresh:
            min_idx = base_frame_index + 1
        else:
            search = False
        
    return base_frame_index

def select_and_export_frames(
    recording: SimulationRecording,
    *,
    exp_dir: Path,
    world: World,
    render_config_init: RendererConfig,
    n_bodies_thresh,
):
    #find a frame number where there are a minimum number of bodies
    base_frame_index = find_base_frame_index(recording, n_bodies_thresh)
    
    #transparent with black boundary
    transp_render_config_w_boundary = replace(
        render_config_init,
        world_color=None,          # transparent world interior
        background_color=None,     # transparent fig + axes
        boundary_color="black",    # visible boundary
    ) 

    transparent_with_boundary_renderer = MatplotlibRenderer(
        transp_render_config_w_boundary,
        body_static=recording.body_static,
    )

    export_frame(
        recording,
        out_path=exp_dir / "frame_transparent_with_boundary.png",
        frame_index=base_frame_index,
        renderer=transparent_with_boundary_renderer,
        world_for_boundary=world,
    )

    # --- B) Transparent world + background, NO boundary ---
    #transparent with black boundary
    transp_render_config_no_boundary = replace(
        render_config_init,
        world_color=None,          # transparent world interior
        background_color=None,     # transparent fig + axes
        boundary_color=None,    # visible boundary
    ) 
    transparent_no_boundary_renderer = MatplotlibRenderer(
        transp_render_config_no_boundary,
        body_static=recording.body_static,
    )

    export_frame(
        recording,
        out_path=exp_dir / "frame_transparent_no_boundary.png",
        frame_index=base_frame_index,
        renderer=transparent_no_boundary_renderer,
        world_for_boundary=world,
    )
