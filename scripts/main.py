# src/simproject/main.py

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, replace

from satisfying_sims.core import run_simulation, SimConfig
from satisfying_sims.render.video import render_video, render_video_with_audio
from satisfying_sims.utils.random import seed_all
from satisfying_sims.presets.basic import make_world
from satisfying_sims.utils.cli import build_parser
from satisfying_sims.render.renderer import MatplotlibRenderer, RendererConfig
from satisfying_sims.render.frame_export import select_and_export_frames
from satisfying_sims.audio.event_rejection import RejectConfig

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = build_parser()
    args = parser.parse_args()
    sim_config = SimConfig.from_args(args)
    seed_all(args.seed)

    # 1. Build world & rules
    world = make_world(sim_config)
    steps_per_frame = -(-args.physics_rate_request // args.frame_rate)  # ceil division
    physics_rate = steps_per_frame * args.frame_rate

    # 2. Run simulation and record
    n_steps = int(args.duration * physics_rate)
    dt = 1 / physics_rate
    recording = run_simulation(world, n_steps, dt, log_interval=int(physics_rate))
    print("Simulation completed. Rendering video...")

    recording.meta = {
        "sim_config": asdict(sim_config),
        "seed": args.seed,
        "engine_version": "0.1.0",
    }

    # 3. Output paths
    exp_dir = PROJECT_ROOT / args.outdir / args.exp_name
    exp_dir.mkdir(exist_ok=True, parents=True)

    video_path = exp_dir / "video.mp4"
    recording_path = exp_dir / "recording.pkl.xz"
    recording.save(recording_path)
    render_config = RendererConfig(background_color=args.background_color, 
                                   width_px=args.width_px, height_px=args.height_px, 
                                   world_color=args.world_color,
                                   boundary_color=args.boundary_color,
                                   show_debug=args.show_debug)
    renderer = MatplotlibRenderer(render_config, body_static=recording.body_static)
    # 4. Render video, optionally with audio
    if args.audio_samples_dir is None:
        # Silent video only
        render_video(
            recording,
            output_path=video_path,
            fps=args.frame_rate,
            world_for_boundary=world,
            renderer=renderer,
            crf=args.crf
        )
    else:
        reject_cfg = RejectConfig.from_args(args) if args.lam0 is not None else None
        # Video + audio
        render_video_with_audio(
            recording,
            output_path=video_path,
            samples_dir=PROJECT_ROOT / args.audio_samples_dir,
            fps=args.frame_rate,
            crf=args.crf,
            audio_sr=args.audio_sr,
            audio_tail=args.audio_tail,
            world_for_boundary=world,
            sample_names=args.sample_map,
            renderer=renderer,
            reject_cfg=reject_cfg,  # You can customize RejectConfig here if desired
        )

    if args.export_design_frames:
        select_and_export_frames(
            recording=recording,
            exp_dir=exp_dir,
            world=world,
            render_config_init=render_config,
            n_bodies_thresh=args.n_bodies_thresh_frame,
        )

if __name__ == "__main__":
    main()
