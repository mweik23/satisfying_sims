# src/simproject/main.py

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict

from satisfying_sims.core import run_simulation

from satisfying_sims.render.video import render_video
from satisfying_sims.utils.random import seed_all
from satisfying_sims.presets.basic import make_world
from satisfying_sims.utils.cli import build_parser
from satisfying_sims.core import rules, SimConfig, boundary

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = build_parser()
    args = parser.parse_args()
    sim_config = SimConfig.from_args(args)
    seed_all(args.seed)
    # 1. Build a world & rules from a preset
    world = make_world(sim_config)  
    steps_per_frame = -(-args.physics_rate_request // args.frame_rate)  # ceil division
    physics_rate = steps_per_frame * args.frame_rate
    # 2. Run simulation and record
    n_steps = int(args.duration * physics_rate)  # physics at physics_rate Hz
    dt = 1 / physics_rate  # physics at physics_rate Hz
    recording = run_simulation(world, n_steps, dt, log_interval=int(physics_rate))
    print('Simulation completed. Rendering video...')
    recording.meta = {
        "sim_config": asdict(sim_config),   # if you’re using dataclasses
        "seed": args.seed,
        "engine_version": "0.1.0",
    }
    
    # 3. Render to video at 60 fps (slo-mo or real-time as you like)
    exp_dir = PROJECT_ROOT / args.outdir / args.exp_name
    exp_dir.mkdir(exist_ok=True, parents=True)
    output_path = exp_dir / "simulation_video.mp4"
    recording.save(exp_dir / "simulation_recording.pkl")
    render_video(
        recording,
        output_path=output_path,
        fps=args.frame_rate,
        world_for_boundary=world,
        bitrate=args.bitrate,
    )


if __name__ == "__main__":
    main()
