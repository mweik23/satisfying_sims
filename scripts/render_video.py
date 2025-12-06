# src/simproject/main.py

from __future__ import annotations

from pathlib import Path

from satisfying_sims.core import World, SpawnOnCollision, SimulationRecording, run_simulation

from satisfying_sims.render.video import render_video


def main():
    # 1. Build a world & rules from a preset
    world = make_default_world()  # or from presets/bouncing_gas.py
    rules = [SpawnOnCollision(...)]  # etc

    # 2. Run simulation and record
    n_steps = 600
    dt = 1 / 120  # physics at 120 Hz
    recording = run_simulation(world, rules, n_steps, dt)

    # 3. Render to video at 60 fps (slo-mo or real-time as you like)
    out_dir = Path("renders")
    out_dir.mkdir(exist_ok=True)
    output_path = out_dir / "bouncing_gas.mp4"

    render_video(
        recording,
        output_path=output_path,
        fps=60,
        world_for_boundary=world,
    )


if __name__ == "__main__":
    main()
