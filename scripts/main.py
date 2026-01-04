# src/simproject/main.py

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, replace
import shutil

from satisfying_sims.core import run_simulation, SimConfig, make_world, SimStopRestartPolicy, SimAction
from satisfying_sims.render.collision_effects import build_collision_effect_router
from satisfying_sims.render.video import render_video, render_video_with_audio
from satisfying_sims.utils.random import seed_all
from satisfying_sims.utils.cli import build_parser
from satisfying_sims.render.renderer import MatplotlibRenderer, RendererConfig
from satisfying_sims.render.frame_export import select_and_export_frames
from satisfying_sims.utils.event_rejection import RejectConfig
from satisfying_sims.render.theme_config_factory import make_body_theme_cfgs
from satisfying_sims.utils.render_utils import find_box_geometry_from_png
from satisfying_sims.utils.preset_loader import load_preset

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = build_parser()
    args = parser.parse_args()
    preset_path_full = PROJECT_ROOT / args.preset_path
    loaded = load_preset(preset_path_full)
    preset_dir = preset_path_full.parent
    shutil.copytree(preset_dir, PROJECT_ROOT / args.outdir / args.exp_name / preset_dir.name, dirs_exist_ok=True)
    sprite_dir = PROJECT_ROOT / args.sprite_dir
    body_theme_cfgs = make_body_theme_cfgs(loaded.resolved['body_theme_registry'], sprite_dir=sprite_dir)
    background_png = str(PROJECT_ROOT / args.background_path / args.background_png_dir) if args.background_png_dir is not None else None
    collision_effects_dir = PROJECT_ROOT / args.collision_effects_dir if args.collision_effects_dir is not None else None
    # Determine world aspect ratio from background PNG if provided
    if background_png is None:
        world_aspect = None
        background_geom = None
    else:
        background_geom = find_box_geometry_from_png(f'{background_png}/processed.png')
        world_aspect = background_geom.world_aspect
        
    sim_config = SimConfig.from_args(
        args, 
        world_aspect=world_aspect,
        body_theme_cfgs=body_theme_cfgs,
        outer_boundary=loaded.resolved.get('outer_boundary', {}),
        inner_walls=loaded.resolved.get('inner_walls', []),
        rules=loaded.resolved.get('physics_rules', {}),
        init_groups=loaded.resolved.get('init_groups', {})
    )
    render_config = RendererConfig(
        background_color=args.background_color, 
        width_px=args.width_px, height_px=args.height_px, 
        world_color=args.world_color,
        boundary_color=args.boundary_color,
        show_debug=args.show_debug,
        theme_configs=body_theme_cfgs,
        background_png=background_png,
        fps=args.frame_rate,
        overlay_png=str(sprite_dir / args.overlay_png) if args.overlay_png is not None else None,
    )
    
    steps_per_frame = -(-args.physics_rate_request // args.frame_rate)
    physics_rate = steps_per_frame * args.frame_rate
    dt = 1 / physics_rate
    
    # base seed for reproducibility
    base_seed = args.seed

    policy = SimStopRestartPolicy(
        n_steps=int(args.duration * physics_rate) if args.duration is not None else None,
        max_bodies=args.max_bodies,
        tmin_1=args.tmin_1,
        num_bodies_1=args.num_bodies_1,
        delta_t21_min=args.delta_t21_min,
        num_bodies_2=args.num_bodies_2,
        tmax_1=args.tmax_1,
        max_frac_diff=args.max_frac_diff,
        max_frac_diff_thresh=args.max_frac_diff_thresh,
        multi_bodies_required=getattr(args, "multi_bodies_required", False),
    )

    max_restarts = getattr(args, "max_restarts", 25)

    for attempt in range(max_restarts + 1):
        run_seed = base_seed + attempt  # simple, deterministic
        seed_all(run_seed)

        world = make_world(sim_config)

        recording, decision = run_simulation(
            world,
            dt,
            log_interval=int(physics_rate),
            record_events=True,
            policy=policy,
        )

        if decision.action == SimAction.STOP:
            print(f"Finished (seed={run_seed}). Reason: {decision.reason}")
            break

        if decision.action == SimAction.RESTART:
            print(f"Restarting (seed={run_seed}). Reason: {decision.reason}")
            continue

    else:
        raise RuntimeError(
            f"Exceeded max_restarts={max_restarts} without meeting conditions "
            f"(base_seed={base_seed})."
        )


    recording.meta = {
        "sim_config": asdict(sim_config),
        "seed": run_seed,
        "engine_version": "0.1.0",
    }

    # 3. Output paths
    exp_dir = PROJECT_ROOT / args.outdir / args.exp_name
    exp_dir.mkdir(exist_ok=True, parents=True)

    video_path = exp_dir / "video.mp4"
    recording_path = exp_dir / "recording.pkl.xz"
    recording.save(recording_path)
    
    rules = loaded.resolved.get('collision_effects', [])
    collision_effects = build_collision_effect_router(
        rules,
        asset_dir=collision_effects_dir,
    )
    renderer = MatplotlibRenderer(render_config, 
                                  body_static=recording.body_static, 
                                  background_geom=background_geom,
                                  collision_effects=collision_effects)
    # 4. Render video, optionally with audio
    if args.audio_samples_dir is None:
        # Silent video only
        render_video(
            recording,
            output_path=video_path,
            fps=args.frame_rate,
            world_for_boundary=world,
            renderer=renderer,
            preview=args.preview,
        )
    else:
        reject_cfg = RejectConfig.from_args(args) if args.lam0 is not None else None
        # Video + audio
        render_video_with_audio(
            recording,
            output_path=video_path,
            samples_dir=PROJECT_ROOT / args.audio_samples_dir,
            fps=args.frame_rate,
            preview=args.preview,
            audio_sr=args.audio_sr,
            audio_tail=args.audio_tail,
            world_for_boundary=world,
            sample_map=args.sample_map,
            rules_cfg=loaded.resolved.get('sound_rules', None),
            renderer=renderer,
            reject_cfg=reject_cfg,
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
