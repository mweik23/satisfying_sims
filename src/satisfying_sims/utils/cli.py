import argparse
import json

def build_parser():
    parser = argparse.ArgumentParser(description='Spacenet Training Script')
    parser.add_argument('--exp_name', type=str, default='', metavar='N',
                        help='experiment_name')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--frame_rate', type=int, default=60, metavar='N',
                        help='frame rate for rendering (default: 60)')
    parser.add_argument('--duration', type=float, default=None, metavar='N',
                        help='duration of the simulation in seconds (default: 10.0)')
    parser.add_argument('--outdir', type=str, default='results', metavar='N',
                        help='directory to save the rendered video (default: results)')
    parser.add_argument('--physics_rate_request', type=int, default=600, metavar='N',
                        help='requested physics simulation rate in Hz (default: 600)')
    parser.add_argument('--bitrate', type=int, default=8000, metavar='N',
                        help='bitrate for the rendered video in kbps (default: 8000)')
    parser.add_argument('--crf', type=str, default='23', metavar='N',
                        help='constant rate factor for video quality (default: 23)')
    parser.add_argument(
        "--gravity",
        type=float,
        default=0.0,
        help="Gravity strength in the downward (negative y) direction.",
    )
    parser.add_argument(
        "--audio-samples-dir",
        type=str,
        default=None,
        help="Directory of WAV samples for building soundtrack. If omitted, no audio is added.",
    )
    parser.add_argument(
        "--audio-sr",
        type=int,
        default=44100,
        help="Audio sample rate for soundtrack.",
    )
    parser.add_argument(
        "--audio-tail",
        type=float,
        default=0.3,
        help="Extra tail seconds after recording end in soundtrack.",
    )
    parser.add_argument(
        "--audio_effects_cfg",
        type=json.loads,
        default=None,
        help=(
            "JSON dict mapping event types to sample names, "
            'e.g. \'{"CollisionEvent": {"asset": "metal_clang"}, "BoundaryCollisionEvent": {"asset": "wood_thud"}}\''
        ),
    )
    parser.add_argument(
        "--sprite_dir",
        type=str,
        default='assets/sprites',
        help="directory containing sprite images for SpriteTheme"
    )
    parser.add_argument(
        "--background_color",
        type=str,
        default="white",
        help="background color for the rendered video"
    )
    parser.add_argument(
        "--background_path",
        type=str,
        default="assets/backgrounds",
        help="path to a background image PNG for the rendered video"
    )
    parser.add_argument(
        "--background_png_dir",
        type=str,
        default=None,
        help="directory with PNG image to use as the background for the rendered video"
    )
    parser.add_argument(
        "--world_color",
        type=str,
        default=None,
        help="background color for the rendered video"
    )
    parser.add_argument(
        "--width_px",
        type=int,
        default=None,
        help="width in pixels for the rendered video"
    )
    parser.add_argument(
        "--height_px",
        type=int,
        default=None,
        help="height in pixels for the rendered video"
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="padding fraction for the rendered video"
    )
    parser.add_argument(
        "--n_bodies_thresh_frame",
        type=int,
        default=5,
        help="find a frame with at least this many bodies for exporting special frames"
    )
    parser.add_argument(
        "--export_design_frames",
        action='store_true',
        help="whether to export special design frames"
    )
    parser.add_argument(
        "--boundary_color",
        type=str,
        default=None,
        help="color of the boundary in the simulation (default: turquoise)"
    )
    parser.add_argument(
        "--show_debug",
        action='store_true',
        help="whether to show debug information overlay in the rendered video"
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
    parser.add_argument(
        "--caption_text",
        type=str,
        default="",
        help="Caption text to overlay on the video.",
    )
    parser.add_argument(
        "--hud_text",
        type=str,
        default="",
        help="HUD text to overlay on the video.",
    )
    parser.add_argument(
        "--preview",
        action='store_true',
        help="whether to use preview settings for faster rendering"
    )
    parser.add_argument(
        "--collision_effects",
        type=json.loads,
        default=None,
        help=(
            "JSON dict mapping event types to collision effect configs, "
            'e.g. \'{"CollisionEvent": {"asset": "fx/firework.mp4", "size_world": 0.35, "cmap": "viridis"}, '
            '"HitWallEvent": {"asset": "fx/spark.mp4", "size_scale_with_radius": 3.0}}\''
        ),
    )
    parser.add_argument(
        "--collision-effects_dir",
        type=str,
        default="assets/effects",
        help="directory containing collision effect assets"
    )
    parser.add_argument(
        "--max_restarts",
        type=int,
        default=25,
        help="maximum number of simulation restarts to attempt"
    )
    parser.add_argument(
        "--max_bodies",
        type=int,
        default=None,
        help="maximum number of bodies before stopping the simulation"
    )
    parser.add_argument(
        "--tmin_1",
        type=float,
        default=0.0,
        help="minimum time to check for num_bodies_1 condition"
    )
    parser.add_argument(
        "--num_bodies_1",
        type=int,
        default=None,
        help="minimum number of bodies required at time tmin_1"
    )
    parser.add_argument(
        "--delta_t21_min",
        type=float,
        default=None,
        help="minimum time difference between tmin_1 and tmin_2"
    )
    parser.add_argument(
        "--num_bodies_2",
        type=int,
        default=None,
        help="minimum number of bodies required at time tmin_1 + delta_t21_min"
    )
    parser.add_argument(
        "--tmax_1",
        type=float,
        default=None,
        help="maximum time by which num_bodies_1 condition must be met"
    )
    parser.add_argument(
        "--sample_map",
        type=json.loads,
        default=None,
        help=(
            "JSON dict mapping event types to sample names for audio, "
            'e.g. \'{"CollisionEvent": "ice_crack", "HitWallEvent": "bloop"}\''
        ),
    )
    parser.add_argument(
        "--preset_path",
        type=str,
        default="presets/preset.yaml",
        help="path to preset configuration file"
    )
    parser.add_argument(
        "--max_frac_diff",
        type=float,
        default=None,
        help="maximum allowed fractional difference between body counts for restart condition"
    )
    parser.add_argument(
        "--max_frac_diff_thresh",
        type=int,
        default=None,
        help="minimum total body count for max_frac_diff condition to apply"
    )
    parser.add_argument(
        "--overlay_png",
        type=str,
        default=None,
        help="path to a PNG image to overlay on the simulation"
    )
    return parser

'''
usage: python render_video.py --exp_name my_experiment --seed 42 --frame_rate 60 --duration 15.0 \
    --outdir results --physics_rate_request 600 --rules '{"SpawnOnCollision": {}}' \
        --boundary '{"type": "BoxBoundary", "params": {"width": 100.0, "height": 100.0}}' \
            --n_bodies 20 --body_color blue --sigma_v 10.0 --radius 1.0 --bitrate 8000 \
                --audio-samples-dir assets/audio --audio-sr 44100 --audio-tail 0.5 \
                    --sample-map '{"CollisionEvent": "ice_crack", "HitWallEvent": "bloop"}' \
                        --body_theme "IceCracksTheme"
'''