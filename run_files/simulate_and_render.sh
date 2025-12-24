collision_effects=$(cat <<'EOF'
{
"CollisionEvent": {
    "asset": "firework.npz",
    "size_world": 20,
    "cmap": "hsv"
  },
  "HitWallEvent": {
    "asset": "firework.npz",
    "size_world": 15
  }
}
EOF
)


python scripts/main.py --exp_name new_years_test --seed 4 --frame_rate 60 --duration 27 \
    --outdir results --physics_rate_request 600 --rules '{"SpawnOnCollision": {"vel_kick": 10.0}}' \
    --boundary '{"type": "BoxBoundary", "params": {"width": 100.0, "height": 100.0}}' \
    --n_bodies 2 --sigma_v 10.0 --radius 2.5 --gravity -15.0 \
    --audio-samples-dir assets/audio --audio-sr 44100 --audio-tail 0.5 \
    --sample_map '{"CollisionEvent": "sword_slash", "HitWallEvent": "bloop"}' \
    --body_theme 'sprite' --sprite_type 'disco_ball' \
    --width_px 1080 --height_px 1920 --padding 0.05 --n_bodies_thresh_frame 5 \
    --background_color black --boundary_color white \
    --collision_effects "$collision_effects" --lam0=10 --lam_max=200