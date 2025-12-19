
python scripts/main.py --exp_name black_neon_square_final --seed 102 --frame_rate 60 --duration 44 \
    --outdir results --physics_rate_request 600 --rules '{"SpawnOnCollision": {"vel_kick": 10.0}}' \
    --boundary '{"type": "BoxBoundary", "params": {"width": 100.0, "height": 100.0}}' \
    --n_bodies 2 --sigma_v 15.0 --radius 2.5 \
    --audio-samples-dir assets/audio --audio-sr 44100 --audio-tail 0.5 \
    --sample_map '{"CollisionEvent": "ice_crack", "HitWallEvent": "bloop"}' \
    --body_theme 'IceCracksTheme' --background_color 'black' --world_color 'black' \
    --width_px 1080 --height_px 1920 --padding 0.05 --export_design_frames --n_bodies_thresh_frame 5 --lam0 10 \
    --crf 18

#101: first collision at 15 seconds, second collision after 45 seconds