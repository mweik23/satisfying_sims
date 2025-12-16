
python scripts/main.py --exp_name radius_2 --seed 2 --frame_rate 60 --duration 41.5 \
    --outdir results --physics_rate_request 600 --rules '{"SpawnOnCollision": {"vel_kick": 30.0}}' \
    --boundary '{"type": "BoxBoundary", "params": {"width": 100.0, "height": 100.0}}' \
    --n_bodies 2 --sigma_v 20.0 --radius 2.0 --bitrate 1000 \
    --audio-samples-dir assets/audio --audio-sr 44100 --audio-tail 0.5 \
    --sample-map '{"CollisionEvent": "ice_crack", "HitWallEvent": "bloop"}' \
    --body_theme 'IceCracksTheme' --background_color 'black' --world_color 'turquoise'
