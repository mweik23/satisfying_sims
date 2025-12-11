
python scripts/main.py --exp_name first_with_sound --seed 2 --frame_rate 60 --duration 23.0 \
    --outdir results --physics_rate_request 600 --rules '{"SpawnOnCollision": {"vel_kick": 15.0}}' \
    --boundary '{"type": "BoxBoundary", "params": {"width": 100.0, "height": 100.0}}' \
    --n_bodies 2 --body_color blue --sigma_v 20.0 --radius 3.0 --bitrate 2000 \
    --audio-samples-dir assets/audio --audio-sr 44100 --audio-tail 0.5 \
    --sample-map '{"CollisionEvent": "ice_crack", "HitWallEvent": "bloop"}'
