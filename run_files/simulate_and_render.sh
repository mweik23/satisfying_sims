python scripts/main.py --exp_name dr_pepper2 --seed 300 --frame_rate 60 \
    --outdir results --physics_rate_request 240 \
    --gravity -18.0 \
    --audio-samples-dir assets/audio --audio-sr 44100 --audio-tail 0.5 \
    --width_px 1080 --height_px 1920 \
    --background_color black --boundary_color white \
    --max_bodies 1000 --max_restarts 100 --duration 68 \
    --lam0 10 --lam_max 50 --adaptive_audio_setting attenuate