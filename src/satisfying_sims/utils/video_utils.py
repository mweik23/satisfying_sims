PREVIEW_FFMPEG_ARGS = [
    "-crf", "35",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-pix_fmt", "yuv420p",
]
FINAL_FFMPEG_ARGS = [
    "-crf", "18",          # 16–20 is a good “high quality” range
    "-preset", "slow",     # or "veryslow" if you really want, but slow is solid
    "-pix_fmt", "yuv420p",
]