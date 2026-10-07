"""
review_possessions.py
Converts detected possessions into real video timestamps (for both cameras)
so you can jump to those moments in the actual footage and confirm they
look like genuine possessions.

Usage:
    python review_possessions.py possessions.json [n_samples]

By default samples 10 possessions spread across the whole game. Pass a
number to sample more/fewer.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as pc

pc.pin_working_directory()
GAME_CONFIG = pc.load_game_config()
FPS = pc.get_fps(GAME_CONFIG)
CAMERA_LABELS = [cam['label'] for cam in pc.get_cameras(GAME_CONFIG)]


def to_timestamp(frame, fps=FPS):
    total_sec = frame / fps
    m, s = divmod(int(total_sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python review_possessions.py possessions.json [n_samples]")
        sys.exit(1)

    path = Path(sys.argv[1])
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    with open(path) as f:
        possessions = json.load(f)

    if not possessions:
        print("No possessions found in file.")
        sys.exit(0)

    # Spread the sample across the whole list rather than just the first N,
    # so you see possessions from all four quarters, not just Q1.
    if len(possessions) <= n_samples:
        sample = possessions
    else:
        step = len(possessions) / n_samples
        sample = [possessions[int(i * step)] for i in range(n_samples)]

    print(f"Sampling {len(sample)} of {len(possessions)} possessions:\n")
    for p in sample:
        q = p['quarter']
        print(f"Q{q}  {p['start_sec']}s -> {p['end_sec']}s  ({p['duration_sec']}s)")
        for label in CAMERA_LABELS:
            qframes = pc.get_quarter_frames(GAME_CONFIG, label)
            qstart, qend = qframes[q]
            start_frame = qstart + int(p['start_sec'] * FPS)
            end_frame = qstart + int(p['end_sec'] * FPS)
            print(f"    {label}: {to_timestamp(start_frame)} -> {to_timestamp(end_frame)}  "
                  f"(frames {start_frame}-{end_frame})")
        print()


if __name__ == "__main__":
    main()
