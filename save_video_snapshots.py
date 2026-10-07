"""
save_video_snapshots.py
Saves a handful of real video frames from a specific quarter/time window to
disk as .jpg files, so you can look at them directly (or upload them back
for review) - no interactive playback needed, just quick visual confirmation
of what's actually happening on screen during a specific window.

Usage:
    python save_video_snapshots.py "SYR UNC MVC Side.mp4" --game SYR_UNC_20260213 \
        --quarter 2 --start 1514.1 --end 1620.8 --n 8 --pad 15 --out snapshots_q2_unc_gap
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import pipeline_common as pc


def pin_to_game_folder(game_name):
    base_dir = Path(__file__).resolve().parent.parent
    game_dir = base_dir / 'games' / game_name
    if not game_dir.exists():
        print(f'\u26a0 Game folder not found: {game_dir}')
        sys.exit(1)
    os.chdir(game_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video_path')
    ap.add_argument('--game', required=True)
    ap.add_argument('--quarter', type=int, required=True)
    ap.add_argument('--start', type=float, required=True, help='Window start, elapsed sec')
    ap.add_argument('--end', type=float, required=True, help='Window end, elapsed sec')
    ap.add_argument('--n', type=int, default=8, help='Number of snapshots to save, evenly spread')
    ap.add_argument('--pad', type=float, default=15.0, help='Seconds of context before/after to include')
    ap.add_argument('--out', default='snapshots', help='Output folder for saved images')
    args = ap.parse_args()

    pin_to_game_folder(args.game)
    game_config = pc.load_game_config()
    FPS = pc.get_fps(game_config)
    label = pc.label_from_tracks_path(Path(args.video_path).with_suffix('.tracks.json'))
    qframes = pc.get_quarter_frames(game_config, label)

    if args.quarter not in qframes:
        print(f'\u26a0 Quarter {args.quarter} not found for {label}')
        sys.exit(1)
    qs, qe = qframes[args.quarter]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = args.start - args.pad
    t_end = args.end + args.pad
    step = (t_end - t_start) / max(args.n - 1, 1)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f'\u26a0 Could not open video: {args.video_path}')
        sys.exit(1)

    print(f'Saving {args.n} snapshot(s) from {label} Q{args.quarter}, '
          f'{t_start:.1f}s-{t_end:.1f}s, to {out_dir}/\n')

    for i in range(args.n):
        t = t_start + i * step
        fi = qs + int(t * FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            print(f'  \u26a0 Could not read frame {fi} (t={t:.1f}s)')
            continue
        inside = args.start <= t <= args.end
        tag = 'INSIDE' if inside else 'context'
        fname = out_dir / f'{i:02d}_t{t:.1f}s_{tag}.jpg'
        cv2.imwrite(str(fname), frame)
        print(f'  Saved {fname}  (t={t:.1f}s, frame={fi}, {tag})')

    cap.release()
    print(f'\nDone. Open the files in {out_dir}/ directly, or upload them back for a look.')


if __name__ == '__main__':
    main()
