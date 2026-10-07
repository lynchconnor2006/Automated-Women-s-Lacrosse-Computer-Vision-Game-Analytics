"""
find_obscure_frames.py
Scans a camera's tracks.json for SUSTAINED low-total-detection stretches -
a proxy for exactly the kind of scenario found this session (a real
free-position sequence with near-zero tracked players anywhere on screen,
in ANY zone including Beyond 30) - and saves a list of candidate frame
indices, spread out within each qualifying stretch, for targeted
annotation with annotate_players.py.

Usage:
    python find_obscure_frames.py "SYR UNC LR Side.tracks.json" \
        --game SYR_UNC_20260213 --out obscure_frames_LR.json
    python find_obscure_frames.py "SYR UNC MVC Side.tracks.json" \
        --game SYR_UNC_20260213 --out obscure_frames_MVC.json
"""
import json
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as pc


def pin_to_game_folder(game_name):
    base_dir = Path(__file__).resolve().parent.parent
    game_dir = base_dir / 'games' / game_name
    if not game_dir.exists():
        print(f'\u26a0 Game folder not found: {game_dir}')
        sys.exit(1)
    os.chdir(game_dir)


def load_tracks(p):
    with open(p) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tracks_path')
    ap.add_argument('--game', required=True)
    ap.add_argument('--out', required=True, help='Output JSON path for the candidate frame list')
    ap.add_argument('--low-count-threshold', type=int, default=4,
                     help='A frame with this many or fewer TOTAL detections (any team, any '
                          'zone including Beyond 30) counts as "low" (default 4)')
    ap.add_argument('--min-sustained-sec', type=float, default=3.0,
                     help='Only keep stretches of low-count frames lasting at least this long '
                          '(filters single-frame noise blips, not real sustained situations)')
    ap.add_argument('--samples-per-stretch', type=int, default=3,
                     help='How many representative frames to pull from each qualifying stretch')
    args = ap.parse_args()

    pin_to_game_folder(args.game)
    game_config = pc.load_game_config()
    FPS = pc.get_fps(game_config)
    label = pc.label_from_tracks_path(Path(args.tracks_path))
    qframes = pc.get_quarter_frames(game_config, label)
    tracks = load_tracks(args.tracks_path)

    candidate_frames = []

    for q, (qs, qe) in qframes.items():
        sorted_frames = sorted(fi for fi in tracks if qs <= fi <= qe)
        if not sorted_frames:
            continue

        counts = [(fi, len(tracks[fi])) for fi in sorted_frames]
        i = 0
        n = len(counts)
        n_stretches = 0
        while i < n:
            fi, c = counts[i]
            if c <= args.low_count_threshold:
                j = i
                while j < n and counts[j][1] <= args.low_count_threshold:
                    j += 1
                start_fi = counts[i][0]
                end_fi = counts[j - 1][0]
                duration_sec = (end_fi - start_fi) / FPS
                if duration_sec >= args.min_sustained_sec:
                    n_stretches += 1
                    stretch_frames = [f for f, _ in counts[i:j]]
                    step = max(1, len(stretch_frames) // args.samples_per_stretch)
                    picked = stretch_frames[::step][:args.samples_per_stretch]
                    candidate_frames.extend(picked)
                    print(f'  Q{q}: low-count stretch frames {start_fi}-{end_fi} '
                          f'({duration_sec:.1f}s) - sampled {len(picked)} frame(s)')
                i = j
            else:
                i += 1

    candidate_frames = sorted(set(candidate_frames))
    print(f'\n{len(candidate_frames)} candidate frame(s) found from {label}.')

    with open(args.out, 'w') as f:
        json.dump({'label': label, 'frames': candidate_frames}, f, indent=2)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
