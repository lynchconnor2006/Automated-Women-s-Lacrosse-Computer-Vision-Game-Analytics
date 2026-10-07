"""
check_full_zone_breakdown.py
For a specific quarter/camera/time window, prints a FULL per-zone
breakdown (Behind Net, Key, Net-to-8, Perimeter, AND Beyond 30 - every
zone, not just the ones that count toward possession detection) for a
given team.

This directly tests the "structural, not a detection gap" hypothesis for
a free-position-style possession: if real players ARE showing up in
numbers, just in Beyond 30 (outside where possession-detection looks),
that confirms it's a zone-coverage/threshold question, not a raw
detection-quality problem. If counts are near-zero across EVERY zone
including Beyond 30, that points back toward a genuine detection gap.

Usage:
    python check_full_zone_breakdown.py "SYR UNC LR Side.tracks.json" \
        --game SYR_UNC_20260213 --quarter 4 --start 629 --end 675 --team SYR
"""
import json
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as pc

ALL_ZONES = ('Behind Net', 'Key', 'Net-to-8', 'Perimeter', 'Beyond 30')


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
    ap.add_argument('--quarter', type=int, required=True)
    ap.add_argument('--start', type=float, required=True, help='Window start, elapsed seconds')
    ap.add_argument('--end', type=float, required=True, help='Window end, elapsed seconds')
    ap.add_argument('--team', required=True, help='Team to break down (e.g. SYR)')
    ap.add_argument('--sample-step', type=float, default=5.0)
    ap.add_argument('--pad', type=float, default=10.0)
    args = ap.parse_args()

    pin_to_game_folder(args.game)
    game_config = pc.load_game_config()
    aliases = game_config.get('team_aliases', {})
    team = aliases.get(args.team, args.team)

    FPS = pc.get_fps(game_config)
    label = pc.label_from_tracks_path(Path(args.tracks_path))
    qframes = pc.get_quarter_frames(game_config, label)
    tracks = load_tracks(args.tracks_path)

    if args.quarter not in qframes:
        print(f'\u26a0 Quarter {args.quarter} not found for {label}')
        sys.exit(1)
    qs, qe = qframes[args.quarter]

    print(f"Full zone breakdown for {team} on [{label}], Q{args.quarter}, "
          f"{args.start}s-{args.end}s (\u00b1{args.pad}s padding):\n")

    header = f"  {'t':>9}" + ''.join(f"  {z[:10]:>10}" for z in ALL_ZONES) + f"  {'TOTAL':>7}"
    print(header)

    t = args.start - args.pad
    end_t = args.end + args.pad
    while t <= end_t:
        fi = qs + int(t * FPS)
        dets = tracks.get(fi, [])
        counts = {z: 0 for z in ALL_ZONES}
        for d in dets:
            if d.get('team') != team:
                continue
            zone = d.get('zone')
            if zone in counts:
                counts[zone] += 1
        total = sum(counts.values())
        marker = '  <- inside window' if args.start <= t <= args.end else ''
        row = f"  {t:8.1f}s" + ''.join(f"  {counts[z]:>10d}" for z in ALL_ZONES) + f"  {total:>7d}"
        print(row + marker)
        t += args.sample_step

    print(f"\nIf 'Beyond 30' shows real counts during the window while the other zones stay low,")
    print(f"that's evidence players ARE on screen, just outside the zones possession-detection")
    print(f"counts - a zone-coverage/threshold question, not a raw detection-quality gap.")
    print(f"If EVERY column stays near zero throughout, that points back toward a genuine")
    print(f"detection gap instead.")


if __name__ == '__main__':
    main()
