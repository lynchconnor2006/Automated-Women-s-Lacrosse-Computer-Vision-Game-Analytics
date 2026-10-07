"""
check_fragment_gaps.py
For each detected possession, finds the gap to the PREVIOUS and NEXT
possession of the SAME team in the SAME quarter - to check whether a
short, isolated possession is actually a fragment of an adjacent real
possession that got split by a gap slightly longer than GRACE_SEC, rather
than a genuinely separate play.

A small gap (a few seconds) on either side of a short possession is a
strong sign it's a fragment; a large gap on both sides means it's likely
a real, standalone short possession.

Usage:
    python check_fragment_gaps.py detected_possessions.json --team UNC
    python check_fragment_gaps.py detected_possessions.json --team UNC --max-duration 15
"""
import json
import os
import sys
import argparse
from pathlib import Path


def pin_to_game_folder(game_name):
    base_dir = Path(__file__).resolve().parent.parent
    game_dir = base_dir / 'games' / game_name
    if not game_dir.exists():
        print(f'\u26a0 Game folder not found: {game_dir}')
        sys.exit(1)
    os.chdir(game_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('detected_path')
    ap.add_argument('--game', required=True, help='Game folder name, e.g. SYR_UNC_20260213')
    ap.add_argument('--team', default=None, help='Only show this team (default: all)')
    ap.add_argument('--max-duration', type=float, default=None,
                     help='Only show possessions at or under this duration in seconds '
                          '(e.g. 15 to focus on the short suspects)')
    args = ap.parse_args()

    pin_to_game_folder(args.game)

    with open(args.detected_path) as f:
        detected = json.load(f)

    by_quarter_team = {}
    for d in detected:
        key = (d['quarter'], d['team'])
        by_quarter_team.setdefault(key, []).append(d)
    for key in by_quarter_team:
        by_quarter_team[key].sort(key=lambda d: d['start_sec'])

    print(f"{'Quarter':<9}{'Team':<6}{'Start':<9}{'End':<9}{'Dur':<8}"
          f"{'PrevGap':<12}{'NextGap':<12}")
    print("-" * 65)

    for d in sorted(detected, key=lambda d: (d['quarter'], d['start_sec'])):
        if args.team and d['team'] != args.team:
            continue
        if args.max_duration is not None and d['duration_sec'] > args.max_duration:
            continue

        key = (d['quarter'], d['team'])
        siblings = by_quarter_team[key]
        idx = siblings.index(d)
        prev_gap = d['start_sec'] - siblings[idx - 1]['end_sec'] if idx > 0 else None
        next_gap = siblings[idx + 1]['start_sec'] - d['end_sec'] if idx + 1 < len(siblings) else None
        prev_str = f"{prev_gap:.1f}s" if prev_gap is not None else "(first)"
        next_str = f"{next_gap:.1f}s" if next_gap is not None else "(last)"

        flag = ''
        if (prev_gap is not None and prev_gap < 15) or (next_gap is not None and next_gap < 15):
            flag = '  <- close to a neighbor, possible fragment'

        print(f"Q{d['quarter']:<8}{d['team']:<6}{d['start_sec']:<9}{d['end_sec']:<9}"
              f"{d['duration_sec']:<8}{prev_str:<12}{next_str:<12}{flag}")


if __name__ == '__main__':
    main()
