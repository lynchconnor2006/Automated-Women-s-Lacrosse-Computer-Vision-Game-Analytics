"""
check_long_possession_internal_gaps.py
For a specific long/flagged detected possession, re-derives the SAME
strict/loose signal used to build it (matching detect_and_match_possessions.py's
own thresholds and GRACE_SEC bridging exactly) and prints it at fine
resolution across the whole span - so you can see directly whether there's
a real internal gap close to (but under) GRACE_SEC that got bridged, versus
genuinely continuous signal throughout (a real long possession, not a merge
artifact).

This tests the "grace-bridging merged two real possessions" hypothesis
directly against the actual signal, without needing an old detected_
possessions.json to diff against.

Usage:
    python check_long_possession_internal_gaps.py "SYR UNC LR Side.tracks.json" \
        --game SYR_UNC_20260213 --quarter 2 --team SYR --off-team SYR --def-team UNC \
        --start 1515.5 --end 1744.5
"""
import json
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as pc

PLAY_ZONES = ('Key', 'Net-to-8', 'Behind Net', 'Perimeter')
DEEP_ZONES = ('Key', 'Net-to-8', 'Behind Net')

# Must match detect_and_match_possessions.py's own current values exactly,
# or this won't actually reproduce the signal that built the possession.
OFFENSE_MIN_FRAME = 6
DEFENSE_MIN_FRAME = 6
DEEP_OFFENSE_MIN = 3
CONTINUATION_OFFENSE_MIN = 5
CONTINUATION_DEFENSE_MIN = 5
GRACE_SEC = 9
TIME_STEP = 0.5


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
    ap.add_argument('--off-team', required=True)
    ap.add_argument('--def-team', required=True)
    ap.add_argument('--start', type=float, required=True, help='Possession start, elapsed sec')
    ap.add_argument('--end', type=float, required=True, help='Possession end, elapsed sec')
    ap.add_argument('--pad', type=float, default=15.0)
    args = ap.parse_args()

    pin_to_game_folder(args.game)
    game_config = pc.load_game_config()
    aliases = game_config.get('team_aliases', {})
    off_team = aliases.get(args.off_team, args.off_team)
    def_team = aliases.get(args.def_team, args.def_team)

    FPS = pc.get_fps(game_config)
    label = pc.label_from_tracks_path(Path(args.tracks_path))
    qframes = pc.get_quarter_frames(game_config, label)
    tracks = load_tracks(args.tracks_path)

    if args.quarter not in qframes:
        print(f'\u26a0 Quarter {args.quarter} not found for {label}')
        sys.exit(1)
    qs, qe = qframes[args.quarter]

    t_start = args.start - args.pad
    t_end = args.end + args.pad

    print(f"Re-derived strict/loose signal for [{label}] Q{args.quarter}, "
          f"{off_team} (off) vs {def_team} (def), {args.start}s-{args.end}s "
          f"(\u00b1{args.pad}s padding):\n")
    print(f"  {'t':>9}  {'off':>4}  {'def':>4}  {'deep':>5}  {'STRICT':>7}  {'LOOSE':>6}   note")

    gap_start = None
    max_gap = 0.0
    t = t_start
    loose_prev = None
    while t <= t_end:
        fi = qs + int(t * FPS)
        dets = tracks.get(fi, [])
        off = deff = deep_off = 0
        for d in dets:
            zone = d.get('zone')
            if zone not in PLAY_ZONES:
                continue
            if d.get('team') == off_team:
                off += 1
                if zone in DEEP_ZONES:
                    deep_off += 1
            elif d.get('team') == def_team or d.get('team') == 'Goalie':
                deff += 1

        strict = off >= OFFENSE_MIN_FRAME and deff >= DEFENSE_MIN_FRAME and deep_off >= DEEP_OFFENSE_MIN
        loose = off >= CONTINUATION_OFFENSE_MIN and deff >= CONTINUATION_DEFENSE_MIN

        note = ''
        if not loose:
            if gap_start is None:
                gap_start = t
        else:
            if gap_start is not None:
                gap_len = t - gap_start
                max_gap = max(max_gap, gap_len)
                bridged = '  <- BRIDGED by GRACE_SEC (gap < 9s)' if gap_len < GRACE_SEC else \
                          '  <- REAL BREAK (gap >= 9s, would split the possession)'
                note = f'gap ended, was {gap_len:.1f}s{bridged}'
                gap_start = None

        marker = '  <- inside possession' if args.start <= t <= args.end else ''
        print(f"  {t:8.1f}s  {off:4d}  {deff:4d}  {deep_off:5d}  {str(strict):>7}  "
              f"{str(loose):>6}   {note}{marker}")
        t += TIME_STEP

    if gap_start is not None:
        gap_len = t_end - gap_start
        max_gap = max(max_gap, gap_len)
        print(f"\n  (window ended mid-gap, {gap_len:.1f}s so far)")

    print(f"\nLargest loose-signal gap found in this window: {max_gap:.1f}s "
          f"(GRACE_SEC={GRACE_SEC})")
    if max_gap >= GRACE_SEC:
        print("  This would NOT have been bridged - if the possession still spans across it, "
              "something else merged it (not grace-bridging).")
    elif max_gap > GRACE_SEC * 0.5:
        print("  This gap is close to the bridging limit - worth judging whether it looks like "
              "a real momentary dip or a genuine possession change.")
    else:
        print("  No gap close to the bridging limit was found - the long possession's signal "
              "genuinely stayed continuous, which argues AGAINST a bridging-caused merge.")


if __name__ == '__main__':
    main()
