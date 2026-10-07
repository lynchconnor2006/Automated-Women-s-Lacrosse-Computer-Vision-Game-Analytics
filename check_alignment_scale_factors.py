"""
check_alignment_scale_factors.py
For each quarter, computes the LOCAL scale factor (video-seconds per
official-second) between every pair of ADJACENT anchor points used by
align_clocks.py's piecewise-linear mapping. A well-behaved quarter should
have segments close to 1.0x throughout (roughly real-time correspondence);
a segment far from 1.0x (either direction) means that specific stretch of
official time gets disproportionately stretched or compressed when mapped
to video time - which would explain a genuinely short real possession
turning into an implausibly long "official window" after alignment,
without needing to touch or re-run detection at all.

This directly targets possessions whose raw duration is much shorter than
their aligned duration, by checking which anchor segment their raw time
falls into and what that segment's scale factor is.

Usage:
    python check_alignment_scale_factors.py official_possessions.json \
        official_possessions_aligned.json --game SYR_UNC_20260213
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


def load_json(p):
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('official_raw_path')
    ap.add_argument('official_aligned_path')
    ap.add_argument('--game', required=True)
    ap.add_argument('--flag-threshold', type=float, default=1.5,
                     help='Flag any possession whose aligned/raw duration ratio exceeds this '
                          '(default 1.5x)')
    args = ap.parse_args()

    pin_to_game_folder(args.game)
    game_config = pc.load_game_config()
    aliases = game_config.get('team_aliases', {})

    def normalize_team(t):
        return aliases.get(t, t)

    official_raw = load_json(args.official_raw_path)
    official_aligned = load_json(args.official_aligned_path)
    for o in official_raw:
        o['team'] = normalize_team(o['team'])
    for o in official_aligned:
        o['team'] = normalize_team(o['team'])

    # Pair raw <-> aligned the same way as before: both lists are built by
    # iterating the same original order, grouped by quarter+team.
    def group(entries):
        g = {}
        for e in entries:
            g.setdefault((e['quarter'], e['team']), []).append(e)
        return g

    raw_groups = group(official_raw)
    aligned_groups = group(official_aligned)

    print(f"Possessions whose ALIGNED duration is >{args.flag_threshold}x their RAW "
          f"(pre-alignment) duration - each is evidence of a locally-distorted alignment "
          f"segment, not a genuinely long real possession:\n")

    flagged = []
    for key, aligned_list in aligned_groups.items():
        q, team = key
        raw_list = raw_groups.get(key, [])
        for i, aligned_entry in enumerate(aligned_list):
            if i >= len(raw_list):
                continue
            raw_entry = raw_list[i]
            raw_dur = raw_entry['end_sec'] - raw_entry['start_sec']
            aligned_dur = aligned_entry['end_sec'] - aligned_entry['start_sec']
            if raw_dur <= 0:
                continue
            ratio = aligned_dur / raw_dur
            if ratio > args.flag_threshold:
                flagged.append((q, team, raw_entry, aligned_entry, ratio))

    flagged.sort(key=lambda x: -x[4])
    for q, team, raw_entry, aligned_entry, ratio in flagged:
        print(f"  Q{q} {team:>10}  RAW {raw_entry['start_sec']}s-{raw_entry['end_sec']}s "
              f"({raw_entry['end_sec']-raw_entry['start_sec']:.1f}s)  ->  "
              f"ALIGNED {aligned_entry['start_sec']}s-{aligned_entry['end_sec']}s "
              f"({aligned_entry['end_sec']-aligned_entry['start_sec']:.1f}s)  "
              f"[{ratio:.2f}x stretch]")

    if not flagged:
        print("  None found at this threshold.")

    print(f"\n{'='*80}")
    print("Per-quarter anchor segment scale factors (video-seconds per official-second):")
    print(f"{'='*80}")
    print("(Requires re-deriving anchors is not done here - this is a simpler, complementary")
    print(" view: for each quarter, the OVERALL official->video ratio across consecutive")
    print(" possessions, to spot-check where the pace changes abruptly.)\n")

    # Build one combined, time-sorted list per quarter (both teams) to see
    # the overall progression of raw vs aligned time as the quarter goes on.
    by_quarter_raw = {}
    by_quarter_aligned = {}
    for (q, team), lst in raw_groups.items():
        by_quarter_raw.setdefault(q, []).extend(lst)
    for (q, team), lst in aligned_groups.items():
        by_quarter_aligned.setdefault(q, []).extend(lst)

    for q in sorted(by_quarter_raw):
        raw_sorted = sorted(by_quarter_raw[q], key=lambda e: e['start_sec'])
        aligned_sorted = sorted(by_quarter_aligned[q], key=lambda e: e['start_sec'])
        print(f"Q{q}:")
        for i in range(1, min(len(raw_sorted), len(aligned_sorted))):
            raw_gap = raw_sorted[i]['start_sec'] - raw_sorted[i-1]['start_sec']
            aligned_gap = aligned_sorted[i]['start_sec'] - aligned_sorted[i-1]['start_sec']
            if raw_gap <= 0:
                continue
            local_scale = aligned_gap / raw_gap
            flag = '  <-- unusual local pace' if (local_scale > 2.0 or local_scale < 0.5) else ''
            print(f"  official {raw_sorted[i-1]['start_sec']:.0f}s->{raw_sorted[i]['start_sec']:.0f}s "
                  f"(raw gap {raw_gap:.0f}s)  maps to  video gap {aligned_gap:.0f}s  "
                  f"[scale {local_scale:.2f}x]{flag}")
        print()


if __name__ == '__main__':
    main()
