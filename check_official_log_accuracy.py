"""
check_official_log_accuracy.py
For possessions whose video signal reads completely silent on BOTH
cameras across a stretch well beyond ordinary alignment slop, this checks
whether the underlying OFFICIAL SOURCE TEXT might be the actual problem
(a transcription error in the raw play-by-play), not our detection or
alignment.

Fully automatic: converts the RAW (pre-alignment) official time to a
game-clock countdown, finds the correct quarter's section of the raw
play-by-play text, and prints every line whose own clock timestamp falls
near that value - so a typo or a clock jump is visible without any manual
lookup, conversion, or grep.

Usage:
    python check_official_log_accuracy.py official_possessions.json \
        official_possessions_aligned.json detected_possessions.json \
        playbyplay_UNC_CUSE_20260213.txt --game SYR_UNC_20260213 --team SYR
"""
import json
import os
import re
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as pc

LINE_RE = re.compile(r'^(?P<time>\d{2}:\d{2}|--)\s+(?P<team>\S+)\s*(?P<desc>.*)$')


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


def normalize_team(t, aliases):
    return aliases.get(t, t)


def sec_to_clock(remaining_sec):
    remaining_sec = max(0, remaining_sec)
    m = int(remaining_sec // 60)
    s = remaining_sec - m * 60
    return f"{m:02d}:{s:05.2f}"


def parse_clock_to_sec(clock_str):
    m, s = clock_str.split(':')
    return int(m) * 60 + float(s)


def split_into_quarters(lines):
    """Returns list of line-lists, one per quarter, splitting on
    'End-of-period' markers - matches parse_playbyplay.py's own convention."""
    quarters = []
    current = []
    for line in lines:
        current.append(line)
        if 'end-of-period' in line.lower():
            quarters.append(current)
            current = []
    if current:
        quarters.append(current)
    return quarters


def group_by_quarter_team(entries):
    g = {}
    for e in entries:
        g.setdefault((e['quarter'], e['team']), []).append(e)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('official_raw_path')
    ap.add_argument('official_aligned_path')
    ap.add_argument('detected_path')
    ap.add_argument('playbyplay_txt')
    ap.add_argument('--game', required=True)
    ap.add_argument('--team', required=True)
    ap.add_argument('--overlap-window', type=float, default=15.0,
                     help='A detected possession within this many seconds of an official '
                          'one counts as a match (default 15s)')
    ap.add_argument('--clock-tolerance', type=float, default=20.0,
                     help='Show raw text lines within this many seconds of the target clock time')
    args = ap.parse_args()

    pin_to_game_folder(args.game)
    game_config = pc.load_game_config()
    period_length_sec = pc.get_period_length_sec(game_config)
    aliases = game_config.get('team_aliases', {})
    team = normalize_team(args.team, aliases)

    official_raw = load_json(args.official_raw_path)
    official_aligned = load_json(args.official_aligned_path)
    for o in official_raw:
        o['team'] = normalize_team(o['team'], aliases)
    for o in official_aligned:
        o['team'] = normalize_team(o['team'], aliases)
    detected = load_json(args.detected_path)

    # Both official_raw and official_aligned are produced by iterating the
    # SAME original parsed possession list in the SAME order, grouped by
    # quarter (align_clocks.py never reorders) - so the Nth (quarter, team)
    # entry in one corresponds to the Nth in the other.
    raw_groups = group_by_quarter_team(official_raw)
    aligned_groups = group_by_quarter_team(official_aligned)

    detected_team_entries = [d for d in detected if d['team'] == team]

    missing_with_raw = []
    for key, aligned_list in aligned_groups.items():
        q, t = key
        if t != team:
            continue
        raw_list = raw_groups.get(key, [])
        for i, op in enumerate(aligned_list):
            has_match = any(
                d['quarter'] == op['quarter'] and
                not (d['end_sec'] < op['start_sec'] - args.overlap_window or
                     d['start_sec'] > op['end_sec'] + args.overlap_window)
                for d in detected_team_entries
            )
            if not has_match:
                raw_entry = raw_list[i] if i < len(raw_list) else None
                missing_with_raw.append((op, raw_entry))

    print(f"{len(missing_with_raw)} missing {team} possession(s) - checking raw source text "
          f"for each:\n")

    with open(args.playbyplay_txt) as f:
        all_lines = [l.rstrip('\n') for l in f]
    quarter_blocks = split_into_quarters(all_lines)

    for op, raw_entry in sorted(missing_with_raw, key=lambda x: (x[0]['quarter'], x[0]['start_sec'])):
        q = op['quarter']
        print(f"{'='*90}")
        print(f"Q{q} {team}  ALIGNED {op['start_sec']}s-{op['end_sec']}s  "
              f"(end_reason={op.get('end_reason','?')})")

        if raw_entry is None:
            print("  \u26a0 Could not find a corresponding RAW (pre-alignment) entry - skipping\n")
            continue

        raw_start_remaining = period_length_sec - raw_entry['start_sec']
        raw_end_remaining = period_length_sec - raw_entry['end_sec']
        print(f"  RAW (pre-alignment) official time: elapsed {raw_entry['start_sec']}s-"
              f"{raw_entry['end_sec']}s  ->  game clock {sec_to_clock(raw_start_remaining)} "
              f"down to {sec_to_clock(raw_end_remaining)}")

        if q - 1 >= len(quarter_blocks) or q - 1 < 0:
            print(f"  \u26a0 Quarter {q} not found in play-by-play text - skipping\n")
            continue
        block = quarter_blocks[q - 1]

        print(f"  Raw text lines near this clock range (\u00b1{args.clock_tolerance:.0f}s):")
        printed_any = False
        lo = raw_end_remaining - args.clock_tolerance
        hi = raw_start_remaining + args.clock_tolerance
        for line in block:
            m = LINE_RE.match(line.strip())
            if not m or m.group('time') == '--':
                continue
            try:
                line_remaining = parse_clock_to_sec(m.group('time'))
            except ValueError:
                continue
            if lo <= line_remaining <= hi:
                marker = ''
                if abs(line_remaining - raw_start_remaining) < 1 or abs(line_remaining - raw_end_remaining) < 1:
                    marker = '  <-- matches official start/end'
                print(f"    {line.strip()}{marker}")
                printed_any = True
        if not printed_any:
            print("    (no lines found in this clock range - check quarter/clock math)")
        print()


if __name__ == '__main__':
    main()
