"""
diff_detected_possessions.py
Direct comparison between two detected_possessions.json snapshots (e.g.
before/after a model swap) - shows exactly which possessions disappeared,
which appeared, and which got merged (a possession in the OLD file whose
span is now fully covered by a single, longer possession in the NEW file
for the same team/quarter).

This replaces guessing about why a raw count changed with a direct,
possession-by-possession diff.

Usage:
    python diff_detected_possessions.py old_detected_possessions.json new_detected_possessions.json
"""
import json
import sys


def load(p):
    with open(p) as f:
        return json.load(f)


def overlaps(a, b):
    return not (a['end_sec'] < b['start_sec'] or a['start_sec'] > b['end_sec'])


def main():
    if len(sys.argv) != 3:
        print("Usage: python diff_detected_possessions.py <old.json> <new.json>")
        sys.exit(1)

    old = load(sys.argv[1])
    new = load(sys.argv[2])

    print(f"OLD: {len(old)} possessions   NEW: {len(new)} possessions   "
          f"(diff: {len(new) - len(old):+d})\n")

    old_by_qt = {}
    new_by_qt = {}
    for d in old:
        old_by_qt.setdefault((d['quarter'], d['team']), []).append(d)
    for d in new:
        new_by_qt.setdefault((d['quarter'], d['team']), []).append(d)

    print("=" * 90)
    print("OLD possessions with NO overlapping NEW possession (same quarter+team) - disappeared:")
    print("=" * 90)
    disappeared = []
    for key, olds in old_by_qt.items():
        news = new_by_qt.get(key, [])
        for o in olds:
            if not any(overlaps(o, n) for n in news):
                disappeared.append(o)
                print(f"  Q{o['quarter']} {o['team']:>4}  {o['start_sec']}s-{o['end_sec']}s  "
                      f"({o['duration_sec']}s)")
    if not disappeared:
        print("  None.")

    print(f"\n{'='*90}")
    print("NEW possessions with NO overlapping OLD possession (same quarter+team) - newly appeared:")
    print("=" * 90)
    appeared = []
    for key, news in new_by_qt.items():
        olds = old_by_qt.get(key, [])
        for n in news:
            if not any(overlaps(n, o) for o in olds):
                appeared.append(n)
                print(f"  Q{n['quarter']} {n['team']:>4}  {n['start_sec']}s-{n['end_sec']}s  "
                      f"({n['duration_sec']}s)")
    if not appeared:
        print("  None.")

    print(f"\n{'='*90}")
    print("Likely MERGES: 2+ OLD possessions (same quarter+team) that now overlap a SINGLE, "
          "longer NEW possession:")
    print("=" * 90)
    found_merge = False
    for key, news in new_by_qt.items():
        olds = old_by_qt.get(key, [])
        for n in news:
            covering = [o for o in olds if overlaps(n, o)]
            if len(covering) >= 2:
                found_merge = True
                print(f"  NEW Q{n['quarter']} {n['team']:>4}  {n['start_sec']}s-{n['end_sec']}s  "
                      f"({n['duration_sec']}s)  now covers {len(covering)} OLD possessions:")
                for o in covering:
                    print(f"      OLD: {o['start_sec']}s-{o['end_sec']}s  ({o['duration_sec']}s)")
    if not found_merge:
        print("  None found.")


if __name__ == '__main__':
    main()
