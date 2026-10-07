"""
compare_raw_vs_aligned.py
Compares official_possessions.json (raw, game-clock-based) against
official_possessions_aligned.json (after align_clocks.py's piecewise
mapping into video time), possession by possession within each quarter, to
isolate whether an implausibly long possession originated in the raw
parse (parse_playbyplay.py missed a possession-ending event, silently
merging two real possessions into one) or was introduced by the clock
alignment stretching one interpolation segment too aggressively.

Assumes both files preserve the same order within each quarter (true as
long as align_clocks.py doesn't reorder or drop entries, which it doesn't).

Usage:
    python compare_raw_vs_aligned.py official_possessions.json official_possessions_aligned.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as pc

pc.pin_working_directory()


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    if len(sys.argv) < 3:
        print('Usage: python compare_raw_vs_aligned.py official_possessions.json official_possessions_aligned.json')
        sys.exit(1)

    raw = load(sys.argv[1])
    aligned = load(sys.argv[2])

    quarters = sorted(set(p['quarter'] for p in raw))
    flagged = []

    for q in quarters:
        raw_q = [p for p in raw if p['quarter'] == q]
        aligned_q = [p for p in aligned if p['quarter'] == q]
        if len(raw_q) != len(aligned_q):
            print(f'⚠ Q{q}: raw has {len(raw_q)} possessions but aligned has {len(aligned_q)} — '
                  f'counts should match! Something dropped/added entries.')
            continue

        print(f'\nQ{q}:')
        for r, a in zip(raw_q, aligned_q):
            stretch = a['duration_sec'] / max(r['duration_sec'], 0.1)
            flag = '  <== SUSPICIOUS' if stretch > 2.5 or a['duration_sec'] > 200 else ''
            print(f"  {r['team']:>9}  raw={r['duration_sec']:>6.1f}s  "
                  f"aligned={a['duration_sec']:>6.1f}s  stretch={stretch:.2f}x{flag}")
            if flag:
                flagged.append((q, r, a, stretch))

    if flagged:
        print(f'\n{len(flagged)} suspicious possessions (stretch > 2.5x or aligned duration > 200s):')
        for q, r, a, stretch in flagged:
            print(f"  Q{q} {r['team']:>9}  raw {r['start_sec']}s-{r['end_sec']}s ({r['duration_sec']}s)  "
                  f"-> aligned {a['start_sec']}s-{a['end_sec']}s ({a['duration_sec']}s)  "
                  f"stretch={stretch:.2f}x")
        print('\nIf raw durations already look implausibly long (60s+), the bug is likely in')
        print('parse_playbyplay.py silently merging possessions (a missed ending event).')
        print('If raw durations look normal but stretch ratios are large, the bug is in')
        print('align_clocks.py stretching a sparsely-anchored segment too aggressively.')


if __name__ == '__main__':
    main()
