"""
apply_possession_corrections.py
Applies any manual end-time corrections recorded during tagging
(tag_possessions.py's 'X' key, saved in possession_results.json's
corrected_end_sec field) to the detector's raw possession list, producing
a corrected list for Step 7 (formations) and Step 8 (cuts) to consume
instead of the raw detected_possessions.json directly - so a possession
the detector ran too long on (the rare rapid-scramble/repeated-turnover
cases we can't fully fix automatically) doesn't blur its formation/cut
data with frames from after the play actually ended.

If possession_results.json doesn't exist yet (no tagging done), or a
given possession has no correction recorded, it passes through unchanged.

Usage:
    python apply_possession_corrections.py detected_possessions.json possession_results.json

Output:
    detected_possessions_final.json - same list, with end_sec/duration_sec
    trimmed wherever a corrected_end_sec was recorded during tagging. Use
    THIS file (not detected_possessions.json) for formation/cut analysis.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as pc

pc.pin_working_directory()


def possession_id(p):
    return f"Q{p['quarter']}_{p['team']}_{p['start_sec']}_{p['end_sec']}"


def main():
    if len(sys.argv) < 3:
        print("Usage: python apply_possession_corrections.py detected_possessions.json "
              "possession_results.json")
        sys.exit(1)

    detected_path = Path(sys.argv[1])
    results_path = Path(sys.argv[2])

    with open(detected_path) as f:
        detected = json.load(f)

    results_by_id = {}
    if results_path.exists():
        with open(results_path) as f:
            for r in json.load(f):
                results_by_id[r['possession_id']] = r
    else:
        print(f"No {results_path.name} found yet (no tagging done) - "
              f"passing all possessions through unchanged")

    n_corrected = 0
    for p in detected:
        pid = possession_id(p)
        result = results_by_id.get(pid)
        if result and result.get('corrected_end_sec') is not None:
            new_end = result['corrected_end_sec']
            if new_end > p['start_sec']:
                print(f"  {pid}: trimming end {p['end_sec']}s -> {new_end}s "
                      f"(manual correction from tagging)")
                p['end_sec'] = new_end
                p['duration_sec'] = round(new_end - p['start_sec'], 1)
                n_corrected += 1
            else:
                print(f"  {pid}: corrected_end_sec ({new_end}s) is not after start_sec "
                      f"({p['start_sec']}s) - ignoring, keeping original end")

    out_path = Path('detected_possessions_final.json')
    with open(out_path, 'w') as f:
        json.dump(detected, f, indent=2)

    print(f"\n\u2713 Applied {n_corrected} manual correction(s), saved {len(detected)} "
          f"possessions to {out_path}")
    print("Use THIS file (not detected_possessions.json) for formation/cut analysis.")


if __name__ == '__main__':
    main()
