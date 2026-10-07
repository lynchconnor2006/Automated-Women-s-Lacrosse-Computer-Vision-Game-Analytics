"""
reconcile_goalie_teams.py
Fixes noisy per-quarter goalie-team readings from identify_goalie_team.py
using two independent consistency checks:

  1. ALTERNATION: since ends switch every quarter, the defending team at a
     given camera must strictly alternate — Q1 must match Q3, Q2 must match
     Q4. Within each pair, the quarter with the bigger score gap (more
     decisive reading) overrides the noisier one.

  2. CROSS-CAMERA COMPLEMENTARITY: the two cameras show opposite nets, so
     whichever team defends camera A a given quarter, the OTHER team must
     defend camera B that same quarter. Used as a secondary check/tiebreaker.

Team names are derived directly from whatever appears in the input files
(no hardcoded team names), falling back to game_config.json's team_a/team_b
only if the data itself doesn't reveal both team names.

Usage:
    python reconcile_goalie_teams.py <goalie_team_cam1.json> <goalie_team_cam2.json>

Output:
    goalie_team_reconciled_<label>.json for each camera.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as pc

pc.pin_working_directory()


def load(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def label_from_path(path):
    name = Path(path).stem
    prefix = 'goalie_team_'
    return name[len(prefix):] if name.startswith(prefix) else name


def build_other_team(data):
    """Derive the two team names directly from the data (every 'dominant'
    value seen across both files). Falls back to game_config.json if the
    data alone doesn't reveal both names."""
    seen = set()
    for label_data in data.values():
        for entry in label_data.values():
            if entry is not None:
                seen.add(entry['dominant'])
    if len(seen) != 2:
        config = pc.load_game_config()
        a, b = pc.get_teams(config)
        seen = {a, b}
    a, b = sorted(seen)

    def other_team(team):
        return b if team == a else a
    return other_team


def main():
    if len(sys.argv) < 3:
        print('Usage: python reconcile_goalie_teams.py <goalie_team_cam1.json> <goalie_team_cam2.json>')
        sys.exit(1)

    paths = [Path(sys.argv[1]), Path(sys.argv[2])]
    labels = [label_from_path(p) for p in paths]
    data = {label: load(p) for label, p in zip(labels, paths)}
    other_team = build_other_team(data)

    for label in labels:
        print(f'{label} raw readings:')
        for q, entry in sorted(data[label].items()):
            if entry is None:
                print(f'  Q{q}: no data')
            else:
                print(f"  Q{q}: {entry['dominant']}  (gap={entry['gap']:.2f})")

    # --- Step 1: alternation check, per camera ---
    reconciled = {label: {} for label in labels}
    for label in labels:
        entries = data[label]
        for group in [(1, 3), (2, 4)]:
            group_entries = [(q, entries.get(q)) for q in group if entries.get(q) is not None]
            if not group_entries:
                for q in group:
                    reconciled[label][q] = None
                continue
            best_q, best_entry = max(group_entries, key=lambda x: x[1]['gap'])
            winner = best_entry['dominant']
            for q in group:
                reconciled[label][q] = winner

    print('\nAfter alternation check (within-camera Q1<->Q3, Q2<->Q4):')
    for label in labels:
        print(f'  {label}: {reconciled[label]}')

    # --- Step 2: cross-camera complementarity check ---
    if len(labels) == 2:
        a, b = labels
        conflicts = []
        for q in reconciled[a]:
            ta, tb = reconciled[a].get(q), reconciled[b].get(q)
            if ta is not None and tb is not None and ta == tb:
                conflicts.append(q)
        if conflicts:
            print(f'\n\u26a0 Cross-camera conflict in quarter(s) {conflicts}: both cameras show the '
                  f'SAME defending team, which is impossible. Resorting to whichever camera had '
                  f'the bigger gap for the conflicting quarter.')
            for q in conflicts:
                gap_a = data[a].get(q, {}).get('gap', 0) if data[a].get(q) else 0
                gap_b = data[b].get(q, {}).get('gap', 0) if data[b].get(q) else 0
                if gap_a >= gap_b:
                    reconciled[b][q] = other_team(reconciled[a][q])
                else:
                    reconciled[a][q] = other_team(reconciled[b][q])
        else:
            print('\n\u2713 No cross-camera conflicts \u2014 both cameras agree on complementary assignment every quarter.')

    for label in labels:
        out_path = Path(f'goalie_team_reconciled_{label}.json')
        with open(out_path, 'w') as f:
            json.dump(reconciled[label], f, indent=2)
        print(f'\n\u2713 Saved {out_path}: {reconciled[label]}')


if __name__ == '__main__':
    main()
