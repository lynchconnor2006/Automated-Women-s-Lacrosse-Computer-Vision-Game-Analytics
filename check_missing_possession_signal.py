"""
check_missing_possession_signal.py
Finds official possessions for a given team that have NO matching detected
possession nearby, then prints the raw off/def/deep-off counts (already
sitting in tracks.json, no video needed) sampled every few seconds around
that official window - so you can see whether it was a near-miss (counts
just under the CONTINUATION_OFFENSE_MIN/DEFENSE_MIN thresholds in
detect_and_match_possessions.py) or a genuine detection gap (counts near
zero throughout).

Usage:
    python check_missing_possession_signal.py official_possessions_aligned.json \
        detected_possessions.json "SYR UNC MVC Side.tracks.json" "SYR UNC LR Side.tracks.json" \
        goalie_team_reconciled_SYR_UNC_MVC_Side.json goalie_team_reconciled_SYR_UNC_LR_Side.json \
        --team SYR
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


FPS = None
TEAM_ALIASES = {}
PLAY_ZONES = ('Key', 'Net-to-8', 'Behind Net', 'Perimeter')
DEEP_ZONES = ('Key', 'Net-to-8', 'Behind Net')


def normalize_team(t):
    return TEAM_ALIASES.get(t, t)


def load_json(p):
    with open(p) as f:
        return json.load(f)


def load_tracks(p):
    raw = load_json(p)
    return {int(k): v for k, v in raw.items()}


def find_camera_for_team(goalie_team_by_label, q, team):
    for label, per_quarter in goalie_team_by_label.items():
        def_team = per_quarter.get(q)
        if def_team is not None and def_team != team:
            return label
    return None


def main():
    global TEAM_ALIASES, FPS

    ap = argparse.ArgumentParser()
    ap.add_argument('official_path')
    ap.add_argument('detected_path')
    ap.add_argument('cam1_tracks')
    ap.add_argument('cam2_tracks')
    ap.add_argument('goalie1')
    ap.add_argument('goalie2')
    ap.add_argument('--game', required=True, help='Game folder name, e.g. SYR_UNC_20260213')
    ap.add_argument('--team', required=True, help='e.g. SYR or UNC (normalized team name)')
    ap.add_argument('--overlap-window', type=float, default=15.0,
                     help='A detected possession within this many seconds of an official '
                          'one counts as a match (default 15s)')
    ap.add_argument('--sample-step', type=float, default=3.0, help='Seconds between printed samples')
    ap.add_argument('--pad', type=float, default=15.0,
                     help='Seconds of context before/after the official window to print')
    args = ap.parse_args()

    pin_to_game_folder(args.game)
    game_config = pc.load_game_config()
    FPS = pc.get_fps(game_config)
    TEAM_ALIASES = game_config.get('team_aliases', {})

    team = normalize_team(args.team)

    official = load_json(args.official_path)
    for o in official:
        o['team'] = normalize_team(o['team'])
    detected = load_json(args.detected_path)

    tracks_by_label = {}
    qframes_by_label = {}
    goalie_by_label = {}
    for tp, gp in [(args.cam1_tracks, args.goalie1), (args.cam2_tracks, args.goalie2)]:
        label = pc.label_from_tracks_path(Path(tp))
        tracks_by_label[label] = load_tracks(tp)
        qframes_by_label[label] = pc.get_quarter_frames(game_config, label)
        raw_g = load_json(gp)
        goalie_by_label[label] = {int(k): v for k, v in raw_g.items()}

    official_team_entries = [o for o in official if o['team'] == team]
    detected_team_entries = [d for d in detected if d['team'] == team]

    missing = []
    for op in official_team_entries:
        has_match = any(
            d['quarter'] == op['quarter'] and
            not (d['end_sec'] < op['start_sec'] - args.overlap_window or
                 d['start_sec'] > op['end_sec'] + args.overlap_window)
            for d in detected_team_entries
        )
        if not has_match:
            missing.append(op)

    print(f"{len(missing)} official {team} possession(s) with no detected match within "
          f"{args.overlap_window}s (out of {len(official_team_entries)} total official {team} "
          f"possessions):\n")

    for op in missing:
        q = op['quarter']
        camera = find_camera_for_team(goalie_by_label, q, team)
        if camera is None:
            print(f"Q{q} {team} {op['start_sec']}s-{op['end_sec']}s: no camera found for this "
                  f"team/quarter, skipping\n")
            continue
        def_team = goalie_by_label[camera][q]
        tracks = tracks_by_label[camera]
        qs, qe = qframes_by_label[camera][q]

        print(f"{'='*72}")
        print(f"Q{q} {team} official {op['start_sec']}s-{op['end_sec']}s  "
              f"(end_reason={op.get('end_reason','?')})  camera=[{camera}]")
        print(f"{'='*72}")

        t = op['start_sec'] - args.pad
        end_t = op['end_sec'] + args.pad
        while t <= end_t:
            fi = qs + int(t * FPS)
            dets = tracks.get(fi, [])
            off = deff = deep_off = 0
            for d in dets:
                zone = d.get('zone')
                if zone not in PLAY_ZONES:
                    continue
                if d.get('team') == team:
                    off += 1
                    if zone in DEEP_ZONES:
                        deep_off += 1
                elif d.get('team') == def_team or d.get('team') == 'Goalie':
                    deff += 1
            marker = '  <- inside official window' if op['start_sec'] <= t <= op['end_sec'] else ''
            print(f"  t={t:8.1f}s   off={off:2d}   def={deff:2d}   deep_off={deep_off:2d}{marker}")
            t += args.sample_step
        print()


if __name__ == '__main__':
    main()
