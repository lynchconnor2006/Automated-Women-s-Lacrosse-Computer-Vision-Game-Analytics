"""
diagnose_unmatched_by_team.py
For every OFFICIAL possession that has no matching detected window at all,
checks the PEAK offense/defense/deep-zone counts reached anywhere in a
buffered window around its (aligned) official time, on the correct camera.
Groups results by team, so we can see directly whether one team's peaks are
systematically lower than the other's (a real detection-recall problem,
likely traceable to Colab) versus comparable (pointing at something else -
alignment/logic - instead).

Usage:
    python diagnose_unmatched_by_team.py official_possessions_aligned.json \\
        detected_possessions.json \\
        "SYR UNC MVC Side.tracks.json" "SYR UNC LR Side.tracks.json" \\
        "goalie_team_reconciled_SYR UNC MVC Side.json" "goalie_team_reconciled_SYR UNC LR Side.json"
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as pc

pc.pin_working_directory()
GAME_CONFIG = pc.load_game_config()
FPS = pc.get_fps(GAME_CONFIG)
other_team = pc.make_other_team(GAME_CONFIG)

PLAY_ZONES = ('Key', 'Behind Net', 'Perimeter')
DEEP_ZONES = ('Key', 'Behind Net')
BUFFER_SEC = 20


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_tracks(path):
    raw = load_json(path)
    return {int(k): v for k, v in raw.items()}


def overlap(a_s, a_e, b_s, b_e):
    return max(0.0, min(a_e, b_e) - max(a_s, b_s))


def peak_counts(tracks, qs, qe, off_team, def_team, t_lo, t_hi):
    f_lo = qs + int(t_lo * FPS)
    f_hi = qs + int(t_hi * FPS)
    peak_off = peak_def = peak_deep = 0
    for fi in range(max(f_lo, qs), min(f_hi, qe) + 1):
        dets = tracks.get(fi)
        if not dets:
            continue
        off = deff = deep = 0
        for d in dets:
            zone = d.get('zone')
            if zone not in PLAY_ZONES:
                continue
            if d.get('team') == off_team:
                off += 1
                if zone in DEEP_ZONES:
                    deep += 1
            elif d.get('team') == def_team or d.get('team') == 'Goalie':
                deff += 1
        peak_off = max(peak_off, off)
        peak_def = max(peak_def, deff)
        peak_deep = max(peak_deep, deep)
    return peak_off, peak_def, peak_deep


def main():
    if len(sys.argv) < 7:
        print("Usage: python diagnose_unmatched_by_team.py official_possessions_aligned.json "
              "detected_possessions.json <cam1.tracks.json> <cam2.tracks.json> "
              "<goalie_cam1.json> <goalie_cam2.json>")
        sys.exit(1)

    official = load_json(sys.argv[1])
    detected = load_json(sys.argv[2])
    cam_paths = [Path(sys.argv[3]), Path(sys.argv[4])]
    goalie_paths = [Path(sys.argv[5]), Path(sys.argv[6])]

    tracks_by_label = {}
    goalie_by_label = {}
    qframes_by_label = {}
    for cam_path, goalie_path in zip(cam_paths, goalie_paths):
        label = pc.label_from_tracks_path(cam_path)
        tracks_by_label[label] = load_tracks(cam_path)
        qframes_by_label[label] = pc.get_quarter_frames(GAME_CONFIG, label)
        raw_g = load_json(goalie_path)
        goalie_by_label[label] = {int(k): v for k, v in raw_g.items()}

    unmatched = []
    for op in official:
        any_overlap = any(overlap(op['start_sec'], op['end_sec'], d['start_sec'], d['end_sec']) > 0
                          for d in detected if d['quarter'] == op['quarter'] and d['team'] == op['team'])
        if not any_overlap:
            unmatched.append(op)

    by_team = defaultdict(list)
    for op in unmatched:
        team = op['team']
        q = op['quarter']
        # find which camera this team attacks this quarter
        camera = None
        for label in tracks_by_label:
            defending = goalie_by_label[label].get(q)
            if defending is not None and defending != team:
                camera = label
                break
        if camera is None:
            continue
        qs, qe = qframes_by_label[camera][q]
        def_team = other_team(team)
        t_lo = max(0.0, op['start_sec'] - BUFFER_SEC)
        t_hi = op['end_sec'] + BUFFER_SEC
        po, pd, pdeep = peak_counts(tracks_by_label[camera], qs, qe, team, def_team, t_lo, t_hi)
        by_team[team].append((op, po, pd, pdeep, camera))

    print(f"{len(unmatched)} official possessions have no detected window at all\n")
    for team, entries in by_team.items():
        print(f"=== {team}: {len(entries)} unmatched possessions ===")
        for op, po, pd, pdeep, camera in entries:
            print(f"  Q{op['quarter']} {op['start_sec']}s-{op['end_sec']}s ({op['duration_sec']}s) "
                  f"[{camera}]  peak_offense={po}  peak_defense={pd}  peak_deep={pdeep}")
        avg_po = sum(e[1] for e in entries) / len(entries)
        avg_pd = sum(e[2] for e in entries) / len(entries)
        print(f"  --> avg peak_offense={avg_po:.1f}  avg peak_defense={avg_pd:.1f}\n")


if __name__ == '__main__':
    main()
