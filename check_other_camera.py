"""
check_other_camera.py
For a given quarter and real elapsed-time range (as measured on ONE camera),
shows what BOTH cameras detected at the SAME quarter + elapsed time, side by
side. This directly tests whether a "dead" stretch on one camera is actually
live action happening at the other end (wrong camera/mapping issue) versus
both cameras being quiet at once (genuine stoppage, or a shared detection gap).

Usage:
    python check_other_camera.py <cam1_tracks.json> <cam2_tracks.json> <quarter> <start_sec> <end_sec>

Note: start_sec/end_sec are elapsed-into-quarter seconds as measured on
cam1's own clock (its own quarter_frames start). Since MVC and LR start
their quarters within a couple seconds of each other, this is close enough
to compare directly without needing a separate conversion step.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as pc

pc.pin_working_directory()
GAME_CONFIG = pc.load_game_config()
FPS = pc.get_fps(GAME_CONFIG)

PLAY_ZONES = ('Key', 'Behind Net', 'Perimeter')
DEEP_ZONES = ('Key', 'Behind Net')


def load_tracks(p):
    with open(p) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def nearest_frame(tracks, qs, qe, target_f, max_dist=15):
    best, best_d = None, max_dist
    for fi in tracks:
        if qs <= fi <= qe and abs(fi - target_f) < best_d:
            best_d = abs(fi - target_f)
            best = fi
    return best


def summarize(tracks, fi):
    if fi is None:
        return "no frame"
    dets = tracks.get(fi, [])
    by_team = {}
    deep = 0
    for d in dets:
        zone = d.get('zone')
        if zone not in PLAY_ZONES:
            continue
        team = d.get('team', '?')
        by_team[team] = by_team.get(team, 0) + 1
        if zone in DEEP_ZONES:
            deep += 1
    parts = ', '.join(f'{t}:{c}' for t, c in sorted(by_team.items()))
    return f"[{parts}]  deep={deep}"


def main():
    if len(sys.argv) < 6:
        print("Usage: python check_other_camera.py <cam1_tracks.json> <cam2_tracks.json> "
              "<quarter> <start_sec> <end_sec>")
        sys.exit(1)

    cam1_path, cam2_path = Path(sys.argv[1]), Path(sys.argv[2])
    q = int(sys.argv[3])
    start_sec = float(sys.argv[4])
    end_sec = float(sys.argv[5])

    label1, label2 = pc.label_from_tracks_path(cam1_path), pc.label_from_tracks_path(cam2_path)
    tracks1, tracks2 = load_tracks(cam1_path), load_tracks(cam2_path)
    qf1 = pc.get_quarter_frames(GAME_CONFIG, label1)[q]
    qf2 = pc.get_quarter_frames(GAME_CONFIG, label2)[q]

    print(f"Comparing {label1} vs {label2} at Q{q}, {start_sec}s-{end_sec}s "
          f"(elapsed time, sampled every 7s):\n")

    t = start_sec
    while t <= end_sec:
        f1_target = qf1[0] + int(t * FPS)
        f2_target = qf2[0] + int(t * FPS)
        b1 = nearest_frame(tracks1, qf1[0], qf1[1], f1_target)
        b2 = nearest_frame(tracks2, qf2[0], qf2[1], f2_target)
        s1 = summarize(tracks1, b1)
        s2 = summarize(tracks2, b2)
        print(f"t={t:7.1f}  {label1}: {s1}")
        print(f"{'':7}   {label2}: {s2}")
        print()
        t += 7.0


if __name__ == '__main__':
    main()
