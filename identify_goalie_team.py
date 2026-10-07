"""
identify_goalie_team.py
Determines which team's goalie defends each camera's net, per quarter, by
directly sampling the goalie's jersey color from the video.

WHY THIS INSTEAD OF CROWD COUNTING: aggregate near-net player counts turned
out not to reliably indicate either the attacking or defending team. The
goalie is different — he never leaves his own crease for long and never
changes teams mid-quarter, so his jersey color is a direct, unambiguous
signal of which team defends this camera's net that quarter.

Usage:
    python identify_goalie_team.py <video_path> <tracks.json>

Output:
    goalie_team_<label>.json — {quarter: team} — the team that defends this
    camera's net each quarter (so the ATTACKING team is the other one).
"""
import sys
import json
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as pc

pc.pin_working_directory()
GAME_CONFIG = pc.load_game_config()
TEAM_A, TEAM_B = pc.get_teams(GAME_CONFIG)

SAMPLES_PER_QUARTER = 15  # goalie-labeled frames to check, spread across the quarter


def load_tracks(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def get_jersey_hsv(frame, ltrb):
    l, t, r, b = ltrb
    h = b - t
    y0 = t + int(h * 0.15)
    y1 = t + int(h * 0.55)
    x0 = l + int((r - l) * 0.2)
    x1 = r - int((r - l) * 0.2)
    roi = frame[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if roi.size == 0:
        return None
    return cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


def score_team(hsv, lo, hi):
    if hsv is None or hsv.size == 0:
        return 0.0
    lo_arr = np.array(lo, dtype=np.uint8)
    hi_arr = np.array(hi, dtype=np.uint8)
    mask = cv2.inRange(hsv, lo_arr, hi_arr)
    return float(mask.sum()) / 255 / max(hsv.shape[0] * hsv.shape[1], 1)


def main():
    if len(sys.argv) < 3:
        print("Usage: python identify_goalie_team.py <video_path> <tracks.json>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    tracks_path = Path(sys.argv[2])
    label = pc.label_from_tracks_path(tracks_path)

    qframes = pc.get_quarter_frames(GAME_CONFIG, label)
    colors = pc.get_team_colors(GAME_CONFIG, label)
    tracks = load_tracks(tracks_path)
    print(f'Loaded {len(tracks):,} frames from {tracks_path.name}')

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'\n\u26a0 Could not open video file: {video_path}')
        print(f'  Check that this path is correct relative to where you are running the script.')
        sys.exit(1)
    results = {}

    for q, (qs, qe) in qframes.items():
        goalie_frames = []
        for fi, dets in tracks.items():
            if qs <= fi <= qe:
                for d in dets:
                    if d.get('team') == 'Goalie' and not d.get('interpolated'):
                        goalie_frames.append((fi, d))
                        break
        if not goalie_frames:
            print(f'  Q{q}: no goalie detections found')
            results[q] = None
            continue

        goalie_frames.sort(key=lambda x: x[0])
        step = max(1, len(goalie_frames) // SAMPLES_PER_QUARTER)
        sample = goalie_frames[::step][:SAMPLES_PER_QUARTER]

        scores = {TEAM_A: 0.0, TEAM_B: 0.0}
        n_checked = 0
        for fi, d in sample:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            hsv = get_jersey_hsv(frame, (d['l'], d['t'], d['r'], d['b']))
            for team, (lo, hi) in colors.items():
                scores[team] += score_team(hsv, lo, hi)
            n_checked += 1

        if n_checked == 0:
            print(f'  Q{q}: found {len(goalie_frames)} goalie detections, but could not read '
                  f'ANY of the {len(sample)} sampled frames from the video.')
            results[q] = None
            continue
        dominant = max(scores, key=scores.get)
        gap = abs(scores[TEAM_A] - scores[TEAM_B])
        print(f"  Q{q}: checked {n_checked} goalie frames -> "
              f"{TEAM_A} score={scores[TEAM_A]:.2f}  {TEAM_B} score={scores[TEAM_B]:.2f}  "
              f"(gap={gap:.2f})  => goalie's team (defends this net): {dominant}")
        results[q] = {TEAM_A: round(scores[TEAM_A], 2), TEAM_B: round(scores[TEAM_B], 2),
                      'dominant': dominant, 'gap': round(gap, 2)}

    cap.release()

    out_path = Path(f'goalie_team_{label}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n\u2713 Saved {out_path}')


if __name__ == '__main__':
    main()
