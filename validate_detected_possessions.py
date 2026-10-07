"""
validate_detected_possessions.py
Pulls a stratified sample of detected_possessions.json entries — spread
across all 4 quarters, both cameras, and a mix of short/medium/long
durations — and saves a handful of annotated frames for each, so you can
do an honest validation pass rather than checking only a few cherry-picked
examples.

Usage:
    python validate_detected_possessions.py detected_possessions.json \\
        "SYR UNC MVC Side.mp4" "SYR UNC MVC Side.tracks.json" \\
        "SYR UNC LR Side.mp4" "SYR UNC LR Side.tracks.json" [per_quarter]

Output:
    validation_review/  — folder of annotated sample frames, a few per
    sampled possession, named so they sort together by quarter/team/time.
    Also prints a summary list of exactly what was sampled, so you can
    track which ones you've checked.
"""
import sys
import json
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as pc

pc.pin_working_directory()
GAME_CONFIG = pc.load_game_config()
FPS = pc.get_fps(GAME_CONFIG)
other_team = pc.make_other_team(GAME_CONFIG)

PLAY_ZONES = ('Key', 'Behind Net', 'Perimeter')


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_tracks(path):
    raw = load_json(path)
    return {int(k): v for k, v in raw.items()}


def stratified_sample(detected, per_quarter=5):
    """Spread picks across quarters and, within each quarter, across the
    range of durations (short/medium/long) rather than picking randomly or
    just the first N."""
    samples = []
    quarters = sorted(set(d['quarter'] for d in detected))
    for q in quarters:
        q_items = [d for d in detected if d['quarter'] == q]
        if len(q_items) <= per_quarter:
            samples.extend(q_items)
            continue
        q_sorted = sorted(q_items, key=lambda d: d['duration_sec'])
        n = len(q_sorted)
        picks_idx = sorted(set(min(int(i * n / per_quarter), n - 1) for i in range(per_quarter)))
        samples.extend(q_sorted[i] for i in picks_idx)
    return samples


def nearest_frame(tracks, qs, qe, target_f, max_dist=15):
    best, best_d = None, max_dist
    for fi in tracks:
        if qs <= fi <= qe and abs(fi - target_f) < best_d:
            best_d = abs(fi - target_f)
            best = fi
    return best


def save_frame(cap, tracks, qs, target_t, team, def_team, fw, fh, out_path, label_text):
    target_f = qs + int(target_t * FPS)
    fi = nearest_frame(tracks, qs, qs + 10**9, target_f, max_dist=30)
    if fi is None:
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret:
        return False
    vis = cv2.resize(frame, (1280, 720))
    dsx, dsy = fw / 1280, fh / 720
    for d in tracks[fi]:
        l, t_, r, b = int(d['l'] / dsx), int(d['t'] / dsy), int(d['r'] / dsx), int(d['b'] / dsy)
        dteam = d.get('team')
        col = (0, 255, 255) if dteam == team else \
              (255, 105, 180) if dteam == def_team else \
              (0, 255, 0) if dteam == 'Goalie' else (160, 160, 160)
        cv2.rectangle(vis, (l, t_), (r, b), col, 2)
        cv2.putText(vis, f"{(dteam or '?')[0]}{(d.get('zone') or '?')[0]}",
                    (l, max(t_ - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    cv2.putText(vis, label_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), vis)
    return True


def main():
    if len(sys.argv) < 6:
        print("Usage: python validate_detected_possessions.py detected_possessions.json "
              "<mvc.mp4> <mvc_tracks.json> <lr.mp4> <lr_tracks.json> [per_quarter]")
        sys.exit(1)

    detected_path = Path(sys.argv[1])
    video_paths = {}
    tracks_by_label = {}

    cam_args = [(sys.argv[2], sys.argv[3]), (sys.argv[4], sys.argv[5])]
    for video_arg, tracks_arg in cam_args:
        tracks_path = Path(tracks_arg)
        label = pc.label_from_tracks_path(tracks_path)
        video_paths[label] = Path(video_arg)
        tracks_by_label[label] = load_tracks(tracks_path)
        print(f'Loaded {label}: {len(tracks_by_label[label]):,} frames')

    per_quarter = int(sys.argv[6]) if len(sys.argv) > 6 else 5

    detected = load_json(detected_path)
    sample = stratified_sample(detected, per_quarter=per_quarter)
    print(f'\nSampled {len(sample)} of {len(detected)} detected possessions '
          f'(~{per_quarter} per quarter, spread across durations):\n')

    out_dir = Path('validation_review')
    out_dir.mkdir(exist_ok=True)

    caps = {}
    for label, path in video_paths.items():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f'⚠ Could not open {path}')
            sys.exit(1)
        caps[label] = cap

    for i, d in enumerate(sample):
        q, team, camera = d['quarter'], d['team'], d['camera']
        start_sec, end_sec, dur = d['start_sec'], d['end_sec'], d['duration_sec']
        def_team = other_team(team)
        qs, qe = pc.get_quarter_frames(GAME_CONFIG, camera)[q]
        cap = caps[camera]
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tracks = tracks_by_label[camera]

        cam_short = camera.replace(" ", "_")
        tag = f"Q{q}_{team}_{cam_short}_{int(start_sec)}"
        print(f"[{i+1}/{len(sample)}] Q{q} {team:>4}  {start_sec}s-{end_sec}s  "
              f"({dur}s)  [{camera}]  -> {tag}_*.jpg")

        sample_points = [start_sec, start_sec + (end_sec - start_sec) / 2, end_sec]
        for j, t in enumerate(sample_points):
            label_text = f"Q{q} {team} {camera} t={t:.1f}s ({j+1}/3)"
            out_path = out_dir / f"{tag}_{j}.jpg"
            save_frame(cap, tracks, qs, t, team, def_team, fw, fh, out_path, label_text)

    for cap in caps.values():
        cap.release()

    print(f'\n✓ Saved review frames to {out_dir}/')
    print('Review each (start/middle/end frame per possession) and tally roughly how many')
    print('look like genuine settled possessions vs. false positives — that answers whether')
    print('the whole-quarter-scan detector is actually solid.')


if __name__ == '__main__':
    main()
