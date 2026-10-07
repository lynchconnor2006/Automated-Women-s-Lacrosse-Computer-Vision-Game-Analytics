"""
visualize_possession.py
For one specific possession, prints the full per-0.5s time series of
offense/defense/deep-offense counts (so you can see exactly when each
threshold holds and where they fail to overlap), and saves a handful of
annotated frame images pulled directly from the video, so you can actually
see what's happening on the field instead of continuing to guess at
threshold numbers.

Usage:
    python visualize_possession.py <video_path> <tracks.json> <quarter> <team> <start_sec> <end_sec> [n_samples]

Example (the Q1 UNC case where all three peaks were individually satisfied
but never sustained together):
    python visualize_possession.py "SYR UNC LR Side.mp4" "SYR UNC LR Side.tracks.json" 1 UNC 272.1 335.5
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
DEEP_ZONES = ('Key', 'Behind Net')
OFFENSE_MIN_FRAME = 6
DEFENSE_MIN_FRAME = 6
DEEP_OFFENSE_MIN = 3


def load_tracks(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def main():
    if len(sys.argv) < 7:
        print("Usage: python visualize_possession.py <video> <tracks.json> <quarter> <team> "
              "<start_sec> <end_sec> [n_samples]")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    tracks_path = Path(sys.argv[2])
    q = int(sys.argv[3])
    team = sys.argv[4]
    start_sec = float(sys.argv[5])
    end_sec = float(sys.argv[6])
    n_samples = int(sys.argv[7]) if len(sys.argv) > 7 else 10

    label = pc.label_from_tracks_path(tracks_path)
    qframes = pc.get_quarter_frames(GAME_CONFIG, label)
    qs, qe = qframes[q]
    tracks = load_tracks(tracks_path)
    def_team = other_team(team)

    # Pre-index sampled frames within this quarter's range for faster lookup
    frame_times = sorted(fi for fi in tracks if qs <= fi <= qe)

    def nearest_frame(target_t, max_dist=0.3):
        best_fi, best_dist = None, max_dist
        for fi in frame_times:
            dist = abs((fi - qs) / FPS - target_t)
            if dist < best_dist:
                best_dist = dist
                best_fi = fi
        return best_fi

    print(f"Time series for Q{q} {team} {start_sec}s-{end_sec}s on {label}:")
    print(f"(thresholds: offense>={OFFENSE_MIN_FRAME}  defense>={DEFENSE_MIN_FRAME}  "
          f"deep_offense>={DEEP_OFFENSE_MIN})\n")
    print(f"{'t':>7} {'off':>4} {'def':>4} {'deep':>5}  near_full?")

    t = start_sec
    step = 0.5
    while t <= end_sec:
        fi = nearest_frame(t)
        off = deff = deep = 0
        if fi is not None:
            for d in tracks[fi]:
                zone = d.get('zone')
                if zone not in PLAY_ZONES:
                    continue
                if d.get('team') == team:
                    off += 1
                    if zone in DEEP_ZONES:
                        deep += 1
                elif d.get('team') == def_team or d.get('team') == 'Goalie':
                    deff += 1
        near_full = off >= OFFENSE_MIN_FRAME and deff >= DEFENSE_MIN_FRAME and deep >= DEEP_OFFENSE_MIN
        marker = '  <== YES' if near_full else ''
        print(f"{t:7.1f} {off:4d} {deff:4d} {deep:5d}{marker}")
        t += step

    # Save sample annotated frames spread across the window
    out_dir = video_path.parent / f"possession_review_Q{q}_{team}_{int(start_sec)}"
    out_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"\n⚠ Could not open video: {video_path}")
        sys.exit(1)

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sample_times = [start_sec + i * (end_sec - start_sec) / max(n_samples - 1, 1)
                    for i in range(n_samples)]
    saved = 0
    for i, ts in enumerate(sample_times):
        fi = nearest_frame(ts, max_dist=1.0)
        if fi is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
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
        cv2.putText(vis, f"F{fi}  t={ts:.1f}s", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out_path = out_dir / f"frame_{i:02d}_t{ts:.1f}.jpg"
        cv2.imwrite(str(out_path), vis)
        saved += 1

    cap.release()
    print(f"\n✓ Saved {saved} annotated sample frames to: {out_dir}")
    print("Open these and compare against the time series above — yellow boxes are the")
    print("possessing team, pink are the opponent, green is the goalie.")


if __name__ == '__main__':
    main()
