"""
render_best_possession_video.py
Renders a FULL annotated video (not just sample frames) of one possession -
by default, whichever possession has the best data quality (highest
fraction of frames actually hitting the real 7-vs-8 roster count on BOTH
offense and defense), so the annotations shown are as accurate as this
pipeline gets. Can also render a specific possession by quarter/team/start.

Usage:
    python render_best_possession_video.py formation_analysis.json
        (auto-picks the best-quality possession)

    python render_best_possession_video.py formation_analysis.json <quarter> <team> <start_sec>
        (renders that specific possession instead)

Output:
    possession_video_Q<q>_<team>_<start>.mp4
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

OUT_W, OUT_H = 1280, 720


def other_team(t):
    return pc.make_other_team(GAME_CONFIG)(t)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def video_path_for_camera(camera_label):
    for cam in pc.get_cameras(GAME_CONFIG):
        if cam['label'] == camera_label:
            return cam['video']
    raise KeyError(f"No camera config for '{camera_label}'")


def load_tracks(camera_label):
    tracks_path = f"{camera_label}.tracks.json"
    with open(tracks_path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def pick_best_possession(formations):
    best, best_score = None, -1
    for f in formations:
        off_frac = f['n_frames_offense_qualifying'] / max(f['n_frames_total'], 1)
        def_frac = f['n_frames_defense_qualifying'] / max(f['n_frames_total'], 1)
        score = min(off_frac, def_frac)  # want BOTH sides reliable, not just one
        if score > best_score:
            best_score, best = score, f
    return best, best_score


def draw_frame(frame, dets, team, def_team, fw, fh):
    vis = cv2.resize(frame, (OUT_W, OUT_H))
    dsx, dsy = fw / OUT_W, fh / OUT_H
    for d in dets:
        l, t_, r, b = int(d['l'] / dsx), int(d['t'] / dsy), int(d['r'] / dsx), int(d['b'] / dsy)
        dteam = d.get('team')
        col = (0, 255, 255) if dteam == team else \
              (255, 105, 180) if dteam == def_team else \
              (0, 255, 0) if dteam == 'Goalie' else (160, 160, 160)
        cv2.rectangle(vis, (l, t_), (r, b), col, 2)
        zone_tag = (d.get('zone') or '?')[0]
        cv2.putText(vis, f"{(dteam or '?')[0]}{zone_tag}", (l, max(t_ - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    return vis


def main():
    if len(sys.argv) < 2:
        print("Usage: python render_best_possession_video.py formation_analysis.json "
              "[quarter team start_sec]")
        sys.exit(1)

    formations = load_json(sys.argv[1])

    if len(sys.argv) >= 5:
        q, team, start_sec = int(sys.argv[2]), sys.argv[3], float(sys.argv[4])
        chosen = next((f for f in formations
                       if f['quarter'] == q and f['team'] == team and f['start_sec'] == start_sec), None)
        if chosen is None:
            print(f"No possession found matching Q{q} {team} start={start_sec}s")
            sys.exit(1)
        score = None
    else:
        chosen, score = pick_best_possession(formations)
        if chosen is None:
            print("No possessions found in formation_analysis.json")
            sys.exit(1)

    q, team, camera = chosen['quarter'], chosen['team'], chosen['camera']
    start_sec, end_sec = chosen['start_sec'], chosen['end_sec']
    def_team = other_team(team)

    if score is not None:
        print(f"Best-quality possession: Q{q} {team} {start_sec}s-{end_sec}s "
              f"[{camera}]  (quality score {score:.2f})")
    else:
        print(f"Rendering Q{q} {team} {start_sec}s-{end_sec}s [{camera}]")
    print(f"  offense qualifying: {chosen['n_frames_offense_qualifying']}/{chosen['n_frames_total']}  "
          f"defense qualifying: {chosen['n_frames_defense_qualifying']}/{chosen['n_frames_total']}")

    qframes = pc.get_quarter_frames(GAME_CONFIG, camera)
    qs, qe = qframes[q]
    start_f = qs + int(start_sec * FPS)
    end_f = qs + int(end_sec * FPS)

    video_path = video_path_for_camera(camera)
    tracks = load_tracks(camera)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        sys.exit(1)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(f"possession_video_Q{q}_{team}_{int(start_sec)}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, (OUT_W, OUT_H))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    n_written = 0
    last_dets = []
    for fi in range(start_f, end_f + 1):
        ret, frame = cap.read()
        if not ret:
            break
        # tracks.json only has entries every YOLO_STRIDE frames - reuse the
        # most recent real detections for frames in between, so the video
        # doesn't flicker boxes on/off between sampled frames
        if fi in tracks:
            last_dets = tracks[fi]
        vis = draw_frame(frame, last_dets, team, def_team, fw, fh)
        cv2.putText(vis, f"Q{q} {team} vs {def_team}  t={((fi - qs) / FPS):.1f}s",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(vis)
        n_written += 1

    cap.release()
    writer.release()
    print(f"\n\u2713 Wrote {n_written} frames to {out_path}")


if __name__ == '__main__':
    main()
