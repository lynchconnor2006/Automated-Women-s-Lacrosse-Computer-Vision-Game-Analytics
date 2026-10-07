"""
verify_cut.py
Pulls a short sequence of frames around ONE specific detected cut event (as
recorded in cuts_analysis.json's 'cuts' list), with that exact track_id's
box highlighted in a distinct color (thick red), so you can directly watch
whether that player genuinely crosses between Perimeter and Net-to-8 around
that frame — a direct check on one specific claimed cut, using the SAME
full-width zone definition detect_cuts.py actually uses (not the narrower
stored zone that visualize_possession.py shows).

Usage:
    python verify_cut.py <video_path> <tracks.json> <quarter> <track_id> <cut_frame> [window_sec]

Find quarter/track_id/cut_frame values by looking at the 'cuts' list inside
cuts_analysis.json for the possession you want to check.
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


def zone_from_point(cx, cy, camera):
    return pc.zone_from_point(cx, cy, camera, GAME_CONFIG)


def load_tracks(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def nearest_frame(tracks, qs, qe, target_f, max_dist=10):
    best, best_d = None, max_dist
    for fi in tracks:
        if qs <= fi <= qe and abs(fi - target_f) < best_d:
            best_d = abs(fi - target_f)
            best = fi
    return best


def main():
    if len(sys.argv) < 6:
        print("Usage: python verify_cut.py <video> <tracks.json> <quarter> <track_id> "
              "<cut_frame> [window_sec]")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    tracks_path = Path(sys.argv[2])
    q = int(sys.argv[3])
    target_tid = int(sys.argv[4])
    cut_frame = int(sys.argv[5])
    window_sec = float(sys.argv[6]) if len(sys.argv) > 6 else 4.0

    label = pc.label_from_tracks_path(tracks_path)
    qs, qe = pc.get_quarter_frames(GAME_CONFIG, label)[q]
    tracks = load_tracks(tracks_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠ Could not open {video_path}")
        sys.exit(1)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = Path(f"cut_verify_Q{q}_track{target_tid}_F{cut_frame}")
    out_dir.mkdir(exist_ok=True)

    window_frames = int(window_sec * FPS)
    f_lo = max(qs, cut_frame - window_frames)
    f_hi = min(qe, cut_frame + window_frames)

    print(f"Verifying track_id={target_tid} around frame {cut_frame} "
          f"(window {f_lo}-{f_hi}), Q{q}, {label}\n")
    print(f"{'frame':>8}  {'this_track_zone':>16}")

    saved = 0
    candidate_frames = sorted(fi for fi in tracks if f_lo <= fi <= f_hi)
    step = max(1, len(candidate_frames) // 12)
    nearest_to_cut = nearest_frame(tracks, qs, qe, cut_frame)
    sample_frames = sorted(set(candidate_frames[::step] + ([nearest_to_cut] if nearest_to_cut else [])))

    for fi in sample_frames:
        if fi is None:
            continue
        dets = tracks.get(fi, [])
        target_det = next((d for d in dets if d.get('track_id') == target_tid), None)
        this_zone = zone_from_point(target_det['cx'], target_det['cy'], label) if target_det else "NOT PRESENT"
        marker = "  <== CUT RECORDED HERE" if abs(fi - cut_frame) < 5 else ""
        print(f"{fi:8d}  {this_zone:>16}{marker}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        vis = cv2.resize(frame, (1280, 720))
        dsx, dsy = fw / 1280, fh / 720
        for d in dets:
            l, t_, r, b = int(d['l'] / dsx), int(d['t'] / dsy), int(d['r'] / dsx), int(d['b'] / dsy)
            is_target = d.get('track_id') == target_tid
            zone = zone_from_point(d['cx'], d['cy'], label)
            col = (0, 0, 255) if is_target else (160, 160, 160)
            thickness = 4 if is_target else 1
            cv2.rectangle(vis, (l, t_), (r, b), col, thickness)
            if is_target:
                cv2.putText(vis, f"TARGET:{zone}", (l, max(t_ - 8, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.putText(vis, f"F{fi}  (cut recorded at F{cut_frame})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / f"F{fi}.jpg"), vis)
        saved += 1

    cap.release()
    print(f"\n✓ Saved {saved} frames to {out_dir}/")
    print("The red box is the specific player this cut was recorded for. Watch it move")
    print("across the frames — does it genuinely cross the net-to-8 line around the")
    print("marked frame, or is this a false positive (e.g. a different real player,")
    print("or noise near the zone boundary)?")


if __name__ == '__main__':
    main()
