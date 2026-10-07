"""
pick_exclusion_zones.py
Run this LOCALLY (not in Colab) to define static false-positive exclusion
zones on a lacrosse video — e.g. a logo/cutout on the back of the net that
YOLO misdetects as a player.

Usage:
    python pick_exclusion_zones.py "SYR UNC MVC Side.mp4"

Controls:
    n / p       - next / previous frame
    N / P       - jump forward / back 100 frames (find a clear view fast)
    left-click  - set corner 1, then click again to set corner 2 (defines a box)
    c           - clear the box currently being drawn
    a           - accept the current box, name it, and start a new one
    s           - save all accepted zones to <video_name>.exclude_zones.json
    q / ESC     - quit (prompts to save if you have unsaved zones)

Output JSON goes next to the video file. Upload it to the same Drive folder
as your calibration/quarter_frames/player_clicks files, then have Colab load
it (see the Colab-side loader snippet in your project notes).
"""

import sys
import json
import cv2
from pathlib import Path
import os

# --- auto-inserted: pin the working directory to Summer Attempt so every
# relative path (arguments AND hardcoded output filenames) resolves correctly
# regardless of where this script is invoked from ---
_ENV_GAME_DIR = os.environ.get('LACROSSE_GAME_DIR')
if _ENV_GAME_DIR:
    _BASE_DIR = Path(_ENV_GAME_DIR)
else:
    _BASE_DIR = Path(__file__).resolve().parent.parent
os.chdir(_BASE_DIR)


WINDOW_NAME = "Exclusion Zone Picker  (n/p=frame  a=accept box  s=save  q=quit)"
DISPLAY_MAX_W = 1400  # scale window down for on-screen display only


def main():
    if len(sys.argv) < 2:
        print("Usage: python pick_exclusion_zones.py <video_path>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"File not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {fw}x{fh}, {total_frames:,} frames")

    scale = min(1.0, DISPLAY_MAX_W / fw)
    disp_w, disp_h = int(fw * scale), int(fh * scale)

    fi = total_frames // 4  # start somewhere in gameplay, not black intro frames
    zones = []              # accepted zones: [{"name":..., "l":..,"t":..,"r":..,"b":...}]
    click_pts = []          # points for the box currently being drawn (in ORIGINAL coords)

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        return frame if ret else None

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # convert displayed-window coords back to ORIGINAL frame coords
            ox, oy = int(x / scale), int(y / scale)
            if len(click_pts) >= 2:
                click_pts.clear()
            click_pts.append((ox, oy))

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    frame = read_frame(fi)

    while True:
        vis = cv2.resize(frame, (disp_w, disp_h))

        # draw already-accepted zones in green
        for z in zones:
            l, t, r, b = [int(v * scale) for v in (z["l"], z["t"], z["r"], z["b"])]
            cv2.rectangle(vis, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(vis, z["name"], (l, max(t - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # draw the box currently being defined in yellow
        if len(click_pts) == 1:
            x, y = click_pts[0]
            cv2.circle(vis, (int(x * scale), int(y * scale)), 5, (0, 255, 255), -1)
        elif len(click_pts) == 2:
            (x1, y1), (x2, y2) = click_pts
            l, t = int(min(x1, x2) * scale), int(min(y1, y2) * scale)
            r, b = int(max(x1, x2) * scale), int(max(y1, y2) * scale)
            cv2.rectangle(vis, (l, t), (r, b), (0, 255, 255), 2)

        cv2.putText(vis, f"Frame {fi:,}/{total_frames:,}  |  zones saved: {len(zones)}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, vis)

        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), 27):
            if zones and not _confirm("Quit without saving remaining zones?"):
                continue
            break
        elif key == ord('n'):
            fi = min(fi + 1, total_frames - 1); frame = read_frame(fi)
        elif key == ord('p'):
            fi = max(fi - 1, 0); frame = read_frame(fi)
        elif key == ord('N'):
            fi = min(fi + 100, total_frames - 1); frame = read_frame(fi)
        elif key == ord('P'):
            fi = max(fi - 100, 0); frame = read_frame(fi)
        elif key == ord('c'):
            click_pts.clear()
        elif key == ord('a'):
            if len(click_pts) == 2:
                (x1, y1), (x2, y2) = click_pts
                l, t = min(x1, x2), min(y1, y2)
                r, b = max(x1, x2), max(y1, y2)
                name = input("Name this zone (e.g. 'net_logo'): ").strip() or f"zone_{len(zones)+1}"
                zones.append({"name": name, "l": l, "t": t, "r": r, "b": b})
                print(f"  ✓ Added '{name}': ({l},{t}) → ({r},{b})")
                click_pts.clear()
            else:
                print("  Click two corners first (top-left, then bottom-right).")
        elif key == ord('s'):
            if not zones:
                print("  No zones to save yet.")
            else:
                out_path = video_path.parent / f"{video_path.stem}.exclude_zones.json"
                with open(out_path, "w") as f:
                    json.dump({
                        "label": video_path.stem,
                        "reference_frame": fi,
                        "video_resolution": [fw, fh],
                        "zones": zones,
                    }, f, indent=2)
                print(f"  ✓ Saved {len(zones)} zone(s) to {out_path}")

    cap.release()
    cv2.destroyAllWindows()


def _confirm(msg):
    return input(f"{msg} (y/n): ").strip().lower().startswith("y")


if __name__ == "__main__":
    main()
