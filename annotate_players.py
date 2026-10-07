"""
annotate_players.py
Local interactive annotation tool for building new YOLO training data to
fine-tune lacrosse_best.pt. Draw boxes around players, referees, and
goalies on real game frames, save in standard YOLO training format.

WHY THIS EXISTS: the current detector has a real, confirmed raw-recall
gap - some players simply never get a candidate box at all, regardless of
downstream logic (color classification, tiling, tracking, possession
search all sit downstream of detection and can't recover a player who was
never detected). Fixing that requires new labeled training examples,
specifically covering the failure modes already identified this session
(near field logos/text, distant/small players, crowded scrambles).

CLASSES (matches the existing model's own class numbering - cls 0 and
cls 2 are already used throughout the pipeline; cls 1 has always been
silently skipped, strongly suggesting it already exists in the base
model's schema, most likely as referee):
    0 = Player
    1 = Referee
    2 = Goalie

CONTROLS:
    Left-click + drag  : draw a box (assigned to the CURRENT class)
    0 / 1 / 2           : switch current class to Player / Referee / Goalie
    U                   : undo the last box drawn on this frame
    C                   : clear all boxes on this frame
    S or SPACE          : save this frame's annotations, advance to next
    N                   : skip this frame (nothing worth annotating), advance
    B                   : go back to the previous frame (re-editable)
    Q or ESC            : quit - progress so far is already saved

OUTPUT (in --out, default 'annotation_data/'):
    images/<label>_F<frame>.jpg          - original frame, full resolution
    labels/<label>_F<frame>.txt          - YOLO format: class cx cy w h (normalized)
    previews/<label>_F<frame>_annotated.jpg - boxes overlaid, for quick review
    classes.txt                          - class names, one per line, in order

Usage:
    python annotate_players.py "SYR UNC MVC Side.mp4" --game "SYR_UNC_20260213" --n 40
    python annotate_players.py "SYR UNC LR Side.mp4" --game "SYR_UNC_20260213" --n 40 --seed 7
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as pc

CLASS_NAMES = {0: 'Player', 1: 'Referee', 2: 'Goalie'}
CLASS_COLORS = {0: (0, 255, 255), 1: (0, 165, 255), 2: (0, 255, 0)}  # BGR
DISPLAY_MAX_WIDTH = 1600  # scale display down for large source frames;
                          # saved coordinates are always converted back to
                          # full original resolution


class AnnotationState:
    def __init__(self, frame_height):
        self.boxes = []          # list of [class_id, x1, y1, x2, y2] in DISPLAY-space pixels
        self.current_class = 0
        self.drawing = False
        self.start_point = None
        self.live_end = None
        self.frame_height = frame_height  # clicks below this (in the banner strip) are ignored


def mouse_callback(event, x, y, flags, state: AnnotationState):
    if event == cv2.EVENT_LBUTTONDOWN:
        if y >= state.frame_height:
            return  # click started in the banner strip - ignore
        state.drawing = True
        state.start_point = (x, y)
        state.live_end = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
        state.live_end = (x, min(y, state.frame_height - 1))
    elif event == cv2.EVENT_LBUTTONUP:
        if not state.drawing:
            return
        state.drawing = False
        y = min(y, state.frame_height - 1)  # a drag that ends in the banner strip
                                             # still completes normally, clamped to
                                             # the image's bottom edge
        if state.start_point is not None:
            x1, y1 = state.start_point
            x2, y2 = x, y
            if abs(x2 - x1) > 3 and abs(y2 - y1) > 3:
                state.boxes.append([state.current_class, min(x1, x2), min(y1, y2),
                                     max(x1, x2), max(y1, y2)])
        state.start_point = None
        state.live_end = None


BANNER_HEIGHT = 40  # separate strip BELOW the frame - never covers any part
                     # of the actual image


def render(display_frame, state: AnnotationState, saved_count, target):
    vis = display_frame.copy()
    for cls_id, x1, y1, x2, y2 in state.boxes:
        color = CLASS_COLORS[cls_id]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, CLASS_NAMES[cls_id], (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if state.drawing and state.start_point and state.live_end:
        color = CLASS_COLORS[state.current_class]
        cv2.rectangle(vis, state.start_point, state.live_end, color, 1)

    banner = np.full((BANNER_HEIGHT, vis.shape[1], 3), 30, dtype=np.uint8)
    banner_color = CLASS_COLORS[state.current_class]
    cv2.putText(banner, f'Class: {CLASS_NAMES[state.current_class]} (0/1/2 to change)  |  '
                         f'{len(state.boxes)} box(es)  |  saved {saved_count}/{target}  |  '
                         f'S=save+next  N=skip  U=undo  C=clear  B=back  Q=quit',
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, banner_color, 1)
    return np.vstack([vis, banner])


def build_frame_pool(video_path, qframes, seed, target_frames=None):
    """A shuffled pool of candidate frames - much larger than any realistic
    target count. The caller advances through it until enough frames have
    been SAVED (not just shown), so skips don't shrink the final count
    below the target.

    If target_frames is given (a specific list of frame indices, e.g. from
    find_obscure_frames.py), the pool is built from THOSE frames instead of
    randomly sampling every in-quarter frame - for targeted annotation of
    a specific known-problem scenario rather than a broad random sweep."""
    cap = cv2.VideoCapture(str(video_path))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    rng = random.Random(seed)
    if target_frames is not None:
        pool = list(target_frames)
    else:
        pool = []
        for q, (s, e) in qframes.items():
            pool.extend(range(s, e))
    rng.shuffle(pool)
    return pool, fw, fh


def save_annotations(out_dir, label, fi, orig_frame, boxes_display, scale, fw, fh):
    images_dir = out_dir / 'images'
    labels_dir = out_dir / 'labels'
    previews_dir = out_dir / 'previews'
    for d in (images_dir, labels_dir, previews_dir):
        d.mkdir(parents=True, exist_ok=True)

    stem = f'{label}_F{fi}'
    img_path = images_dir / f'{stem}.jpg'
    lbl_path = labels_dir / f'{stem}.txt'
    prev_path = previews_dir / f'{stem}_annotated.jpg'

    cv2.imwrite(str(img_path), orig_frame)

    lines = []
    preview = orig_frame.copy()
    for cls_id, x1, y1, x2, y2 in boxes_display:
        ox1, oy1, ox2, oy2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
        cx = (ox1 + ox2) / 2 / fw
        cy = (oy1 + oy2) / 2 / fh
        w = (ox2 - ox1) / fw
        h = (oy2 - oy1) / fh
        lines.append(f'{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
        color = CLASS_COLORS[cls_id]
        cv2.rectangle(preview, (int(ox1), int(oy1)), (int(ox2), int(oy2)), color, 3)
        cv2.putText(preview, CLASS_NAMES[cls_id], (int(ox1), max(int(oy1) - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    with open(lbl_path, 'w') as f:
        f.write('\n'.join(lines))
    cv2.imwrite(str(prev_path), preview)


def pin_to_game_folder(game_name):
    """Standalone tools (run directly, not via run_pipeline.py) need to
    find the game folder themselves. Mirrors the same relative-path
    convention used everywhere else in this project."""
    base_dir = Path(__file__).resolve().parent.parent
    game_dir = base_dir / 'games' / game_name
    if not game_dir.exists():
        print(f'\u26a0 Game folder not found: {game_dir}')
        sys.exit(1)
    os.chdir(game_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video', help='Path to camera video file, e.g. "SYR UNC MVC Side.mp4"')
    ap.add_argument('--game', required=True, help='Game folder name, e.g. SYR_UNC_20260213')
    ap.add_argument('--out', default='annotation_data', help='Output directory')
    ap.add_argument('--n', type=int, default=40, help='Target number of frames to SAVE this '
                                                        'session (skips don\'t count toward this)')
    ap.add_argument('--seed', type=int, default=0, help='Random seed for frame sampling')
    ap.add_argument('--target-frames', default=None,
                     help='Path to a JSON file (from find_obscure_frames.py) with a specific '
                          'list of frames to sample from, instead of randomly sampling every '
                          'in-quarter frame - for targeted annotation of a known scenario')
    args = ap.parse_args()

    pin_to_game_folder(args.game)
    config = pc.load_game_config()
    label = pc.label_from_tracks_path(Path(args.video))
    qframes = pc.get_quarter_frames(config, label)

    target_frames = None
    if args.target_frames:
        with open(args.target_frames) as f:
            target_frames = json.load(f)['frames']
        print(f'Using {len(target_frames)} targeted frame(s) from {args.target_frames} '
              f'instead of random in-quarter sampling.')

    pool, fw, fh = build_frame_pool(args.video, qframes, args.seed, target_frames=target_frames)
    scale = min(1.0, DISPLAY_MAX_WIDTH / fw)
    disp_w, disp_h = int(fw * scale), int(fh * scale)

    out_dir = Path(args.out)
    classes_path = out_dir / 'classes.txt'
    if not classes_path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(classes_path, 'w') as f:
            f.write('\n'.join(CLASS_NAMES[i] for i in sorted(CLASS_NAMES)))

    cap = cv2.VideoCapture(args.video)
    window = 'Annotate - ' + label
    cv2.namedWindow(window)
    state = AnnotationState(frame_height=disp_h)
    cv2.setMouseCallback(window, mouse_callback, state)

    target = args.n
    print(f'Target: save {target} frame(s) from {label} (pool of {len(pool)} candidates - '
          f'skips don\'t count against the target).')
    print('Controls: draw boxes with click-drag | 0/1/2=class | S/SPACE=save+next | '
          'N=skip | U=undo | C=clear | B=back | Q=quit\n')

    decisions = {}  # pool index -> 'saved' or 'skipped', so B(ack) + changing your
                     # mind adjusts saved_count correctly instead of double-counting
    i = 0
    saved_count = 0

    while saved_count < target:
        if i < 0:
            i = 0
        if i >= len(pool):
            print(f'\n\u26a0 Ran out of candidate frames before reaching the target of '
                  f'{target} - only {saved_count} saved.')
            break

        fi = pool[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            i += 1
            continue

        display_frame = cv2.resize(frame, (disp_w, disp_h))
        state.boxes = []
        state.current_class = 0

        advance = None
        while advance is None:
            vis = render(display_frame, state, saved_count, target)
            cv2.imshow(window, vis)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord('0'), ord('1'), ord('2')):
                state.current_class = key - ord('0')
            elif key == ord('u'):
                if state.boxes:
                    state.boxes.pop()
            elif key == ord('c'):
                state.boxes = []
            elif key in (ord('s'), ord(' ')):
                new_decision = 'saved' if state.boxes else 'skipped'
                if new_decision == 'saved':
                    save_annotations(out_dir, label, fi, frame, state.boxes, scale, fw, fh)
                    print(f'  F{fi}: saved with {len(state.boxes)} box(es)')
                else:
                    print(f'  F{fi}: no boxes drawn - treated as skip')
                advance = ('decide', new_decision)
            elif key == ord('n'):
                print(f'  F{fi}: skipped')
                advance = ('decide', 'skipped')
            elif key == ord('b'):
                advance = ('nav', -1)
            elif key in (ord('q'), 27):
                advance = ('quit', None)

        kind, value = advance
        if kind == 'quit':
            break
        if kind == 'decide':
            old = decisions.get(i)
            if old == 'saved' and value != 'saved':
                saved_count -= 1
            elif old != 'saved' and value == 'saved':
                saved_count += 1
            decisions[i] = value
            i += 1
        elif kind == 'nav':
            i += value

    cap.release()
    cv2.destroyAllWindows()
    print(f'\n\u2713 Session complete: {saved_count} frame(s) saved to {out_dir}/')
    print(f'  images/  labels/  previews/  classes.txt')


if __name__ == '__main__':
    main()
