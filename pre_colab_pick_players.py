"""
pre_colab_pick_players.py
Click as many players as you want per team, across different frames, before
saving - unlike the older version of this tool, which only ever captured
ONE click per team total (the root cause of team-color classification being
too narrow: you can't build a reliable HSV range from a single sample).

Aim for 8+ clicks per team, deliberately including some in shadowed spots,
some far from the camera, and spread across different quarters - not just
easy, well-lit, close-up players. That variety is what
pre_colab_compute_team_colors.py needs to build a range that actually holds
up across the whole game.

Run locally:
    python pre_colab_pick_players.py

Output:
    <camera label>.player_clicks.json per camera, in the game's folder.
"""
import sys
import cv2
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline_common as pc

pc.pin_working_directory()
GAME_CONFIG = pc.load_game_config()
TEAM_A, TEAM_B = pc.get_teams(GAME_CONFIG)

DISP_W, DISP_H = 1280, 720


def pick_for_video(camera_label, video_path):
    out_path = Path(f'{camera_label}.player_clicks.json')

    if not Path(video_path).exists():
        print(f'  [{camera_label}] Video not found: {video_path}')
        return

    existing_clicks = []
    if out_path.exists():
        with open(out_path) as f:
            existing_clicks = json.load(f).get('clicks', [])
        print(f'  [{camera_label}] {out_path.name} already has {len(existing_clicks)} click(s) - '
              f'loading them, new clicks will be ADDED on top. Delete the file first for a clean redo.')

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = ow / DISP_W, oh / DISP_H

    state = {'fi': min(15000, total - 1), 'clicks': list(existing_clicks), 'team': TEAM_A}
    cache = {}

    def get_frame(fi):
        fi = max(0, min(fi, total - 1))
        if fi not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, f = cap.read()
            if ret:
                cache[fi] = f
        return cache.get(fi, np.zeros((oh, ow, 3), np.uint8))

    def counts():
        na = sum(1 for c in state['clicks'] if c['team'] == TEAM_A)
        nb = sum(1 for c in state['clicks'] if c['team'] == TEAM_B)
        return na, nb

    def draw():
        img = cv2.resize(get_frame(state['fi']), (DISP_W, DISP_H)).copy()
        ts = state['fi'] / fps
        m, s = int(ts // 60), ts % 60

        for c in state['clicks']:
            cx, cy = int(c['x'] / sx), int(c['y'] / sy)
            color = (0, 140, 255) if c['team'] == TEAM_A else (180, 105, 255)
            cv2.circle(img, (cx, cy), 10, color, -1)
            cv2.circle(img, (cx, cy), 10, (255, 255, 255), 2)

        na, nb = counts()
        cv2.rectangle(img, (0, DISP_H - 90), (DISP_W, DISP_H), (0, 0, 0), -1)
        cv2.putText(img, f"Current team: {state['team']}  (1={TEAM_A}  2={TEAM_B})   "
                          f"Clicked so far: {TEAM_A}={na}  {TEAM_B}={nb}",
                    (10, DISP_H - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
        cv2.putText(img, "Click players (any number, any frame). A/D=+-1  ,/.=+-30  "
                          "U=undo last  R=reset all  S=save+next",
                    (10, DISP_H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1)
        cv2.putText(img, f'Frame {state["fi"]} | {m:02d}:{s:05.2f} | {camera_label}',
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return img

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ox, oy = int(x * sx), int(y * sy)
        state['clicks'].append({'team': state['team'], 'x': ox, 'y': oy, 'frame': state['fi']})
        print(f"  \u2713 {state['team']} player clicked at ({ox}, {oy}) on frame {state['fi']}")

    print(f'\n{"="*60}\n  {camera_label}\n{"="*60}')
    print(f'  Click as many {TEAM_A}/{TEAM_B} players as you want, across as many frames as you want.')
    print(f'  Press 1/2 to switch which team you\'re clicking for. Aim for 8+ per team,')
    print(f'  spread across quarters/lighting/distance. S=save and move to next video, Q=skip.\n')

    WIN = f'PickPlayers_{camera_label}'
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    saved = False

    while True:
        cv2.imshow(WIN, draw())
        k = cv2.waitKey(20) & 0xFF
        if k in (ord('a'), ord('A')):
            state['fi'] = max(0, state['fi'] - 1)
        elif k in (ord('d'), ord('D')):
            state['fi'] = min(total - 1, state['fi'] + 1)
        elif k == ord(','):
            state['fi'] = max(0, state['fi'] - 30)
        elif k == ord('.'):
            state['fi'] = min(total - 1, state['fi'] + 30)
        elif k == ord('1'):
            state['team'] = TEAM_A
        elif k == ord('2'):
            state['team'] = TEAM_B
        elif k in (ord('u'), ord('U')):
            if state['clicks']:
                removed = state['clicks'].pop()
                print(f"  Undid {removed['team']} click at ({removed['x']}, {removed['y']})")
        elif k in (ord('r'), ord('R')):
            state['clicks'].clear()
            print('  Reset all clicks.')
        elif k in (ord('s'), ord('S')):
            na, nb = counts()
            if na == 0 or nb == 0:
                print(f'  Need at least 1 click for each team first ({TEAM_A}={na}, {TEAM_B}={nb})')
            else:
                cv2.destroyWindow(WIN)
                saved = True
                break
        elif k in (ord('q'), ord('Q')):
            cv2.destroyWindow(WIN)
            break

    cap.release()

    if saved:
        na, nb = counts()
        with open(out_path, 'w') as f:
            json.dump({'label': camera_label, 'team_a': TEAM_A, 'team_b': TEAM_B,
                       'clicks': state['clicks']}, f, indent=2)
        print(f'  \u2713 Saved {na} {TEAM_A} + {nb} {TEAM_B} clicks \u2192 {out_path}')
        if na < 8 or nb < 8:
            print(f'  \u26a0 Consider adding more before recomputing team_colors - '
                  f'8+ per team spread across different lighting/distance is recommended.')
    else:
        print(f'  Skipped {camera_label} (no file saved)')


def main():
    print('PLAYER CLICK PICKER')
    print('Click as many players per team as you want, across different frames/lighting.\n')

    for cam in pc.get_cameras(GAME_CONFIG):
        pick_for_video(cam['label'], cam['video'])

    print('\nDone. Now run pre_colab_compute_team_colors.py on each camera\'s clicks file.')


if __name__ == '__main__':
    main()
