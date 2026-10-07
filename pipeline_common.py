"""
pipeline_common.py
Shared per-game configuration loader and generic helpers used by every
script in the pipeline. Each game's folder must contain a game_config.json
describing its team names, cameras, quarter-frame boundaries, field
calibration, and jersey color ranges - everything that used to be
hardcoded per-game inside individual scripts. See create_game_config.py
for a template generator.
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def pin_working_directory():
    """Resolves and chdir's to the active game's folder, in this priority:
      1. a '--game NAME' flag anywhere in sys.argv (removed from argv after
         reading, so the script's own positional argument parsing is
         unaffected by it)
      2. the LACROSSE_GAME_DIR env var (set by run_pipeline.py's subprocess calls)
      3. falls back to Summer Attempt root (old default, for ad-hoc runs)
    Returns the resolved base dir."""
    game_name = None
    if '--game' in sys.argv:
        idx = sys.argv.index('--game')
        if idx + 1 < len(sys.argv):
            game_name = sys.argv[idx + 1]
            del sys.argv[idx:idx + 2]
        else:
            del sys.argv[idx:idx + 1]

    if game_name:
        base = Path(__file__).resolve().parent.parent / 'games' / game_name
    else:
        env_dir = os.environ.get('LACROSSE_GAME_DIR')
        base = Path(env_dir) if env_dir else Path(__file__).resolve().parent.parent
    os.chdir(base)
    return base


def load_game_config(path='game_config.json'):
    with open(path) as f:
        return json.load(f)


def get_teams(config):
    return config['team_a'], config['team_b']


def make_other_team(config):
    a, b = get_teams(config)
    def other_team(t):
        return b if t == a else a
    return other_team


def _find_camera(config, camera_label):
    for cam in config['cameras']:
        if cam['label'] == camera_label:
            return cam
    raise KeyError(f"No camera config found for label '{camera_label}'")


def get_quarter_frames(config, camera_label):
    cam = _find_camera(config, camera_label)
    return {int(q): tuple(v) for q, v in cam['quarter_frames'].items()}


def load_raw_calibration_points(camera_label):
    """Reads <camera_label>.calibration.json directly for the raw 8 field
    calibration points, saved verbatim by setup_only.py's calibration step
    (P1=Goal Line-Left, P2=Goal Line-Right, P3=Key Top-Left, P4=Key
    Top-Right, P5=8-Yard-Left, P6=8-Yard-Right, P7=30-Yard-Left,
    P8=30-Yard-Right). This is the single source of truth for field
    geometry - reading it fresh here means there's no separate reduced
    summary that can drift out of sync with it."""
    cal_path = Path(f'{camera_label}.calibration.json')
    with open(cal_path) as f:
        cal = json.load(f)
    return cal['points']


def get_calibration_y(config, camera_label):
    """Derives goal/8-yard/30-yard depth lines AND the raw key-top points
    (needed for the Key zone's real arc shape - see build_key_polygon)
    directly from calibration.json, not a pre-reduced summary."""
    points = load_raw_calibration_points(camera_label)
    p1, p2, p3, p4, p5, p6, p7, p8 = [tuple(p) for p in points[:8]]
    return {
        'goal_y': (p1[1] + p2[1]) / 2,
        'yard8_y': (p5[1] + p6[1]) / 2,
        'yard30_y': (p7[1] + p8[1]) / 2,
        'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4,
    }


def get_frame_height(config, camera_label):
    """Reads frame height directly from the camera's video file - avoids
    depending on calibration.json's presence/location, since the video
    itself is guaranteed to exist in the game folder (the whole pipeline
    already depends on it)."""
    import cv2
    cam = _find_camera(config, camera_label)
    cap = cv2.VideoCapture(cam['video'])
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return height


def get_team_colors(config, camera_label):
    cam = _find_camera(config, camera_label)
    return {team: (tuple(lo), tuple(hi)) for team, (lo, hi) in cam['team_colors'].items()}


def build_key_polygon(p1, p2, p3, p4, bulge_frac=0.3, n_arc_points=40):
    """The Key zone's real shape: straight line P1(crease-left) -> P3(key-top-left),
    a CURVED arc from P3 across to P4(key-top-right) that bulges AWAY from
    the goal, then straight P4 -> P2(crease-right), closing back to P1 along
    the crease/goal line. The arc uses the same quadratic-Bezier
    control-point-offset technique as the original field-line-drawing code:
    control point = key-top midpoint, pushed further from goal by
    bulge_frac * (distance between P3 and P4).

    Returns a numpy array of (x,y) points suitable for cv2.pointPolygonTest.
    """
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    p3 = np.array(p3, dtype=np.float64)
    p4 = np.array(p4, dtype=np.float64)

    goal_mid = (p1 + p2) / 2
    key_mid = (p3 + p4) / 2
    depth_vec = key_mid - goal_mid
    depth_len = max(np.linalg.norm(depth_vec), 1.0)
    key_span = max(np.linalg.norm(p4 - p3), 1.0)
    ctrl = key_mid + (depth_vec / depth_len) * key_span * bulge_frac

    arc_points = []
    for k in range(n_arc_points + 1):
        t = k / n_arc_points
        pt = (1 - t) ** 2 * p3 + 2 * (1 - t) * t * ctrl + t ** 2 * p4
        arc_points.append(pt)

    polygon = [p1] + arc_points + [p2]
    return np.array(polygon, dtype=np.float32)


def point_in_key(cx, cy, key_polygon):
    return cv2.pointPolygonTest(key_polygon, (float(cx), float(cy)), False) >= 0


def zone_from_point(cx, cy, camera_label, config):
    """Corrected zone classification using the real Key-arc geometry, not
    the old linear-narrowing approximation (which had the shape backwards -
    it shrank to a point at the key top instead of widening from the
    crease out to the key-top points).

    Requires calibration_y to include the raw P1-P4 points (goal/crease
    left+right, key-top left+right) - NOT just the reduced scalar y-values
    the old version relied on. See build_key_polygon's docstring for the
    exact shape being tested."""
    cal = get_calibration_y(config, camera_label)
    goal_y = cal['goal_y']

    if cy > goal_y:
        return 'Behind Net'

    key_polygon = build_key_polygon(cal['p1'], cal['p2'], cal['p3'], cal['p4'])
    if point_in_key(cx, cy, key_polygon):
        return 'Key'

    if cy >= cal['yard8_y']:
        return 'Net-to-8'
    if cy >= cal['yard30_y']:
        return 'Perimeter'
    return 'Beyond 30'


def zone_from_cy(cy, camera_label, config):
    """DEPRECATED - kept only so old callers don't immediately break, but
    this cy-only version CANNOT correctly classify Key vs Net-to-8, since
    that boundary is now a real 2D shape (needs cx too). Every caller
    should migrate to zone_from_point(cx, cy, ...) instead. This shim just
    falls back to the old flat-line behavior, which will misclassify
    points inside the key's now-wider top-arc region as Net-to-8."""
    import warnings
    warnings.warn('zone_from_cy is deprecated - the Key zone boundary now depends on '
                   'cx too (it is a real curved shape, not a flat y-cutoff). '
                   'Switch callers to zone_from_point(cx, cy, camera_label, config).',
                   DeprecationWarning, stacklevel=2)
    cal = get_calibration_y(config, camera_label)
    if cy > cal['goal_y']:
        return 'Behind Net'
    if cy >= cal['yard8_y']:
        return 'Net-to-8'
    if cy >= cal['yard30_y']:
        return 'Perimeter'
    return 'Beyond 30'


def get_fps(config):
    return config.get('fps', 30.0)


def get_period_length_sec(config):
    return config.get('period_length_sec', 900)


def get_cameras(config):
    return config['cameras']


def get_playbyplay_txt(config):
    return config['playbyplay_txt']


def get_trusted_camera(config):
    return config.get('trusted_camera', config['cameras'][0]['label'])


def label_from_tracks_path(path):
    stem = Path(path).stem
    return stem[: -len('.tracks')] if stem.endswith('.tracks') else stem
