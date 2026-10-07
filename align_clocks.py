"""
align_clocks.py
Builds a per-quarter mapping from OFFICIAL game-clock elapsed time to VIDEO
elapsed time, and re-expresses official_possessions.json in video time.

WHY THIS IS NEEDED: the official game clock stops during dead-ball
situations; the video's elapsed time runs continuously through everything.
A single linear offset for the whole quarter is not accurate enough — this
builds a mapping from multiple anchor points spread through the quarter.

ANCHORS USED:
  1. Bookends: (official=0, video=0) and (official=period_length, video=quarter_duration).
  2. Official stoppage markers (Timeout, Free position, Card) matched
     against VIDEO-detected "quiet windows" (sustained near-net absence).
  3. Official GOAL events matched against VIDEO-detected "huddle windows" -
     right after a goal, players cluster tightly near the net (the scoring
     team celebrating, the scored-on team regrouping) - a visually distinct,
     learnable signature independent of how many whistles got logged that
     quarter. This is a genuinely different, usually more plentiful anchor
     source than stoppage markers, since every quarter has goals even when
     it has few timeouts/fouls.

Usage:
    python align_clocks.py <cam1.tracks.json> <cam2.tracks.json> \\
        official_possessions.json stoppage_markers.json goal_markers.json

Output:
    official_possessions_aligned.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as pc

pc.pin_working_directory()
GAME_CONFIG = pc.load_game_config()
FPS = pc.get_fps(GAME_CONFIG)
PERIOD_LENGTH_SEC = pc.get_period_length_sec(GAME_CONFIG)

PLAY_ZONES = ('Key', 'Behind Net')  # near-net zones, for detecting quiet AND huddle windows

TIME_STEP = 0.5
MIN_QUIET_SAMPLES = 6       # 6 * 0.5s = 3 seconds sustained — filters single-frame
                             # track dropouts; a real stoppage should show up as
                             # multiple consecutive quiet samples, not one blip
QUIET_COUNT_THRESHOLD = 1    # near-net count <= this (on whichever camera has
                              # the max) counts as "quiet" for that time bucket
ANCHOR_SEARCH_RADIUS_SEC = 90  # how far from the baseline linear guess to look
                                # for a matching video window (quiet or huddle)

HUDDLE_SPREAD_THRESHOLD = 3.0   # same box-width-relative spacing math as
                                # detect_and_match_possessions.py's is_huddled -
                                # a huddle looks the same regardless of which
                                # script is looking for it
MIN_PLAYERS_FOR_HUDDLE = 4
MIN_HUDDLE_SAMPLES = 3         # 1.5 seconds sustained. Reduced from 6 (3 sec) -
                               # players CONVERGE onto a goal celebration over
                               # a couple seconds rather than snapping
                               # instantly into a tight cluster, so the
                               # genuinely tight moment may not hold steady as
                               # long as a real dead-ball "quiet window" would.
HUDDLE_GRACE_SAMPLES = 4      # bridges brief gaps (~2 sec) of momentarily-not-
                              # quite-huddled frames within an otherwise real,
                              # forming huddle - players converging aren't
                              # perfectly tight frame-to-frame even during a
                              # real celebration, so one brief flicker below
                              # threshold shouldn't split one real huddle into
                              # two separate windows that each individually
                              # fail the duration requirement.
GOAL_HUDDLE_DELAY_SEC = 2.5    # players don't instantly cluster the moment a goal is
                               # scored - it takes a couple seconds for the celebration/
                               # regroup to actually form. Shifting the search guess
                               # later by this much avoids searching right at the
                               # goal's own timestamp when the real clustering hasn't
                               # happened yet.


def load_tracks(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def frame_to_quarter_time(fi, qframes):
    for q, (s, e) in qframes.items():
        if s <= fi <= e:
            return q, (fi - s) / FPS
    return None, None


def build_camera_timeline(tracks, qframes):
    """Per-quarter (t, near_net_count) pairs, for quiet-window detection."""
    timeline = {q: [] for q in qframes}
    for fi, dets in tracks.items():
        q, t = frame_to_quarter_time(fi, qframes)
        if q is None:
            continue
        count = sum(1 for d in dets if d.get('zone') in PLAY_ZONES)
        timeline[q].append((t, count))
    for q in timeline:
        timeline[q].sort()
    return timeline


def is_frame_huddled(dets):
    """True if players near the net (either team, goalie excluded) are
    clustered tightly enough to be a goal celebration/regroup rather than
    real spread play."""
    near_net = [d for d in dets if d.get('zone') in PLAY_ZONES and d.get('team') != 'Goalie']
    if len(near_net) < MIN_PLAYERS_FOR_HUDDLE:
        return False
    widths = [d['r'] - d['l'] for d in near_net]
    avg_width = sum(widths) / len(widths)
    if avg_width <= 0:
        return False
    pts = [(d['cx'], d['cy']) for d in near_net]
    dists = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dists.append(((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2) ** 0.5)
    avg_dist = sum(dists) / len(dists)
    return (avg_dist / avg_width) < HUDDLE_SPREAD_THRESHOLD


def build_huddle_timeline(tracks, qframes):
    """Per-quarter (t, is_huddled) pairs, for huddle-window detection."""
    timeline = {q: [] for q in qframes}
    for fi, dets in tracks.items():
        q, t = frame_to_quarter_time(fi, qframes)
        if q is None:
            continue
        timeline[q].append((t, is_frame_huddled(dets)))
    for q in timeline:
        timeline[q].sort()
    return timeline


def resample(timeline_pairs, step):
    if not timeline_pairs:
        return []
    max_t = timeline_pairs[-1][0]
    n_buckets = int(max_t // step) + 1
    out = [0] * n_buckets
    idx = 0
    for b in range(n_buckets):
        target = b * step
        while idx < len(timeline_pairs) - 1 and \
              abs(timeline_pairs[idx + 1][0] - target) <= abs(timeline_pairs[idx][0] - target):
            idx += 1
        out[b] = timeline_pairs[idx][1]
    return out


def find_quiet_windows(combined_counts, step, min_samples, threshold):
    """(start_sec, end_sec) windows where combined_counts stayed at or below
    `threshold` for at least `min_samples` CONSECUTIVE buckets."""
    windows = []
    n = len(combined_counts)
    i = 0
    while i < n:
        if combined_counts[i] <= threshold:
            j = i
            while j < n and combined_counts[j] <= threshold:
                j += 1
            if j - i >= min_samples:
                windows.append((i * step, j * step))
            i = j
        else:
            i += 1
    return windows


def find_true_windows(bool_list, step, min_samples):
    """(start_sec, end_sec) windows where bool_list stayed True for at
    least `min_samples` CONSECUTIVE buckets."""
    windows = []
    n = len(bool_list)
    i = 0
    while i < n:
        if bool_list[i]:
            j = i
            while j < n and bool_list[j]:
                j += 1
            if j - i >= min_samples:
                windows.append((i * step, j * step))
            i = j
        else:
            i += 1
    return windows


def build_quiet_windows_per_quarter(cam_timelines, quarters):
    quiet = {}
    for q in quarters:
        resampled = [resample(cam_timelines[i].get(q, []), TIME_STEP) for i in range(2)]
        max_len = max((len(r) for r in resampled), default=0)
        if max_len == 0:
            quiet[q] = []
            continue
        combined = [max((r[b] if b < len(r) else 0) for r in resampled) for b in range(max_len)]
        quiet[q] = find_quiet_windows(combined, TIME_STEP, MIN_QUIET_SAMPLES, QUIET_COUNT_THRESHOLD)
    return quiet


def build_huddle_windows_per_quarter(cam_huddle_timelines, quarters):
    """A real celebration huddle only needs to show up on ONE camera (the
    one whose net just got scored on), so combine via OR, not max-of-count."""
    huddle = {}
    for q in quarters:
        resampled = [resample(cam_huddle_timelines[i].get(q, []), TIME_STEP) for i in range(2)]
        max_len = max((len(r) for r in resampled), default=0)
        if max_len == 0:
            huddle[q] = []
            continue
        combined = [any((r[b] if b < len(r) else False) for r in resampled) for b in range(max_len)]
        huddle[q] = find_true_windows(combined, TIME_STEP, MIN_HUDDLE_SAMPLES)
    return huddle


BAD_ANCHOR_SCALE_RATIO = 1.6  # if BOTH segments touching an anchor deviate from the
                              # quarter's overall expected scale by more than this
                              # factor (in either direction), that anchor is likely
                              # itself wrong, not just a temporarily fast/slow patch
                              # of play. A single bad anchor corrupts BOTH the segment
                              # before it (artificially stretched or compressed) and
                              # the segment after it (compensating in the opposite
                              # direction to get back in sync with the next anchor) -
                              # that paired signature is what this specifically targets,
                              # rather than flagging any one unusually fast/slow stretch
                              # of real gameplay on its own.


def filter_bad_anchors(anchors, expected_scale, ratio_threshold=BAD_ANCHOR_SCALE_RATIO):
    """Drops interior anchor points (never the bookends) whose surrounding
    segments BOTH deviate strongly from the quarter's overall expected
    scale (video_duration / PERIOD_LENGTH_SEC) - the signature of one bad
    anchor corrupting two segments at once, rather than genuinely uneven
    game pace (which would only affect one segment, not both symmetrically)."""
    if len(anchors) <= 2:
        return anchors

    kept = list(anchors)
    changed = True
    while changed and len(kept) > 2:
        changed = False
        for i in range(1, len(kept) - 1):
            t_prev, v_prev = kept[i - 1]
            t_cur, v_cur = kept[i]
            t_next, v_next = kept[i + 1]
            if t_cur - t_prev <= 0 or t_next - t_cur <= 0:
                continue
            scale_before = (v_cur - v_prev) / (t_cur - t_prev)
            scale_after = (v_next - v_cur) / (t_next - t_cur)

            def deviates(s):
                return s > expected_scale * ratio_threshold or s < expected_scale / ratio_threshold

            if deviates(scale_before) and deviates(scale_after):
                print(f'    [diag] dropping anchor at official={t_cur:.1f}s '
                      f'(video={v_cur:.1f}s) - both surrounding segments deviate from '
                      f'the quarter\'s expected scale {expected_scale:.2f}x '
                      f'(before={scale_before:.2f}x, after={scale_after:.2f}x)')
                del kept[i]
                changed = True
                break
    return kept


def match_markers_to_windows(q, markers, windows, video_duration, label, verbose=False, marker_delay_sec=0.0):
    """Generic marker-to-window matching, shared by stoppage-vs-quiet and
    goal-vs-huddle anchor finding. Returns a list of (official_time,
    video_time) anchor pairs (bookends are added separately by the caller).

    marker_delay_sec: added to each marker's own elapsed_sec before
    computing its baseline guess - the official goal event fires the
    instant the goal is scored, but the real video clustering (celebration/
    regroup) takes a couple seconds to actually form. The returned anchor's
    OFFICIAL time is still the marker's true, unshifted elapsed_sec - only
    the SEARCH guess is shifted, so the anchor itself stays accurate."""
    q_markers = sorted([m for m in markers if m['quarter'] == q], key=lambda m: m['elapsed_sec'])

    if verbose:
        print(f'    [diag] Q{q} {label}: {len(q_markers)} marker(s) at '
              f'{[round(m["elapsed_sec"], 1) for m in q_markers]}')
        print(f'    [diag] Q{q} {label}: {len(windows)} window(s), first 5: '
              f'{[(round(a, 1), round(b, 1)) for a, b in windows[:5]]}')

    anchors = []
    if not q_markers or not windows:
        if verbose:
            print(f'    [diag] Q{q} {label}: no markers or no windows - nothing to match')
        return anchors

    def baseline_guess(t_off):
        return ((t_off + marker_delay_sec) / PERIOD_LENGTH_SEC) * video_duration

    used = set()
    for m in q_markers:
        guess = baseline_guess(m['elapsed_sec'])
        best_wi, best_mid, best_dist = None, None, float('inf')
        for wi, (ws, we) in enumerate(windows):
            if wi in used:
                continue
            mid = (ws + we) / 2
            dist = abs(mid - guess)
            if dist < best_dist:
                best_dist, best_wi, best_mid = dist, wi, mid
        matched = best_wi is not None and best_dist < ANCHOR_SEARCH_RADIUS_SEC
        if verbose:
            status = f'matched window {best_wi} at dist={best_dist:.1f}s' if matched else \
                      f'NO MATCH (closest dist={best_dist:.1f}s, radius={ANCHOR_SEARCH_RADIUS_SEC}s)'
            print(f'    [diag] Q{q} {label} marker@{m["elapsed_sec"]:.1f}s -> '
                  f'baseline guess {guess:.1f}s -> {status}')
        if matched:
            used.add(best_wi)
            anchors.append((m['elapsed_sec'], best_mid))

    return anchors


def official_to_video(t_off, anchors):
    """Piecewise-linear interpolation between sorted anchor points."""
    for i in range(len(anchors) - 1):
        t0, v0 = anchors[i]
        t1, v1 = anchors[i + 1]
        if t0 <= t_off <= t1:
            if t1 == t0:
                return v0
            frac = (t_off - t0) / (t1 - t0)
            return v0 + frac * (v1 - v0)
    return anchors[0][1] if t_off < anchors[0][0] else anchors[-1][1]


def main():
    if len(sys.argv) < 6:
        print('Usage: python align_clocks.py <cam1.tracks.json> <cam2.tracks.json> '
              '<official_possessions.json> <stoppage_markers.json> <goal_markers.json>')
        sys.exit(1)

    cam_paths = [Path(sys.argv[1]), Path(sys.argv[2])]
    official_path = Path(sys.argv[3])
    markers_path = Path(sys.argv[4])
    goal_markers_path = Path(sys.argv[5])

    labels = [pc.label_from_tracks_path(p) for p in cam_paths]
    qframes_by_cam = [pc.get_quarter_frames(GAME_CONFIG, lbl) for lbl in labels]
    # reference camera (for quarter list + "official" video-duration axis) = first arg
    ref_qframes = qframes_by_cam[0]

    cam_timelines = []
    cam_huddle_timelines = []
    for p, label, qframes in zip(cam_paths, labels, qframes_by_cam):
        tracks = load_tracks(p)
        cam_timelines.append(build_camera_timeline(tracks, qframes))
        cam_huddle_timelines.append(build_huddle_timeline(tracks, qframes))
        print(f'Loaded {label}: {len(tracks):,} frames')

    quarters = sorted(ref_qframes.keys())
    quiet = build_quiet_windows_per_quarter(cam_timelines, quarters)
    for q, windows in quiet.items():
        print(f'  Q{q}: {len(windows)} sustained quiet window(s) detected in video '
              f'(>= {MIN_QUIET_SAMPLES * TIME_STEP:.0f}s each)')

    huddle = build_huddle_windows_per_quarter(cam_huddle_timelines, quarters)
    for q, windows in huddle.items():
        print(f'  Q{q}: {len(windows)} sustained huddle window(s) detected in video '
              f'(>= {MIN_HUDDLE_SAMPLES * TIME_STEP:.0f}s each)')

    with open(markers_path) as f:
        stoppage_markers = json.load(f)
    with open(official_path) as f:
        official = json.load(f)
    with open(goal_markers_path) as f:
        goal_markers = json.load(f)

    aligned = []
    for q in quarters:
        s, e = ref_qframes[q]
        video_duration = (e - s) / FPS

        stoppage_anchors = match_markers_to_windows(
            q, stoppage_markers, quiet.get(q, []), video_duration, 'stoppage', verbose=True)
        goal_anchors = match_markers_to_windows(
            q, goal_markers, huddle.get(q, []), video_duration, 'goal-huddle',
            verbose=True, marker_delay_sec=GOAL_HUDDLE_DELAY_SEC)

        anchor_set = {(0.0, 0.0), (float(PERIOD_LENGTH_SEC), video_duration)}
        anchor_set.update(stoppage_anchors)
        anchor_set.update(goal_anchors)
        anchors = sorted(anchor_set)

        expected_scale = video_duration / PERIOD_LENGTH_SEC
        n_before_filter = len(anchors)
        anchors = filter_bad_anchors(anchors, expected_scale)
        if len(anchors) < n_before_filter:
            print(f'  Q{q}: dropped {n_before_filter - len(anchors)} anchor(s) whose '
                  f'surrounding segments both deviated from the expected {expected_scale:.2f}x '
                  f'pace (see [diag] lines above)')

        print(f'  Q{q}: using {len(anchors)} anchor point(s) for clock alignment '
              f'({len(stoppage_anchors)} stoppage + {len(goal_anchors)} goal-huddle, '
              f'video quarter duration ~{video_duration:.0f}s)')

        for p in official:
            if p['quarter'] != q:
                continue
            new_start = official_to_video(p['start_sec'], anchors)
            new_end = official_to_video(p['end_sec'], anchors)
            aligned.append({
                'quarter': q, 'team': p['team'],
                'start_sec': round(new_start, 1), 'end_sec': round(new_end, 1),
                'duration_sec': round(new_end - new_start, 1),
                'end_reason': p.get('end_reason', 'other'),
            })

    out_path = Path('official_possessions_aligned.json')
    with open(out_path, 'w') as f:
        json.dump(aligned, f, indent=2)
    print(f'\n\u2713 Saved {len(aligned)} aligned official possessions to {out_path}')


if __name__ == '__main__':
    main()
