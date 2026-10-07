"""
track_players.py
Step 4 — simple nearest-neighbor player tracking across frames.

Run this SEPARATELY for each camera's detections.json — tracking does not
require the other camera; each camera covers a different half of the field,
so its tracks are self-contained. The two cameras only get reconciled later,
at the formation/possession analysis stage, by aligning on game-time
(quarter + elapsed seconds) rather than raw frame number, since the two
recordings don't start at the same absolute frame.

No goalie relabeling is applied here — an earlier attempt to reclassify
ambiguous crease-area detections as "Goalie" was dropped: it risked mislabeling
real field players as the goalie, and goalie tracking accuracy specifically
isn't important for this project, so it's not worth that risk. Goalie is
tracked as its own team category whenever the raw detector labels it that
way (cls==2), and nothing else.

Usage:
    python track_players.py "SYR UNC MVC Side.detections.json"
    python track_players.py "SYR UNC LR Side.detections.json"

Output:
    <label>.tracks.json — same frame-keyed structure as the input, but every
    detection gains a 'track_id' field (or track_id: null for 'Unknown'-team
    detections, which are too ambiguous to safely link across frames).
"""

import json
import sys
import math
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


# Known camera specs — needed to scale the match-distance threshold, since
# MVC (3840x2160) and LR (1920x1080) have different pixel scales for the
# same physical distance on the field. Add new cameras here as needed.
CAMERA_SPECS = {
    'SYR UNC MVC Side': {'width': 3840, 'height': 2160, 'fps': 30.0},
    'SYR UNC LR Side':  {'width': 1920, 'height': 1080, 'fps': 30.0},
}
REF_WIDTH = 3840  # same reference used throughout the detection pipeline

# ── Tracking parameters (tuned at REF_WIDTH, scaled per-camera below) ──
# Match distance is NOT a fixed cap — it scales with how long it's been since
# the track was last seen. A fixed cap caused massive over-fragmentation:
# a player can easily move 200-300px over a ~1-second gap, but a constant
# 120px cap only allowed matches for gaps under ~13 frames even though tracks
# were allowed to survive gaps up to GAP_TOLERANCE_FRAMES (30) — so any track
# with a real gap longer than ~13 frames would fail to match and spawn a
# spurious new track instead, even though it was still "alive."
BASE_SPEED_PX_PER_FRAME_REF = 12   # generous upper bound for real player sprint
                                     # speed (px/frame, at REF_WIDTH) — allowed
                                     # match distance grows by this much per
                                     # elapsed real frame since last detection
MATCH_SLACK_REF       = 40          # extra flat slack for detection jitter/box noise
GAP_TOLERANCE_FRAMES  = 90          # real video frames a track can go unmatched
                                     # before being retired (~3 sec at 30fps).
                                     # Raised from 30 (~1 sec): a track deleted
                                     # from `active` can NEVER be recovered even
                                     # if the same real player reappears moments
                                     # later at a plausible position - a brand
                                     # new track_id gets spawned instead, and no
                                     # amount of downstream gap-interpolation can
                                     # rejoin two track_ids that were already
                                     # split apart here. Given real players can
                                     # go undetected for a bit over a second even
                                     # after all of today's detection fixes
                                     # (logos, distant positions), 30 frames was
                                     # very plausibly THE dominant cause of the
                                     # massive fragment counts seen throughout
                                     # this project.
MAX_MATCH_DIST_REF    = 500          # hard cap on allowed match distance (px, at
                                     # REF_WIDTH), regardless of elapsed frames.
                                     # Match distance already grows unboundedly
                                     # with elapsed time (BASE_SPEED_PX_PER_FRAME_REF
                                     # * elapsed) - raising GAP_TOLERANCE_FRAMES
                                     # means that growth now has 3x longer to
                                     # accumulate before a match is attempted,
                                     # which without a cap could let two
                                     # genuinely DIFFERENT players (teammates
                                     # crossing paths) get incorrectly linked
                                     # into one track. This caps the growth at a
                                     # value still generous enough for a real
                                     # sprint across a few seconds, without
                                     # licensing an implausible full-field jump.


def load_detections(path):
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def label_from_path(path):
    stem = Path(path).stem
    if stem.endswith('.detections'):
        stem = stem[: -len('.detections')]
    return stem


def track_camera(detections, linear_scale, gap_tolerance_frames):
    """
    detections: dict[frame_idx] -> list of detection dicts
                (must include l,t,r,b,cx,cy,team,...)
    Returns: (detections_with_track_id, track_summaries)
    """
    frames_sorted = sorted(detections.keys())

    # active tracks: track_id -> {'team', 'cx', 'cy', 'last_frame'}
    active = {}
    next_id = 1
    track_summaries = {}  # track_id -> {'team','start_frame','end_frame','frame_count'}
    out = {}

    for fi in frames_sorted:
        dets = detections[fi]
        out[fi] = []

        # Only track SYR/UNC/Goalie — 'Unknown'-team detections are too
        # ambiguous to safely link across frames by position alone (risk of
        # silently merging two different real players), so they pass through
        # untracked rather than being force-assigned to a track.
        trackable   = [d for d in dets if d['team'] in ('SYR', 'UNC', 'Goalie')]
        untrackable = [d for d in dets if d['team'] not in ('SYR', 'UNC', 'Goalie')]

        # Retire tracks that have gone unmatched too long, based on the
        # actual frame gap (not just "calls"), since detection frames are
        # sparse (every YOLO_STRIDE-th real frame).
        for tid in list(active.keys()):
            if fi - active[tid]['last_frame'] > gap_tolerance_frames:
                del active[tid]

        # Greedy nearest-neighbor matching, same-team only, closest pairs first.
        # Allowed match distance grows with how long it's been since the
        # track was last seen — a fixed radius here was the root cause of
        # the over-fragmentation bug (see comment above MAX params).
        candidates = []
        for di, d in enumerate(trackable):
            for tid, t in active.items():
                if t['team'] != d['team']:
                    continue
                elapsed = fi - t['last_frame']
                allowed_dist = min(
                    (BASE_SPEED_PX_PER_FRAME_REF * elapsed + MATCH_SLACK_REF) * linear_scale,
                    MAX_MATCH_DIST_REF * linear_scale,
                )
                dist = math.hypot(d['cx'] - t['cx'], d['cy'] - t['cy'])
                if dist <= allowed_dist:
                    candidates.append((dist, di, tid))
        candidates.sort(key=lambda c: c[0])

        matched_dets, matched_tracks, assignment = set(), set(), {}
        for dist, di, tid in candidates:
            if di in matched_dets or tid in matched_tracks:
                continue
            assignment[di] = tid
            matched_dets.add(di)
            matched_tracks.add(tid)

        for di, d in enumerate(trackable):
            if di in assignment:
                tid = assignment[di]
            else:
                tid = next_id
                next_id += 1
                track_summaries[tid] = {
                    'team': d['team'], 'start_frame': fi,
                    'end_frame': fi, 'frame_count': 0,
                }
            active[tid] = {'team': d['team'], 'cx': d['cx'], 'cy': d['cy'], 'last_frame': fi}
            track_summaries[tid]['end_frame'] = fi
            track_summaries[tid]['frame_count'] += 1
            d_out = dict(d)
            d_out['track_id'] = tid
            out[fi].append(d_out)

        for d in untrackable:
            d_out = dict(d)
            d_out['track_id'] = None
            out[fi].append(d_out)

    return out, track_summaries


def main():
    if len(sys.argv) < 2:
        print('Usage: python track_players.py <detections.json path>')
        sys.exit(1)

    path = Path(sys.argv[1])
    label = label_from_path(path)
    specs = CAMERA_SPECS.get(label)
    if specs is None:
        print(f'⚠ No known camera specs for label "{label}" — add it to CAMERA_SPECS.')
        print(f'  Known labels: {list(CAMERA_SPECS.keys())}')
        sys.exit(1)

    linear_scale = specs['width'] / REF_WIDTH
    print(f'Loading {path.name}  (camera specs: {specs}, linear_scale={linear_scale:.3f})')

    detections = load_detections(path)
    print(f'  {len(detections):,} frames loaded')

    out, summaries = track_camera(detections, linear_scale, GAP_TOLERANCE_FRAMES)

    out_path = path.parent / f'{label}.tracks.json'
    with open(out_path, 'w') as f:
        json.dump({str(k): v for k, v in out.items()}, f)
    print(f'✓ Saved tracked detections: {out_path.name}')

    by_team = {}
    for tid, s in summaries.items():
        by_team.setdefault(s['team'], []).append(s)

    print(f'\nTrack summary:')
    for team, tracks in by_team.items():
        durations = [s['frame_count'] for s in tracks]
        print(f'  {team}: {len(tracks)} tracks, '
              f'avg length {sum(durations)/len(durations):.0f} detections, '
              f'longest {max(durations)}')

    short_tracks = [tid for tid, s in summaries.items() if s['frame_count'] <= 2]
    print(f'\n  {len(short_tracks)} tracks are 2 detections long or shorter — '
          f'likely fragments (a real player briefly missed then re-detected as')
    print(f'  a "new" track) rather than distinct people. Expected and fine for')
    print(f'  now — formation/possession analysis in later steps can tolerate')
    print(f'  some fragmentation.')


if __name__ == '__main__':
    main()
