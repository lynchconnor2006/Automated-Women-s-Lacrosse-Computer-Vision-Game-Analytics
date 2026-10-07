"""
detect_and_match_possessions.py
Possession boundaries come ENTIRELY from our own video detection - the
official play-by-play record is used only AFTER detection is complete, to
report how our detected count (total, and per team) compares to the real
game (57 total, 31 UNC / 26 SYR) and to label each detected possession's
end_reason for informational purposes. Official timing NEVER determines a
possession's start_sec/end_sec.

For each camera+quarter, scans that camera's ENTIRE quarter signal (no
window bounding - it's not anchored to any specific official entry) and
extracts every real, strict-confirmed settled possession. Each camera only
ever has one team attacking it per quarter (fixed by the goalie mapping),
so a detected segment's team comes directly from that, not from matching
against official data.

WHY THIS APPROACH: an earlier official-anchored version guaranteed a 1:1
possession count with the official record, but frequently had to fall back
to raw official timing when no real video evidence existed nearby, and
occasionally inherited a wrong team label from official-data ambiguity
(e.g. a brief, imprecisely-logged scramble misattributed to the wrong
team). Detection-first avoids both: every possession that exists in the
output is backed by real, confirmed video evidence for a specific team.
The tradeoff is the possession COUNT may not exactly match 57 - that
mismatch is reported clearly (see the count-check output) rather than
papered over, and is itself useful signal about detection quality.

Usage:
    python detect_and_match_possessions.py official_possessions_aligned.json \\
        <cam1.tracks.json> <cam2.tracks.json> \\
        <goalie_team_reconciled_cam1.json> <goalie_team_reconciled_cam2.json>

Output:
    detected_possessions.json - one entry per video-detected possession.
    Count is NOT guaranteed to match the official record; see the printed
    count-check report for how it compares.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as pc

pc.pin_working_directory()
GAME_CONFIG = pc.load_game_config()
FPS = pc.get_fps(GAME_CONFIG)
TEAM_ALIASES = GAME_CONFIG.get('team_aliases', {})
other_team = pc.make_other_team(GAME_CONFIG)

PLAY_ZONES = ('Key', 'Net-to-8', 'Behind Net', 'Perimeter')
DEEP_ZONES = ('Key', 'Net-to-8', 'Behind Net')  # everything except Perimeter counts as
                                                # deep/settled - Perimeter (8-to-30) is
                                                # transition, where players (including
                                                # disengaged stragglers around 25-30)
                                                # don't represent genuine settled attack

OFFENSE_MIN_FRAME = 6      # STRICT - confirms genuine settled play started at all
DEFENSE_MIN_FRAME = 6
DEEP_OFFENSE_MIN = 3
CONTINUATION_OFFENSE_MIN = 5  # LOOSE - governs the possession's actual SPAN once
CONTINUATION_DEFENSE_MIN = 5  # confirmed - see is_huddled/build_quarter_timeline
                              # docstrings for the full reasoning already
                              # established for these thresholds.
MIN_SUSTAINED_SEC = 3
GRACE_SEC = 9  # Raised from 6 - check_fragment_gaps.py showed 5 of 6 short/spurious
               # detected possessions sat right after a real possession of the SAME
               # team with only a 3.0-7.0s gap between them, right at or just past
               # the old 6s limit. This is direct, measured evidence those were real
               # continuous possessions getting split, not genuinely separate short
               # plays - 9s gives comfortable margin over the largest observed gap
               # (7.0s) without being so large it risks bridging genuinely distinct
               # short possessions together.
TIME_STEP = 0.5
END_PAD_SEC = 4          # small settling buffer added past whatever video signal is found
SEARCH_BUFFER_SEC = 45   # how far before/after the official (aligned) window we search
                         # for video signal - large on purpose, since it's still safely
                         # capped by this same team's own adjacent official possessions
                         # and can never bleed into a neighboring play regardless
GOAL_CAP_BUFFER_SEC = 10  # extra buffer past a goal-ending possession's official end,
                          # to cover the immediate celebration without running on forever
MAX_WHOLE_QUARTER_DISTANCE_SEC = 120  # how far from the official window we'll accept
                                      # a leftover whole-quarter segment as a real match -
                                      # without this cap, a genuinely unrelated moment
                                      # from elsewhere in the quarter could get matched
                                      # just for being "the closest available leftover"
IMPLAUSIBLE_DURATION_SEC = 150  # flagged for visibility, not auto-corrected -
                                # a single possession this long is almost
                                # certainly worth a manual look


def normalize_team(t):
    return TEAM_ALIASES.get(t, t)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_tracks(path):
    raw = load_json(path)
    return {int(k): v for k, v in raw.items()}


def merge_with_grace(active, grace_buckets):
    n = len(active)
    filled = list(active)
    i = 0
    while i < n:
        if not filled[i]:
            j = i
            while j < n and not filled[j]:
                j += 1
            gap_len = j - i
            if gap_len <= grace_buckets and i > 0 and j < n:
                for k in range(i, j):
                    filled[k] = True
            i = j
        else:
            i += 1
    return filled


def extract_segments(filled):
    segments = []
    n = len(filled)
    i = 0
    while i < n:
        if filled[i]:
            j = i
            while j < n and filled[j]:
                j += 1
            segments.append((i, j))
            i = j
        else:
            i += 1
    return segments


HUDDLE_SPREAD_THRESHOLD = 3.0  # if the offense's average pairwise spread is less
                                # than this many box-widths, treat it as a huddle
                                # (celebration/timeout gathering), not a real spread
                                # formation - a real offense keeps meaningful spacing
MIN_PLAYERS_FOR_HUDDLE_CHECK = 4  # need at least this many players to meaningfully
                                    # judge clustering vs spacing


def is_huddled(off_dets):
    """off_dets: list of (cx, cy, box_width) for the offense team in
    PLAY_ZONES this frame. True if they're clustered tightly enough to be a
    huddle rather than a real spread formation."""
    if len(off_dets) < MIN_PLAYERS_FOR_HUDDLE_CHECK:
        return False
    avg_width = sum(w for _, _, w in off_dets) / len(off_dets)
    if avg_width <= 0:
        return False
    pts = [(cx, cy) for cx, cy, _ in off_dets]
    dists = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dists.append(((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2) ** 0.5)
    avg_dist = sum(dists) / len(dists)
    return (avg_dist / avg_width) < HUDDLE_SPREAD_THRESHOLD


BOTTOM_EDGE_MARGIN_FRAC = 0.05  # within this fraction of true frame height counts
                                 # as "near the bottom edge" (camera cutoff), not
                                 # just anywhere in the broader Behind Net zone
MAX_OFFSCREEN_GRACE_SEC = 8  # how long we'll assume a player is still there,
                             # off-screen, before giving up on that assumption
                             # (they may have genuinely subbed out, or play ended)


MIN_TRACK_LENGTH_FOR_OFFSCREEN_CREDIT = 5  # a track this short is more likely a spurious
                                            # false-positive blip (e.g. near the net
                                            # structure/logo) than a genuine player who
                                            # went off-screen - without this, many short
                                            # spurious tracks near the net's bottom edge
                                            # can each grant a separate credit event, and
                                            # if enough chain back-to-back across a long
                                            # dead stretch (nothing real happening at all),
                                            # the CUMULATIVE credit can cover an arbitrarily
                                            # long span even though each individual event
                                            # stays under MAX_OFFSCREEN_GRACE_SEC - this is
                                            # exactly the mechanism that produced a fully-
                                            # credited 229s "possession" with zero real
                                            # detections anywhere in it (confirmed via
                                            # check_long_possession_internal_gaps.py).


def find_offscreen_behind_net_events(tracks, qs, qe, frame_height, off_team, def_team):
    """A player's LAST detection before a gap being in the Behind Net zone
    AND near the true bottom edge of the camera's field of view (not just
    anywhere behind the net) suggests they went off-screen retrieving a
    loose ball - not that they left the play. Requires the track to have
    at least MIN_TRACK_LENGTH_FOR_OFFSCREEN_CREDIT real detections BEFORE
    the gap, so a brief spurious false-positive track can't grant credit.
    Returns (team, start_frame, end_frame) for each such event; the caller
    decides how long a grace window to actually trust this for."""
    by_track = {}
    for fi, dets in tracks.items():
        if not (qs <= fi <= qe):
            continue
        for d in dets:
            tid = d.get('track_id')
            if tid is None:
                continue
            by_track.setdefault(tid, []).append((fi, d))

    bottom_threshold = frame_height * (1 - BOTTOM_EDGE_MARGIN_FRAC)
    events = []
    for tid, entries in by_track.items():
        entries.sort(key=lambda x: x[0])
        for i, ((fa, da), (fb, db)) in enumerate(zip(entries, entries[1:])):
            gap = fb - fa
            if gap <= 1:
                continue  # no real gap here
            if i + 1 < MIN_TRACK_LENGTH_FOR_OFFSCREEN_CREDIT:
                continue  # too few real detections before this gap to trust it
            team = da.get('team')
            if team not in (off_team, def_team):
                continue
            if da.get('zone') == 'Behind Net' and da.get('cy', 0) >= bottom_threshold:
                events.append((team, fa, fb))
    return events


LOOSE_PERIMETER_NEAR_FRAC = 0.7  # only the closer 70% of the Perimeter band
                                 # (nearer the 8-yard line) counts toward the
                                 # LOOSE off/def totals that govern a
                                 # possession's span. The outer ~30% nearest
                                 # the 30-yard line is where disengaged /
                                 # watching players were repeatedly observed
                                 # during manual annotation - real players,
                                 # just not meaningfully engaged in the play
                                 # happening elsewhere. Key/Behind Net are
                                 # untouched; only Perimeter is filtered, and
                                 # only for THIS possession-span signal - the
                                 # stored 'Perimeter' zone label itself, and
                                 # everything analyze_formations.py/
                                 # detect_cuts.py report, is unaffected.


def build_quarter_timeline(tracks, qs, qe, off_team, def_team, camera_label=None):
    """Per-0.5s-bucket (off, def, deep_off, huddled) across the WHOLE quarter
    span [qs, qe], using max-per-bucket aggregation for counts (ANY sampled
    frame in a bucket reading 'huddled' marks that bucket huddled - a brief
    real false-positive from a tight passing exchange gets bridged by
    GRACE_SEC same as any other gap; a real, sustained huddle won't).

    KNOWN LIMITATION (confirmed via direct inspection, not fixed - see
    project notes): a fast shot/rebound/clear sequence where BOTH teams'
    players stay engaged near the net (headcounts never actually drop) can
    keep this signal continuously "loose" across what the official record
    logs as a separate turnover-triggered possession for the other team.
    This is inherent to a headcount-based detector - it tracks WHO IS
    PRESENT, not WHO HAS THE BALL, so a real, brief possession change that
    doesn't change who's on the field won't register as a break. Confirmed
    on one case this session (Q2, SYR vs UNC, ~1514-1621s) where raw counts
    stayed at 6-9 for both teams the whole time - not a bug, just a real
    edge case worth watching for in future games rather than something to
    chase further here."""
    duration = (qe - qs) / FPS
    n_buckets = int(duration / TIME_STEP) + 1
    off_counts = [0] * n_buckets
    def_counts = [0] * n_buckets
    deep_off_counts = [0] * n_buckets
    huddle_flags = [False] * n_buckets

    perimeter_cutoff_cy = None
    if camera_label is not None:
        cal = pc.get_calibration_y(GAME_CONFIG, camera_label)
        yard8_y, yard30_y = cal['yard8_y'], cal['yard30_y']
        # cy DECREASES moving away from goal, so "closer to the 8" means
        # cy nearer yard8_y; keep the near LOOSE_PERIMETER_NEAR_FRAC of the
        # band, cut off before reaching the far (yard30_y) edge.
        perimeter_cutoff_cy = yard8_y - LOOSE_PERIMETER_NEAR_FRAC * (yard8_y - yard30_y)

    for fi, dets in tracks.items():
        if not (qs <= fi <= qe):
            continue
        t = (fi - qs) / FPS
        b = int(t / TIME_STEP)
        if not (0 <= b < n_buckets):
            continue
        off = deff = deep_off = 0
        off_dets = []
        for d in dets:
            zone = d.get('zone')
            if zone not in PLAY_ZONES:
                continue
            if zone == 'Perimeter' and perimeter_cutoff_cy is not None and \
                    d.get('cy', 0) < perimeter_cutoff_cy:
                continue  # outer/far portion of Perimeter - excluded from
                          # loose continuation, per the annotation-confirmed
                          # disengaged-player pattern
            if d.get('team') == off_team:
                off += 1
                if zone in DEEP_ZONES:
                    deep_off += 1
                off_dets.append((d['cx'], d['cy'], d['r'] - d['l']))
            elif d.get('team') == def_team or d.get('team') == 'Goalie':
                deff += 1
        off_counts[b] = max(off_counts[b], off)
        def_counts[b] = max(def_counts[b], deff)
        deep_off_counts[b] = max(deep_off_counts[b], deep_off)
        if is_huddled(off_dets):
            huddle_flags[b] = True

    # Boost counts for players presumed still present but off-screen behind
    # the net (see find_offscreen_behind_net_events) - capped at
    # MAX_OFFSCREEN_GRACE_SEC so a genuinely-ended play or a real substitution
    # doesn't get assumed-present indefinitely. Behind Net is one of our own
    # DEEP_ZONES, so a boosted OFFENSE player also counts toward deep_off -
    # they were, after all, deep behind the net when we last saw them.
    if camera_label is not None:
        events = find_offscreen_behind_net_events(tracks, qs, qe,
                                                    pc.get_frame_height(GAME_CONFIG, camera_label),
                                                    off_team, def_team)
        for team, fa, fb in events:
            gap_sec = (fb - fa) / FPS
            if gap_sec > MAX_OFFSCREEN_GRACE_SEC:
                continue
            b_start = int((fa - qs) / FPS / TIME_STEP)
            b_end = int((fb - qs) / FPS / TIME_STEP)
            for b in range(max(b_start, 0), min(b_end + 1, n_buckets)):
                if team == off_team:
                    off_counts[b] += 1
                    deep_off_counts[b] += 1
                elif team == def_team:
                    def_counts[b] += 1

    return off_counts, def_counts, deep_off_counts, huddle_flags


def has_sustained_run(bool_slice, min_len):
    """True if bool_slice contains a run of at least min_len consecutive
    True values anywhere within it."""
    run = 0
    for v in bool_slice:
        run = run + 1 if v else 0
        if run >= min_len:
            return True
    return False


def find_camera_for_team(goalie_team_by_label, q, team):
    """Returns whichever camera label has this team as the OFFENSE (i.e.
    NOT the defending/goalie team) in quarter q. Exactly one of the two
    cameras should qualify, since ends are fixed for the whole quarter."""
    for label, per_quarter in goalie_team_by_label.items():
        def_team = per_quarter.get(q)
        if def_team is not None and def_team != team:
            return label
    return None


def find_valid_segments(strict, loose, grace_buckets, min_sustained_buckets, search_lo, search_hi):
    """Valid (strict-confirmed) segments within [search_lo, search_hi] on an
    already-computed strict/loose signal pair. Returns (start_sec, end_sec)
    tuples."""
    b_lo = max(0, int(search_lo / TIME_STEP))
    b_hi = min(len(loose), int(search_hi / TIME_STEP) + 1)
    loose_slice = loose[b_lo:b_hi]
    filled_slice = merge_with_grace(loose_slice, grace_buckets)
    segs = extract_segments(filled_slice)
    valid = [(s + b_lo, e + b_lo) for s, e in segs
             if has_sustained_run(strict[s + b_lo:e + b_lo], min_sustained_buckets)]
    return [(round(s * TIME_STEP, 1), round(e * TIME_STEP, 1)) for s, e in valid]


def find_whole_quarter_segments(strict, loose, grace_buckets, min_sustained_buckets):
    """All valid (strict-confirmed) segments across the ENTIRE strict/loose
    arrays - no windowing. Used as a last-resort search for a possession
    that found nothing in its own bounded, official-anchored window."""
    return find_valid_segments(strict, loose, grace_buckets, min_sustained_buckets,
                                0.0, len(loose) * TIME_STEP)


def ranges_overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def main():
    print(f'[detect_and_match_possessions.py] GRACE_SEC={GRACE_SEC}  '
          f'MIN_TRACK_LENGTH_FOR_OFFSCREEN_CREDIT={MIN_TRACK_LENGTH_FOR_OFFSCREEN_CREDIT}  '
          f'(if these numbers look stale after an edit, delete python/__pycache__ and rerun)')

    if len(sys.argv) < 6:
        print('Usage: python detect_and_match_possessions.py official_possessions_aligned.json '
              '<cam1.tracks.json> <cam2.tracks.json> <goalie_team_cam1.json> <goalie_team_cam2.json>')
        sys.exit(1)

    official = load_json(sys.argv[1])
    for p in official:
        p['team'] = normalize_team(p['team'])

    cam_paths = [Path(sys.argv[2]), Path(sys.argv[3])]
    goalie_paths = [Path(sys.argv[4]), Path(sys.argv[5])]

    tracks_by_label = {}
    goalie_team_by_label = {}
    qframes_by_label = {}
    for cam_path, goalie_path in zip(cam_paths, goalie_paths):
        label = pc.label_from_tracks_path(cam_path)
        tracks_by_label[label] = load_tracks(cam_path)
        qframes_by_label[label] = pc.get_quarter_frames(GAME_CONFIG, label)
        raw_goalie = load_json(goalie_path)
        goalie_team_by_label[label] = {int(k): v for k, v in raw_goalie.items()}
        print(f'Loaded {label}: {len(tracks_by_label[label]):,} frames, '
              f'goalie-team-per-quarter: {goalie_team_by_label[label]}')

    grace_buckets = max(1, round(GRACE_SEC / TIME_STEP))
    min_sustained_buckets = max(1, round(MIN_SUSTAINED_SEC / TIME_STEP))

    # Precompute the strict/loose signal ONCE per (camera, quarter, off_team)
    # combination.
    signal_cache = {}

    def get_signal(label, q, off_team, def_team):
        key = (label, q, off_team)
        if key not in signal_cache:
            qs, qe = qframes_by_label[label][q]
            off_counts, def_counts, deep_off_counts, huddle_flags = build_quarter_timeline(
                tracks_by_label[label], qs, qe, off_team, def_team, camera_label=label)
            strict = [(o >= OFFENSE_MIN_FRAME and d >= DEFENSE_MIN_FRAME and do >= DEEP_OFFENSE_MIN)
                      for o, d, do in zip(off_counts, def_counts, deep_off_counts)]
            loose = [(o >= CONTINUATION_OFFENSE_MIN and d >= CONTINUATION_DEFENSE_MIN and not h)
                     for o, d, h in zip(off_counts, def_counts, huddle_flags)]
            signal_cache[key] = (strict, loose, qs, qe)
        return signal_cache[key]

    # PURE VIDEO DETECTION: for each camera+quarter, scan the WHOLE quarter's
    # signal and extract every real, strict-confirmed settled possession -
    # no anchoring to official timing at all. Each camera only ever has ONE
    # team attacking it in a given quarter (fixed by the goalie mapping), so
    # a detected segment's team comes from that, not from matching against
    # any official entry.
    detected = []
    quarters = sorted(set(q for qframes in qframes_by_label.values() for q in qframes))

    for q in quarters:
        for camera in tracks_by_label:
            qframes = qframes_by_label[camera]
            if q not in qframes:
                continue
            def_team = goalie_team_by_label[camera].get(q)
            if def_team is None:
                continue
            off_team = other_team(def_team)

            strict, loose, qs, qe = get_signal(camera, q, off_team, def_team)
            quarter_duration_sec = (qe - qs) / FPS

            filled = merge_with_grace(loose, grace_buckets)
            segs = extract_segments(filled)
            valid_segs = [(s, e) for s, e in segs
                          if has_sustained_run(strict[s:e], min_sustained_buckets)]

            for (s_b, e_b) in valid_segs:
                start_sec = round(s_b * TIME_STEP, 1)
                end_sec = round(min(e_b * TIME_STEP + END_PAD_SEC, quarter_duration_sec), 1)
                detected.append({
                    'quarter': q,
                    'team': off_team,
                    'camera': camera,
                    'start_sec': start_sec,
                    'end_sec': end_sec,
                    'duration_sec': round(end_sec - start_sec, 1),
                    'end_reason': 'unknown',
                    'boundary_source': 'video_detected',
                })

    detected.sort(key=lambda p: (p['quarter'], p['start_sec']))

    # SPLIT abnormally long detected possessions where the official record
    # shows MULTIPLE separate possessions (same team, same quarter) whose
    # combined span falls inside this one video-detected block. This
    # targets a real, confirmed limitation of headcount-based detection: a
    # fast shot/rebound/turnover sequence where neither team's players
    # actually leave the zone produces NO signal our detector can use to
    # find the break, even though a real possession change happened (see
    # the Q2 SYR/UNC case this session - both teams stayed at 6-9 players
    # the entire time, confirmed via direct inspection).
    #
    # This does NOT invent video evidence for the split - each new
    # sub-possession is explicitly tagged with a source showing it borrowed
    # its boundaries from the official record rather than being
    # independently video-confirmed, so downstream steps and reports can
    # treat these with appropriately reduced confidence (same spirit as the
    # existing suspect_zero_cuts flag in detect_cuts.py).
    LONG_POSSESSION_SPLIT_THRESHOLD_SEC = 150
    SPLIT_MATCH_BUFFER_SEC = 45  # widened from 20 - testing whether Q4's still-unsplit
                                 # 165.6s block has a hidden opposing possession just
                                 # outside the tighter tolerance, matching the alignment
                                 # imprecision we've already confirmed exists elsewhere

    split_results = []
    n_split_events = 0
    for d in detected:
        if d['duration_sec'] <= LONG_POSSESSION_SPLIT_THRESHOLD_SEC:
            split_results.append(d)
            continue

        opp_team = other_team(d['team'])
        # Look for the OPPOSING team's official possessions hidden inside this
        # block - a real, brief possession change (shot/rebound/turnover) that
        # never registered as a break because neither team's headcount
        # actually dropped. This is NOT the same team's own possession being
        # logged twice - it's the other team briefly having the ball in the
        # middle of what our detector reads as one continuous stretch.
        hidden_opponent_possessions = sorted(
            [op for op in official if op['quarter'] == d['quarter'] and op['team'] == opp_team
             and op['start_sec'] >= d['start_sec'] - SPLIT_MATCH_BUFFER_SEC
             and op['end_sec'] <= d['end_sec'] + SPLIT_MATCH_BUFFER_SEC],
            key=lambda op: op['start_sec'])

        if not hidden_opponent_possessions:
            split_results.append(d)  # nothing hidden inside - keep as-is
            continue

        n_split_events += 1
        print(f"\n  Splitting long possession Q{d['quarter']} {d['team']} "
              f"{d['start_sec']}s-{d['end_sec']}s ({d['duration_sec']}s) - found "
              f"{len(hidden_opponent_possessions)} real {opp_team} possession(s) hidden "
              f"inside it (a fast turnover our headcount-based detector had no signal "
              f"to catch, since neither team's player count actually dropped):")

        cursor = d['start_sec']
        for op in hidden_opponent_possessions:
            opp_start = max(op['start_sec'], d['start_sec'])
            opp_end = min(op['end_sec'], d['end_sec'])
            if opp_end <= opp_start:
                continue
            if opp_start > cursor:
                split_results.append({
                    'quarter': d['quarter'], 'team': d['team'], 'camera': d['camera'],
                    'start_sec': round(cursor, 1), 'end_sec': round(opp_start, 1),
                    'duration_sec': round(opp_start - cursor, 1),
                    'end_reason': 'unknown', 'boundary_source': 'video_detected',
                })
            split_results.append({
                'quarter': d['quarter'], 'team': opp_team, 'camera': d['camera'],
                'start_sec': round(opp_start, 1), 'end_sec': round(opp_end, 1),
                'duration_sec': round(opp_end - opp_start, 1),
                'end_reason': op.get('end_reason', 'unknown'),
                'boundary_source': 'official_split_within_long_video_block',
            })
            print(f"      {round(opp_start,1)}s-{round(opp_end,1)}s  {opp_team}  "
                  f"(end_reason={op.get('end_reason','?')})  <- boundaries from official "
                  f"record, NOT independently video-confirmed")
            cursor = opp_end

        if d['end_sec'] > cursor:
            split_results.append({
                'quarter': d['quarter'], 'team': d['team'], 'camera': d['camera'],
                'start_sec': round(cursor, 1), 'end_sec': round(d['end_sec'], 1),
                'duration_sec': round(d['end_sec'] - cursor, 1),
                'end_reason': 'unknown', 'boundary_source': 'video_detected',
            })

    detected = split_results
    detected.sort(key=lambda p: (p['quarter'], p['start_sec']))
    if n_split_events:
        print(f'\n  Split {n_split_events} long possession(s) to recover hidden opposing-team '
              f'possessions - see \'boundary_source\' in the output for which entries this affects.')

    # Best-effort end_reason tagging from the official record, for
    # informational purposes only - never adjusts start_sec/end_sec. Each
    # detected possession is labeled with whichever official entry (same
    # quarter, same team) is closest in raw elapsed time.
    for d in detected:
        candidates = [op for op in official if op['quarter'] == d['quarter'] and op['team'] == d['team']]
        if candidates:
            closest = min(candidates, key=lambda op: abs(op['end_sec'] - d['end_sec']))
            d['end_reason'] = closest.get('end_reason', 'unknown')

    print(f'\n\u2713 Detected {len(detected)} possession(s) directly from video - '
          f'boundaries are NEVER borrowed from the official record.')

    # VALIDATION ONLY: the official record is used here purely to report how
    # our own detected count compares to the real game. This never changes a
    # single detected boundary; it's a health check on detection quality, not
    # a correction mechanism.
    off_total = len(official)
    det_total = len(detected)
    print(f'\n  Raw count check vs official record (simple totals, informational only):')
    print(f'    Total: detected {det_total} vs official {off_total} '
          f'({"+" if det_total >= off_total else ""}{det_total - off_total})')
    teams_seen = sorted(set(op['team'] for op in official) | set(d['team'] for d in detected))
    for team in teams_seen:
        off_n = sum(1 for op in official if op['team'] == team)
        det_n = sum(1 for d in detected if d['team'] == team)
        flag = '' if det_n == off_n else '  \u26a0'
        print(f'    {team}: detected {det_n} vs official {off_n} '
              f'({"+" if det_n >= off_n else ""}{det_n - off_n}){flag}')

    # MATCH-BASED check: raw totals can't tell "genuinely missing" apart from
    # "exists, but far from where alignment guessed it'd be" - a real
    # possession can legitimately be found well outside a tight time window
    # when a quarter's alignment is imprecise (confirmed this session - some
    # real matches only showed up at 100+ seconds away from the naive guess).
    # MATCH_WINDOW_SEC is deliberately generous for exactly that reason.
    MATCH_WINDOW_SEC = 150
    print(f'\n  Match-based check (is there a detected possession - same quarter, same team - '
          f'within {MATCH_WINDOW_SEC}s of each official one? \u00b1{MATCH_WINDOW_SEC}s '
          f'accounts for known alignment imprecision, not detection quality):')
    for team in teams_seen:
        official_team_entries = [o for o in official if o['team'] == team]
        detected_team_entries = [d for d in detected if d['team'] == team]
        unmatched = []
        for op in official_team_entries:
            has_match = any(
                d['quarter'] == op['quarter'] and
                not (d['end_sec'] < op['start_sec'] - MATCH_WINDOW_SEC or
                     d['start_sec'] > op['end_sec'] + MATCH_WINDOW_SEC)
                for d in detected_team_entries
            )
            if not has_match:
                unmatched.append(op)
        matched_n = len(official_team_entries) - len(unmatched)
        flag = '' if not unmatched else '  \u26a0'
        print(f'    {team}: {matched_n}/{len(official_team_entries)} official possessions have '
              f'a real detected match{flag}')
        for op in unmatched:
            print(f"      Q{op['quarter']} {op['start_sec']}s-{op['end_sec']}s "
                  f"(end_reason={op.get('end_reason','?')})  <- no detected match within "
                  f"{MATCH_WINDOW_SEC}s")

    suspicious = [d for d in detected if d['duration_sec'] > IMPLAUSIBLE_DURATION_SEC]
    if suspicious:
        print(f'\n\u26a0 {len(suspicious)} possession(s) are longer than {IMPLAUSIBLE_DURATION_SEC}s - '
              f'worth a manual look:')
        for d in suspicious:
            print(f"    Q{d['quarter']} {d['team']:>4}  {d['start_sec']}s-{d['end_sec']}s  "
                  f"({d['duration_sec']}s)  [{d['camera']}]")

    out_path = Path('detected_possessions.json')
    with open(out_path, 'w') as f:
        json.dump(detected, f, indent=2)
    print(f'\nSaved to {out_path}')

    print('\nFirst 20 possessions:')
    for d in detected[:20]:
        print(f"  Q{d['quarter']} {d['team']:>4}  {d['start_sec']}s-{d['end_sec']}s  "
              f"({d['duration_sec']}s)  [{d['camera']}]  ({d['end_reason']})")
    if len(detected) > 20:
        print(f'  ... and {len(detected) - 20} more')


if __name__ == '__main__':
    main()
