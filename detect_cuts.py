"""
detect_cuts.py
Step 8 — cut detection. A cut = a tracked OFFENSIVE player's zone changing
between Perimeter and Net-to-8 (in either direction), sustained for a
minimum number of consecutive samples — this filters out a player standing
near the zone boundary and flickering back and forth, which is not a real
cut. Counted per possession, and aggregated per team across the whole game.

This is the first step in this pipeline that actually needs track_id
continuity — every earlier step (possession detection, formation counting)
only needed frame-by-frame counts, not persistent player identity. Given
known track fragmentation (discussed earlier in this project), a real cut
that happens to occur exactly when a track drops (occlusion) may be missed
rather than falsely counted — undercounting is the safer failure mode, and
it's reported here as a known limitation rather than silently ignored.

Usage:
    python detect_cuts.py detected_possessions.json \\
        "SYR UNC MVC Side.tracks.json" "SYR UNC LR Side.tracks.json"

Output:
    cuts_analysis.json — per-possession cut list and counts, plus a
    game-level summary per team.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as pc

pc.pin_working_directory()
GAME_CONFIG = pc.load_game_config()
FPS = pc.get_fps(GAME_CONFIG)

CUT_ZONES = ('Net-to-8', 'Perimeter')  # only these two zones are relevant to a "cut"
MIN_CUT_SAMPLES = 3  # a new zone must hold for at least this many consecutive
                     # sampled detections (~0.3s at stride-3/30fps) before it
                     # counts as a real cut rather than boundary flicker


def zone_from_point(cx, cy, camera):
    return pc.zone_from_point(cx, cy, camera, GAME_CONFIG)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_tracks(path):
    raw = load_json(path)
    return {int(k): v for k, v in raw.items()}


def build_track_sequences(tracks, qs, qe, team, f_lo, f_hi, camera):
    """Returns {track_id: [(frame, zone), ...]} for the given team's tracked
    detections within [f_lo, f_hi], restricted to CUT_ZONES (Behind Net /
    Beyond 30 readings are skipped as non-informative gaps, not resets)."""
    sequences = defaultdict(list)
    for fi in range(f_lo, f_hi + 1):
        dets = tracks.get(fi)
        if not dets:
            continue
        for d in dets:
            if d.get('team') != team:
                continue
            tid = d.get('track_id')
            if tid is None:
                continue
            zone = zone_from_point(d['cx'], d['cy'], camera)
            if zone not in CUT_ZONES:
                continue
            sequences[tid].append((fi, zone))
    return sequences


def find_cuts_in_sequence(seq):
    """seq: list of (frame, zone), already time-ordered. Returns list of
    {frame, direction} for confirmed cuts (sustained runs only)."""
    if not seq:
        return []

    # run-length encode
    runs = []
    cur_zone, run_start = seq[0][1], 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i][1] != cur_zone:
            runs.append({'zone': cur_zone, 'start_idx': run_start, 'end_idx': i - 1,
                        'length': i - run_start, 'start_frame': seq[run_start][0]})
            if i < len(seq):
                cur_zone, run_start = seq[i][1], i
    # keep only sustained runs
    kept = [r for r in runs if r['length'] >= MIN_CUT_SAMPLES]

    cuts = []
    for i in range(len(kept) - 1):
        a, b = kept[i], kept[i + 1]
        if a['zone'] == b['zone']:
            continue  # shouldn't happen after run-length encoding, but safe
        if a['zone'] == 'Perimeter' and b['zone'] == 'Net-to-8':
            direction = 'in'
        elif a['zone'] == 'Net-to-8' and b['zone'] == 'Perimeter':
            direction = 'out'
        else:
            continue
        cuts.append({'frame': b['start_frame'], 'direction': direction})
    return cuts


def main():
    if len(sys.argv) < 4:
        print("Usage: python detect_cuts.py detected_possessions.json "
              "<cam1.tracks.json> <cam2.tracks.json>")
        sys.exit(1)

    detected_path = Path(sys.argv[1])
    cam_paths = [Path(sys.argv[2]), Path(sys.argv[3])]

    tracks_by_label = {}
    for p in cam_paths:
        label = pc.label_from_tracks_path(p)
        tracks_by_label[label] = load_tracks(p)
        print(f"Loaded {label}: {len(tracks_by_label[label]):,} frames")

    detected = load_json(detected_path)
    results = []
    game_summary = defaultdict(lambda: {'cuts_in': 0, 'cuts_out': 0, 'possessions_with_cuts': 0})

    for d in detected:
        q, team, camera = d['quarter'], d['team'], d['camera']
        qs, qe = pc.get_quarter_frames(GAME_CONFIG, camera)[q]
        tracks = tracks_by_label[camera]
        f_lo = qs + int(d['start_sec'] * FPS)
        f_hi = qs + int(d['end_sec'] * FPS)

        sequences = build_track_sequences(tracks, qs, qe, team, f_lo, f_hi, camera)

        all_cuts = []
        for tid, seq in sequences.items():
            seq.sort(key=lambda x: x[0])
            cuts = find_cuts_in_sequence(seq)
            for c in cuts:
                c['track_id'] = tid
            all_cuts.extend(cuts)

        all_cuts.sort(key=lambda c: c['frame'])
        n_in = sum(1 for c in all_cuts if c['direction'] == 'in')
        n_out = sum(1 for c in all_cuts if c['direction'] == 'out')

        n_total_cuts = n_in + n_out
        results.append({
            **d,
            'cuts': all_cuts,
            'n_cuts_in': n_in,
            'n_cuts_out': n_out,
            'n_tracks_considered': len(sequences),
            'suspect_zero_cuts': n_total_cuts == 0,
        })

        game_summary[team]['cuts_in'] += n_in
        game_summary[team]['cuts_out'] += n_out
        if all_cuts:
            game_summary[team]['possessions_with_cuts'] += 1

    out_path = Path('cuts_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Analyzed cuts for {len(results)} possessions, saved to {out_path}")

    print("\nPER-POSSESSION cut counts:")
    for r in results:
        tag = f"Q{r['quarter']} {r['team']:>4}  {r['start_sec']}s-{r['end_sec']}s  ({r['duration_sec']}s)"
        flag = "  \u26a0 SUSPECT (zero cuts)" if r['suspect_zero_cuts'] else ""
        print(f"{tag}  cuts_in={r['n_cuts_in']}  cuts_out={r['n_cuts_out']}  "
              f"tracks_considered={r['n_tracks_considered']}{flag}")

    print("\n" + "=" * 70)
    print("GAME-LEVEL cut summary by team")
    print("=" * 70)
    for team, s in game_summary.items():
        total_poss = sum(1 for r in results if r['team'] == team)
        print(f"\n  {team}:")
        print(f"    Cuts in (Perimeter -> Net-to-8):  {s['cuts_in']}")
        print(f"    Cuts out (Net-to-8 -> Perimeter): {s['cuts_out']}")
        print(f"    Total cuts: {s['cuts_in'] + s['cuts_out']}")
        print(f"    Possessions with at least one cut: {s['possessions_with_cuts']}/{total_poss}")

    # A real lacrosse possession involves near-constant motion - genuinely
    # having ZERO cuts for its whole duration is implausible. Flag these
    # explicitly as unreliable tracking data for that specific window,
    # rather than something to trust at face value.
    suspects = [r for r in results if r['suspect_zero_cuts']]
    if suspects:
        print("\n" + "=" * 70)
        print(f"\u26a0 {len(suspects)} possession(s) show ZERO cuts - a real possession almost")
        print("  always involves at least one Perimeter<->Net-to-8 transition, so this")
        print("  points at unreliable/fragmented tracking for that window, not genuinely")
        print("  static play. Treat these with reduced confidence in the report:")
        print("=" * 70)
        for r in suspects:
            print(f"    Q{r['quarter']} {r['team']:>4}  {r['start_sec']}s-{r['end_sec']}s  "
                  f"({r['duration_sec']}s)  tracks_considered={r['n_tracks_considered']}  "
                  f"boundary_source={r.get('boundary_source', 'n/a')}")


if __name__ == '__main__':
    main()
