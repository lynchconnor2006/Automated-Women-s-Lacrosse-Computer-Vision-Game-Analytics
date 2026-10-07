"""
interpolate_track_gaps.py
Step 4.5 - fills short gaps in individual player tracks (a real player
temporarily missed by detection due to occlusion/crowding, then re-detected
shortly after) by linearly interpolating their position across the gap.

WHY: possession-boundary detection (Step 6) counts players per frame; a
detection RECALL gap (not a real end of play) can make the count drop
below threshold long enough to prematurely end a possession, even though
the same players are still on the field the whole time. Filling short gaps
at the source fixes this without loosening possession-detection thresholds
(tried that - it caused separate real possessions to merge together).

Interpolated detections are tagged 'interpolated': true, so formation
analysis (Step 7) can still require an EXACT, actually-observed roster
count and exclude synthetic frames from that tally. This only helps
possession-boundary detection (Step 6) and cut detection (Step 8) see
through brief real gaps - it never fabricates formation data.

LIMITATION: an interpolated point reuses whichever real endpoint's 'zone'
is temporally closer, rather than precisely recomputing the zone from the
interpolated position - a reasonable approximation for bridging a short
gap, not a substitute for real detection.

Usage:
    python interpolate_track_gaps.py <tracks.json>

Output:
    <tracks.json> is updated in place (a .backup.json copy is made first,
    only if one doesn't already exist).
"""
import json
import sys
import shutil
from pathlib import Path

YOLO_STRIDE = 3       # detections are sampled every 3rd real frame
MAX_GAP_FRAMES = 30   # ~1 second at 30fps - shortened from 3s. A genuine occlusion
                      # blip is usually brief; a longer gap is more likely a track ID
                      # getting reassigned to a different real person than a single
                      # player staying hidden that long
MAX_JUMP_BOX_WIDTHS = 8  # skip interpolation if the two endpoints are farther apart
                         # than this many box-widths - a real player can't teleport,
                         # so a big jump means this is probably a DIFFERENT person who
                         # picked up the same track_id after a gap, not real occlusion


def main():
    if len(sys.argv) < 2:
        print("Usage: python interpolate_track_gaps.py <tracks.json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    backup = Path(str(path) + '.backup')
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"Backed up original to {backup.name}")
    else:
        print(f"Backup already exists ({backup.name}) - not overwriting it")

    with open(path) as f:
        raw = json.load(f)
    tracks = {int(k): v for k, v in raw.items()}
    print(f"Loaded {len(tracks):,} frames")

    by_track = {}
    for fi, dets in tracks.items():
        for d in dets:
            tid = d.get('track_id')
            if tid is None:
                continue
            by_track.setdefault(tid, []).append((fi, d))

    n_filled = 0
    n_gaps_too_long = 0
    n_gaps_too_far = 0
    for tid, entries in by_track.items():
        entries.sort(key=lambda x: x[0])
        for (fa, da), (fb, db) in zip(entries, entries[1:]):
            gap = fb - fa
            if gap <= YOLO_STRIDE:
                continue  # no missing sampled frame in between
            if gap > MAX_GAP_FRAMES:
                n_gaps_too_long += 1
                continue  # likely a genuine absence, don't fabricate across it

            box_width_avg = ((da['r'] - da['l']) + (db['r'] - db['l'])) / 2
            dist = ((da['cx'] - db['cx']) ** 2 + (da['cy'] - db['cy']) ** 2) ** 0.5
            if box_width_avg > 0 and dist > box_width_avg * MAX_JUMP_BOX_WIDTHS:
                n_gaps_too_far += 1
                continue  # implausible jump - likely a different real person
                          # reusing this track_id, not genuine occlusion

            fi = fa + YOLO_STRIDE
            while fi < fb:
                frac = (fi - fa) / gap
                interp = {
                    'l': int(round(da['l'] + (db['l'] - da['l']) * frac)),
                    't': int(round(da['t'] + (db['t'] - da['t']) * frac)),
                    'r': int(round(da['r'] + (db['r'] - da['r']) * frac)),
                    'b': int(round(da['b'] + (db['b'] - da['b']) * frac)),
                    'cx': int(round(da['cx'] + (db['cx'] - da['cx']) * frac)),
                    'cy': int(round(da['cy'] + (db['cy'] - da['cy']) * frac)),
                    'team': da['team'],
                    'zone': da['zone'] if frac < 0.5 else db['zone'],
                    'track_id': tid,
                    'interpolated': True,
                }
                tracks.setdefault(fi, []).append(interp)
                n_filled += 1
                fi += YOLO_STRIDE

    with open(path, 'w') as f:
        json.dump({str(k): v for k, v in tracks.items()}, f)

    print(f"\nFilled {n_filled} interpolated detections across gaps <= {MAX_GAP_FRAMES} frames "
          f"(~{MAX_GAP_FRAMES/30:.1f}s)")
    print(f"Left {n_gaps_too_long} longer gaps untouched (likely genuine absences, not occlusion blips)")
    print(f"Skipped {n_gaps_too_far} gaps with an implausible spatial jump "
          f"(likely a different real person reusing the same track_id, not real occlusion)")
    print(f"Saved back to {path}")


if __name__ == '__main__':
    main()
