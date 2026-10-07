"""
analyze_formations.py
Step 7 — formation analysis. For each validated possession (from
detect_and_match_possessions.py's detected_possessions.json), counts the
possessing team's players by zone (Behind Net, Key, Perimeter) across the
possession's settled window, and determines the dominant formation shape.
Aggregates across all possessions into an overall formation-frequency
table per team.

RELIABILITY RULE: only frames where the possessing team's total count is
>= OFFENSE_FULL_FRAME (7 — the true roster size; allows 7 or 8 to cover
legitimate extra-man/MAN-UP situations) count toward the formation tally.
A frame with only 5 or 6 detected is EXCLUDED entirely, not corrected or
loosened — fixing those gaps belongs in a future interpolation step
(filling a temporarily-missed player using nearby tracked frames), not in
loosening this threshold. This will mean fewer qualifying frames per
possession than the earlier (looser) version; that's intentional, and the
qualifying-frame count is reported per possession so the impact is visible.

GOALIE NOTE: no position-based correction is applied to 'Goalie'-labeled
detections here. A real goalie can legitimately be off-center in the
crease, or entirely upfield outside of a possession window — being strict
about "the goalie must be centered on the goal line" would risk
reclassifying genuine goalie positions as errors. Left as-is for now.

ZONE DEFINITION FIX: the pipeline's stored 'zone' field uses a narrow,
width-shrinking "Key" diamond for the middle zone — useful for crease-
congestion detection elsewhere, but NOT what the original formation plan
meant by "Net to 8." That was always meant to be a full-width band,
sideline to sideline, bounded by the actual 8-yard line (a different
reference line than the Key-top line the stored zone uses). Using the
narrow diamond was pushing real net-to-8-depth players (standing wide, at
X, in the alley) into "Perimeter" instead — exactly why so many formations
were showing an implausible 0 in the middle slot. This script recomputes
the zone directly from each detection's raw cy against each camera's real
calibration lines, ignoring the stored (narrower) zone field entirely for
this purpose.

Usage:
    python analyze_formations.py detected_possessions.json \\
        "SYR UNC MVC Side.tracks.json" "SYR UNC LR Side.tracks.json"

Output:
    formation_analysis.json — per-possession formation data (full
    distribution, not just a single dominant value, plus frame reliability
    counts)
    Prints BOTH per-possession formation trends and the overall
    formation-frequency summary per team — not just the aggregate.
"""
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as pc

pc.pin_working_directory()
GAME_CONFIG = pc.load_game_config()
FPS = pc.get_fps(GAME_CONFIG)
other_team = pc.make_other_team(GAME_CONFIG)

ZONE_ORDER = ('Behind Net', 'Net-to-8', 'Perimeter')  # full-width bands, not the narrow Key diamond
OFFENSE_FULL_FRAME = 7  # exact-roster floor for offense — frames below this excluded entirely
DEFENSE_FULL_FRAME = 7  # same idea for defense (opponent + goalie combined)


def zone_from_point(cx, cy, camera):
    return pc.zone_from_point(cx, cy, camera, GAME_CONFIG)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_tracks(path):
    raw = load_json(path)
    return {int(k): v for k, v in raw.items()}


def analyze_possession(tracks, qs, qe, team, start_sec, end_sec, camera):
    """Returns (off_formation_counter, def_formation_counter, n_frames_total,
    n_off_qualifying, n_def_qualifying). Offense = the possessing team;
    defense = the opponent + Goalie. Both use the same full-width zone bands
    (see zone_from_cy) and the same exact-roster reliability rule, applied
    independently — a frame can qualify for offense, defense, both, or
    neither depending on each side's own count that frame."""
    def_team_name = other_team(team)
    f_lo = qs + int(start_sec * FPS)
    f_hi = qs + int(end_sec * FPS)
    off_formation_counts = Counter()
    def_formation_counts = Counter()
    n_total = 0
    n_off_qualifying = 0
    n_def_qualifying = 0

    for fi, dets in tracks.items():
        if not (f_lo <= fi <= f_hi):
            continue
        n_total += 1
        off_zone_counts = {z: 0 for z in ZONE_ORDER}
        def_zone_counts = {z: 0 for z in ZONE_ORDER}
        total_off = 0
        total_def = 0
        for d in dets:
            if d.get('interpolated'):
                continue  # formation counts must be exact/actually-observed;
                          # interpolated detections only help possession/cut
                          # detection see through brief gaps, never formation data
            zone = zone_from_point(d['cx'], d['cy'], camera)
            if zone not in ZONE_ORDER:
                continue
            dteam = d.get('team')
            if dteam == team:
                off_zone_counts[zone] += 1
                total_off += 1
            elif dteam == def_team_name or dteam == 'Goalie':
                def_zone_counts[zone] += 1
                total_def += 1

        if total_off >= OFFENSE_FULL_FRAME:
            n_off_qualifying += 1
            off_formation_counts[tuple(off_zone_counts[z] for z in ZONE_ORDER)] += 1
        if total_def >= DEFENSE_FULL_FRAME:
            n_def_qualifying += 1
            def_formation_counts[tuple(def_zone_counts[z] for z in ZONE_ORDER)] += 1

    return off_formation_counts, def_formation_counts, n_total, n_off_qualifying, n_def_qualifying


def main():
    if len(sys.argv) < 4:
        print("Usage: python analyze_formations.py detected_possessions.json "
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
    overall_off_freq = {}  # team -> Counter of dominant offensive formations
    overall_def_freq = {}  # team -> Counter of dominant DEFENSIVE formations (the opponent, defending)

    for d in detected:
        q, team, camera = d['quarter'], d['team'], d['camera']
        qs, qe = pc.get_quarter_frames(GAME_CONFIG, camera)[q]
        tracks = tracks_by_label[camera]
        off_counts, def_counts, n_total, n_off_qual, n_def_qual = analyze_possession(
            tracks, qs, qe, team, d['start_sec'], d['end_sec'], camera)

        off_dominant = max(off_counts, key=off_counts.get) if n_off_qual > 0 else None
        def_dominant = max(def_counts, key=def_counts.get) if n_def_qual > 0 else None
        def_team = other_team(team)

        results.append({
            **d,
            'offense_dominant_formation': off_dominant,
            'offense_formation_distribution': {str(k): v for k, v in off_counts.items()},
            'n_frames_offense_qualifying': n_off_qual,
            'defense_team': def_team,
            'defense_dominant_formation': def_dominant,
            'defense_formation_distribution': {str(k): v for k, v in def_counts.items()},
            'n_frames_defense_qualifying': n_def_qual,
            'n_frames_total': n_total,
        })

        if off_dominant is not None:
            overall_off_freq.setdefault(team, Counter())[off_dominant] += 1
        if def_dominant is not None:
            overall_def_freq.setdefault(def_team, Counter())[def_dominant] += 1

    out_path = Path('formation_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Analyzed {len(results)} possessions, saved to {out_path}")

    no_off = sum(1 for r in results if r['offense_dominant_formation'] is None)
    no_def = sum(1 for r in results if r['defense_dominant_formation'] is None)
    print(f"{no_off} possessions had zero qualifying OFFENSE frames; "
          f"{no_def} had zero qualifying DEFENSE frames")
    print(f"(both counts will likely shrink once a gap-filling/interpolation step exists)")

    print("\n" + "=" * 70)
    print("PER-POSSESSION formation trends (OFFENSE, then DEFENSE)")
    print("=" * 70)
    for r in results:
        tag = f"Q{r['quarter']} {r['team']:>4}  {r['start_sec']}s-{r['end_sec']}s  ({r['duration_sec']}s)"
        print(tag)

        if r['offense_dominant_formation'] is None:
            print(f"  OFFENSE ({r['team']}): no qualifying frames ({r['n_frames_offense_qualifying']}/{r['n_frames_total']})")
        else:
            dist = r['offense_formation_distribution']
            total_qual = sum(dist.values())
            ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
            top_str = ', '.join(f"{k}:{v}/{total_qual}({v/total_qual*100:.0f}%)" for k, v in ranked[:3])
            print(f"  OFFENSE ({r['team']})  [{r['n_frames_offense_qualifying']}/{r['n_frames_total']} qual]  {top_str}")

        if r['defense_dominant_formation'] is None:
            print(f"  DEFENSE ({r['defense_team']}): no qualifying frames ({r['n_frames_defense_qualifying']}/{r['n_frames_total']})")
        else:
            dist = r['defense_formation_distribution']
            total_qual = sum(dist.values())
            ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
            top_str = ', '.join(f"{k}:{v}/{total_qual}({v/total_qual*100:.0f}%)" for k, v in ranked[:3])
            print(f"  DEFENSE ({r['defense_team']})  [{r['n_frames_defense_qualifying']}/{r['n_frames_total']} qual]  {top_str}")
        print()

    print("=" * 70)
    print("OVERALL OFFENSIVE formation frequency by team (Behind Net, Net-to-8, Perimeter)")
    print("=" * 70)
    for team, counter in overall_off_freq.items():
        print(f"\n  {team} (offense):")
        total = sum(counter.values())
        for formation, count in counter.most_common():
            print(f"    {formation}: {count} possessions ({count/total*100:.0f}%)")

    print("\n" + "=" * 70)
    print("OVERALL DEFENSIVE formation frequency by team (Behind Net, Net-to-8, Perimeter)")
    print("=" * 70)
    for team, counter in overall_def_freq.items():
        print(f"\n  {team} (defense):")
        total = sum(counter.values())
        for formation, count in counter.most_common():
            print(f"    {formation}: {count} possessions ({count/total*100:.0f}%)")


if __name__ == '__main__':
    main()
