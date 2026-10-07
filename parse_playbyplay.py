"""
parse_playbyplay.py
Parses an NCAA/PrestoSports-style play-by-play text export (the same format
as the game PDF: lines like "13:42 SYRACUSE Ground ball pickup by SYRACUSE
Izzy Lahah.") into ground-truth possession segments with real game-clock
timestamps.

Team names are read directly from the text itself - this script never
hardcodes which two teams are playing, so it works unchanged for any game.
Only PERIOD_LENGTH_SEC comes from game_config.json (defaults to 900s /
15-minute quarters if not specified, but can differ for other levels/sports).

A new possession begins at any "Draw control by TEAM" or "Ground ball
pickup by TEAM" event, or at a "Clear attempt by TEAM" for a team that
isn't already tracked as possessing (catches turnovers where no explicit
pickup line was ever logged for the recovering team). A possession also
closes immediately at a "GOAL by TEAM" for the currently-possessing team,
using the goal's own real timestamp.

Lines with no explicit clock time ("--") are forward-filled to the next
timed event in the same period.

Usage:
    python parse_playbyplay.py <playbyplay.txt>

Output:
    official_possessions.json — list of {quarter, team, start_sec, end_sec, duration_sec}
    stoppage_markers.json — timeout/free-position events for clock alignment
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as pc

pc.pin_working_directory()
GAME_CONFIG = pc.load_game_config()
PERIOD_LENGTH_SEC = pc.get_period_length_sec(GAME_CONFIG)

LINE_RE = re.compile(r'^(?P<time>\d{2}:\d{2}|--)\s+(?P<team>\S+)\s*(?P<desc>.*)$')
GAIN_RE = re.compile(r'^(?:Draw control by|Ground ball pickup by)\s+(\S+)')
CLEAR_RE = re.compile(r'^Clear attempt by\s+(\S+)')
GOAL_RE = re.compile(r'^GOAL by\s+(\S+)')


def classify_stoppage(desc):
    """Definite whistle/dead-ball events, used as clock-alignment anchors
    in align_clocks.py. Broadened to include card events - a real whistle
    stoppage just like a timeout or free-position call, which a quarter
    can otherwise have zero of (no timeouts, no fouls called yet) even
    though the clock genuinely paused for other reasons."""
    if desc.startswith('Timeout by'):
        return 'timeout'
    if desc.startswith('Free position attempt for'):
        return 'free_position'
    if re.match(r'^(Green|Yellow|Red) card', desc):
        return 'card'
    return None


def parse_periods(text):
    """Split the raw text into a list of periods, each a list of
    {'time_str', 'team', 'desc'} events, in order."""
    periods = []
    current = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith('time team play'):
            continue
        m = LINE_RE.match(line)
        if not m:
            continue  # skip plain score-update lines like "North Carolina 1, Syracuse 0"

        time_str = m.group('time')
        team = m.group('team')
        desc = m.group('desc').strip()

        if team == '0' and 'End-of-period' in desc:
            periods.append(current)
            current = []
            continue
        if team == '0' and desc == '':
            continue  # the bare "15:00 0" period-start marker line

        current.append({'time_str': time_str, 'team': team, 'desc': desc})
    if current:
        periods.append(current)
    return periods


def clock_to_elapsed(time_str):
    if time_str == '--':
        return None
    mm, ss = time_str.split(':')
    remaining = int(mm) * 60 + int(ss)
    return PERIOD_LENGTH_SEC - remaining


def fill_missing_times(events):
    """Forward-fill '--' (untimed) events to the next known timestamp."""
    elapsed = [clock_to_elapsed(e['time_str']) for e in events]
    n = len(elapsed)
    for i in range(n):
        if elapsed[i] is None:
            j = i
            while j < n and elapsed[j] is None:
                j += 1
            elapsed[i] = elapsed[j] if j < n else (elapsed[i - 1] if i > 0 else 0.0)
    for e, t in zip(events, elapsed):
        e['elapsed_sec'] = t
    return events


def build_possessions_for_period(events, period_num):
    """
    Walks events in order, tracking who currently has the ball. A new
    possession begins whenever:
      - a 'Draw control by TEAM' or 'Ground ball pickup by TEAM' event fires
        (always a fresh gain), OR
      - a 'Clear attempt by TEAM' fires for a TEAM that is NOT the team
        currently tracked as possessing — this catches cases where a
        turnover is followed directly by the new team clearing, with no
        explicit 'Ground ball pickup' line ever logged for the transition.
    A GOAL by the currently-possessing team closes their segment AT THE
    GOAL'S OWN TIMESTAMP, rather than waiting for whatever gain event
    happens to come next (which is frequently untimed and forward-filled
    to a much later point).
    """
    possessions = []
    current_team = None
    current_start = None

    def close_current(end_t, end_reason='other'):
        if current_team is not None and current_start is not None and end_t > current_start:
            possessions.append({
                'quarter': period_num,
                'team': current_team,
                'start_sec': round(current_start, 1),
                'end_sec': round(end_t, 1),
                'duration_sec': round(end_t - current_start, 1),
                'end_reason': end_reason,
            })

    for e in events:
        desc = e['desc']
        t = e['elapsed_sec']

        mg = GOAL_RE.match(desc)
        if mg and mg.group(1) == current_team:
            close_current(t, end_reason='goal')
            current_team = None
            current_start = None
            continue

        gain_team = None
        m = GAIN_RE.match(desc)
        if m:
            gain_team = m.group(1)
        else:
            mc = CLEAR_RE.match(desc)
            if mc:
                clear_team = mc.group(1)
                if clear_team != current_team:
                    gain_team = clear_team

        if gain_team and gain_team != current_team:
            close_current(t, end_reason='turnover')
            current_team = gain_team
            current_start = t

    close_current(PERIOD_LENGTH_SEC, end_reason='period_end')
    return possessions


def extract_stoppage_markers(events, period_num):
    markers = []
    for e in events:
        kind = classify_stoppage(e['desc'])
        if kind:
            markers.append({
                'quarter': period_num,
                'elapsed_sec': round(e['elapsed_sec'], 1),
                'type': kind,
            })
    return markers


def extract_goal_markers(events, period_num):
    """Every 'GOAL by TEAM' line, matched DIRECTLY against the raw text -
    completely independent of possession-tracking state. A possession's own
    end_reason=='goal' tag depends on build_possessions_for_period
    correctly recognizing the scoring team as the currently-tracked
    possessor at that exact moment; any edge case there (unusual sequence,
    a man-up goal, etc.) could silently mislabel a real goal with some
    other end_reason even though the text plainly says GOAL. Scanning
    directly for the GOAL line itself can't miss a goal for that reason."""
    markers = []
    for e in events:
        m = GOAL_RE.match(e['desc'])
        if m:
            markers.append({
                'quarter': period_num,
                'elapsed_sec': round(e['elapsed_sec'], 1),
                'team': m.group(1),
            })
    return markers


def main():
    if len(sys.argv) < 2:
        print('Usage: python parse_playbyplay.py <playbyplay.txt>')
        sys.exit(1)

    path = Path(sys.argv[1])
    text = path.read_text()

    periods = parse_periods(text)
    print(f'Parsed {len(periods)} periods, '
          f'{sum(len(p) for p in periods)} total events')

    all_possessions = []
    all_stoppage_markers = []
    all_goal_markers = []
    for i, events in enumerate(periods, start=1):
        events = fill_missing_times(events)
        poss = build_possessions_for_period(events, i)
        all_possessions.extend(poss)
        all_stoppage_markers.extend(extract_stoppage_markers(events, i))
        all_goal_markers.extend(extract_goal_markers(events, i))
        print(f'  Q{i}: {len(events)} events -> {len(poss)} possessions')

    out_path = Path('official_possessions.json')
    with open(out_path, 'w') as f:
        json.dump(all_possessions, f, indent=2)

    markers_path = Path('stoppage_markers.json')
    with open(markers_path, 'w') as f:
        json.dump(all_stoppage_markers, f, indent=2)
    print(f'\u2713 Saved {len(all_stoppage_markers)} stoppage markers (timeouts/free-position) '
          f'to {markers_path}')

    goal_markers_path = Path('goal_markers.json')
    with open(goal_markers_path, 'w') as f:
        json.dump(all_goal_markers, f, indent=2)
    print(f'\u2713 Saved {len(all_goal_markers)} goal markers to {goal_markers_path}')
    goal_end_reason_count = sum(1 for p in all_possessions if p.get('end_reason') == 'goal')
    if goal_end_reason_count != len(all_goal_markers):
        print(f'  \u26a0 Note: {goal_end_reason_count} possessions ended with end_reason==\'goal\', '
              f'but {len(all_goal_markers)} GOAL lines exist in the raw text - some goals are not '
              f'lining up with possession-tracking as expected. goal_markers.json is the complete, '
              f'directly-extracted list; use it, not end_reason, for anything needing every goal.')

    print(f'\n\u2713 Saved {len(all_possessions)} official possessions to {out_path}')
    total_dur = sum(p['duration_sec'] for p in all_possessions)
    print(f'Total official possession time: {total_dur:.0f}s ({total_dur/60:.1f} min)')

    by_team = {}
    for p in all_possessions:
        by_team.setdefault(p['team'], []).append(p['duration_sec'])
    print('\nPossessions by team:')
    for team, durs in by_team.items():
        print(f'  {team}: {len(durs)} possessions, '
              f'avg {sum(durs)/len(durs):.1f}s, total {sum(durs):.0f}s')


if __name__ == '__main__':
    main()
