"""
tag_possessions.py
Step 9 — manual possession outcome tagging. Plays each possession's clip
(by default just the last PREVIEW_SECS seconds, since that's usually where
the outcome-defining moment happens - the shot, turnover, or whistle) and
lets you tag a result plus an optional man-situation, resuming at the first
untagged possession each time you run it.

Deliberately kept SEPARATE from run_pipeline.py - see conversation for why:
tagging is a genuinely manual, multi-sitting task, unlike the automated,
fast, cacheable Steps 4-8. Run this whenever you're ready to review film,
not as part of the automated batch run.

Controls:
  SPACE       : play/pause
  R           : restart the short preview (last PREVIEW_SECS before the end)
  W           : watch the FULL possession from its actual start
  ]           : extend the clip end further (in case the result still isn't
                shown yet - pushes the end out a few more seconds, repeatable)
  [           : rewind 8 seconds from wherever you currently are (no need to
                jump all the way back to the start to re-watch a moment)
  G           : Goal
  K           : Goal - Penalty Shot (distinct from a normal run-of-play goal)
  S           : Save
  C           : Caused Turnover
  U           : Unforced Turnover
  T           : Shot Clock / Violation
  P           : Penalty
  O           : Shot Out of Bounds - Defense Recovery
  F           : Foul - Turnover
  E           : End of Quarter
  1 / 2 / 3   : Man situation - Normal / Man Up / Man Down (relative to the
                possessing team - no team name needed)
  N           : add a text note
  TAB / ENTER : next possession
  B           : previous possession
  Q           : quit (already-tagged possessions are saved as you go)

Usage:
    python tag_possessions.py detected_possessions.json

Output:
    possession_results.json - one entry per tagged possession, matched back
    to detected_possessions.json via quarter+team+start_sec+end_sec.
"""
import sys
import os
import json
import subprocess
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_common as pc

GAME_DIR = pc.pin_working_directory()
GAME_CONFIG = pc.load_game_config()
FPS = pc.get_fps(GAME_CONFIG)

DISP_W, DISP_H = 1280, 720
PREVIEW_SECS = 8   # how many seconds before the end to start the default preview
END_BUFFER_SEC = 4  # extra seconds tacked onto the clip's end - our own detector's
                    # end_sec marks when the settled FORMATION broke down, which is
                    # often a moment before the actual shot/whistle/celebration
EXTEND_STEP_SEC = 3  # how much ']' pushes the end out per press
REWIND_STEP_SEC = 8  # how much '[' rewinds from the current position per press

RESULT_KEYS = {
    ord('g'): 'Goal', ord('G'): 'Goal',
    ord('k'): 'Goal - Penalty Shot', ord('K'): 'Goal - Penalty Shot',
    ord('s'): 'Save', ord('S'): 'Save',
    ord('c'): 'Caused Turnover', ord('C'): 'Caused Turnover',
    ord('u'): 'Unforced Turnover', ord('U'): 'Unforced Turnover',
    ord('t'): 'Shot Clock', ord('T'): 'Shot Clock',
    ord('p'): 'Penalty', ord('P'): 'Penalty',
    ord('o'): 'Shot OOB - Defense Recovery', ord('O'): 'Shot OOB - Defense Recovery',
    ord('f'): 'Foul - Turnover', ord('F'): 'Foul - Turnover',
    ord('e'): 'End of Quarter', ord('E'): 'End of Quarter',
}
MAN_KEYS = {
    ord('1'): 'Normal',
    ord('2'): 'Man Up',
    ord('3'): 'Man Down',
}


def video_path_for_camera(camera_label):
    for cam in pc.get_cameras(GAME_CONFIG):
        if cam['label'] == camera_label:
            return cam['video']
    raise KeyError(f"No camera config for '{camera_label}'")


def possession_id(p):
    return f"Q{p['quarter']}_{p['team']}_{p['start_sec']}_{p['end_sec']}"


def load_possessions(path):
    with open(path) as f:
        return json.load(f)


def load_results(path):
    if Path(path).exists():
        with open(path) as f:
            return {r['possession_id']: r for r in json.load(f)}
    return {}


def save_results(results_dict, path):
    with open(path, 'w') as f:
        json.dump(list(results_dict.values()), f, indent=2)


def play_possession(p, video_path, results_dict, idx, total):
    pid = possession_id(p)
    qframes = pc.get_quarter_frames(GAME_CONFIG, p['camera'])
    qs, qe = qframes[p['quarter']]
    start_f = qs + int(p['start_sec'] * FPS)
    end_f = min(qe, qs + int((p['end_sec'] + END_BUFFER_SEC) * FPS))

    preview_start = max(start_f, end_f - int(FPS * PREVIEW_SECS))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return None, 'next'

    existing = results_dict.get(pid, {})
    state = {
        'result': existing.get('result'),
        'man_situation': existing.get('man_situation', 'Normal'),
        'notes': existing.get('notes', ''),
        'corrected_end_sec': existing.get('corrected_end_sec'),
        'playing': True,
        'action': 'next',
        'tagged': existing.get('result') is not None,
    }

    def draw_hud(frame, fi):
        img = cv2.resize(frame, (DISP_W, DISP_H)) if frame is not None \
            else np.zeros((DISP_H, DISP_W, 3), np.uint8)

        total_f = max(end_f - start_f, 1)
        prog = max(0, min((fi - start_f) / total_f, 1))
        cv2.rectangle(img, (0, DISP_H - 6), (DISP_W, DISP_H), (40, 40, 40), -1)
        cv2.rectangle(img, (0, DISP_H - 6), (int(DISP_W * prog), DISP_H), (0, 200, 100), -1)
        preview_x = int(DISP_W * (preview_start - start_f) / total_f)
        cv2.line(img, (preview_x, DISP_H - 6), (preview_x, DISP_H), (0, 180, 255), 2)

        cv2.rectangle(img, (0, DISP_H - 110), (DISP_W, DISP_H - 6), (0, 0, 0), -1)
        tc = (0, 255, 128) if state['tagged'] else (0, 140, 255)
        res_str = state['result'] if state['result'] else '[ not tagged ]'
        ts = fi / FPS
        m, s = int(ts // 60), ts % 60
        corrected_str = f"  [TRUE END: {state['corrected_end_sec']:.1f}s]" if state['corrected_end_sec'] is not None else ""

        cv2.putText(img,
            f"{pid} | Q{p['quarter']} | {p['team']} | {p['duration_sec']:.0f}s | ({idx+1}/{total}){corrected_str}",
            (8, DISP_H - 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
        cv2.putText(img,
            f"Result: {res_str}   Man: {state['man_situation']}   "
            f"Notes: {state['notes'][:40] or '(none)'}   {m:02d}:{s:05.2f}",
            (8, DISP_H - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.46, tc, 1)
        cv2.putText(img,
            "G=Goal  K=PenaltyShotGoal  S=Save  C=CausedTO  U=UnforcedTO  T=ShotClock  P=Penalty  O=ShotOOB-DefRecovery  F=Foul-Turnover  E=EndOfQtr  "
            "1=Normal  2=ManUp  3=ManDown",
            (8, DISP_H - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        cv2.putText(img,
            "SPACE=play/pause  R=restart preview  W=full clip  [=rewind 8s  ]=extend end  X=mark TRUE end here  "
            "N=note  B=prev  TAB/ENTER=next  Q=quit",
            (8, DISP_H - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 100), 1)

        mc = {'Normal': (80, 80, 80), 'Man Up': (0, 180, 0), 'Man Down': (0, 0, 200)}.get(
            state['man_situation'], (80, 80, 80))
        cv2.rectangle(img, (DISP_W - 160, DISP_H - 108), (DISP_W - 4, DISP_H - 80), mc, -1)
        cv2.putText(img, state['man_situation'],
                    (DISP_W - 155, DISP_H - 88), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        return img

    WIN = f'Tagger_{pid}'
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    cap.set(cv2.CAP_PROP_POS_FRAMES, preview_start)
    fi = preview_start
    ret, frame = cap.read()

    while True:
        if state['playing'] and fi <= end_f:
            ret, frame = cap.read()
            if ret:
                cv2.imshow(WIN, draw_hud(frame, fi))
                fi += 1
            else:
                state['playing'] = False
        else:
            state['playing'] = False
            cv2.imshow(WIN, draw_hud(frame, fi))

        delay = max(1, int(1000 / FPS)) if state['playing'] else 20
        k = cv2.waitKey(delay) & 0xFF

        if k in RESULT_KEYS:
            state['result'] = RESULT_KEYS[k]
            state['tagged'] = True
            print(f"  {pid} -> {state['result']}")
        elif k in MAN_KEYS:
            state['man_situation'] = MAN_KEYS[k]
            print(f"  {pid} man -> {state['man_situation']}")
        elif k == ord(' '):
            state['playing'] = not state['playing']
            if state['playing'] and fi >= end_f:
                fi = preview_start
                cap.set(cv2.CAP_PROP_POS_FRAMES, preview_start)
        elif k in (ord('r'), ord('R')):
            fi = preview_start
            cap.set(cv2.CAP_PROP_POS_FRAMES, preview_start)
            state['playing'] = True
        elif k in (ord('w'), ord('W')):
            fi = start_f
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            state['playing'] = True
        elif k == ord(']'):
            new_end_f = min(qe, end_f + int(EXTEND_STEP_SEC * FPS))
            if new_end_f > end_f:
                end_f = new_end_f
                state['playing'] = True
                print(f"  Extended clip end by {EXTEND_STEP_SEC}s")
            else:
                print(f"  Already at the end of this quarter's video - can't extend further")
        elif k == ord('['):
            fi = max(start_f, fi - int(REWIND_STEP_SEC * FPS))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            state['playing'] = True
            print(f"  Rewound {REWIND_STEP_SEC}s")
        elif k in (ord('x'), ord('X')):
            state['corrected_end_sec'] = round((fi - qs) / FPS, 1)
            print(f"  {pid}: marked TRUE end at {state['corrected_end_sec']}s "
                  f"(detector had it at {p['end_sec']}s)")
        elif k in (ord('n'), ord('N')):
            cap.release()
            cv2.destroyWindow(WIN)
            state['notes'] = input(f"  Note for {pid}: ").strip()
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        elif k in (81, ord('b'), ord('B')):
            state['action'] = 'prev'
            break
        elif k in (83, 9, 13, ord('\r'), ord('\n')):
            state['action'] = 'next'
            break
        elif k in (ord('q'), ord('Q')):
            state['action'] = 'quit'
            break

    cap.release()
    cv2.destroyWindow(WIN)

    entry = None
    if state['tagged'] or state['corrected_end_sec'] is not None:
        entry = {
            'possession_id': pid,
            'quarter': p['quarter'], 'team': p['team'], 'camera': p['camera'],
            'start_sec': p['start_sec'], 'end_sec': p['end_sec'],
            'corrected_end_sec': state['corrected_end_sec'],
            'result': state['result'],
            'man_situation': state['man_situation'],
            'notes': state['notes'],
            'tagged_at': datetime.now().isoformat(),
        }
    return entry, state['action']


def main():
    if len(sys.argv) < 2:
        print("Usage: python tag_possessions.py detected_possessions.json")
        sys.exit(1)

    possessions_path = Path(sys.argv[1])
    results_path = Path('possession_results.json')

    possessions = load_possessions(possessions_path)
    possessions.sort(key=lambda p: (p['quarter'], p['start_sec']))
    results_dict = load_results(results_path)
    total = len(possessions)

    print(f"\n{'='*60}")
    print(f"  POSSESSION TAGGER - {total} possessions")
    print(f"  Already tagged: {len(results_dict)}")
    print(f"{'='*60}")
    print(f"  Default preview: last {PREVIEW_SECS} seconds (+{END_BUFFER_SEC}s buffer). "
          f"W = watch full clip, R = restart preview, ] = extend end.")
    print("  G/K/S/C/U/T/P/O/F/E = tag result   1/2/3 = man situation")
    print("  N = note   B = prev   TAB/ENTER = next   Q = quit\n")

    untagged = [i for i, p in enumerate(possessions) if possession_id(p) not in results_dict]

    if not untagged:
        print(f"  All {total} possessions are already tagged - nothing to review.")
        idx = total  # skips the loop below entirely
    else:
        idx = untagged[0]
        print(f"  Resuming at possession {idx+1} (first untagged)")

    while 0 <= idx < total:
        p = possessions[idx]
        video_path = video_path_for_camera(p['camera'])

        entry, action = play_possession(p, video_path, results_dict, idx, total)

        if entry:
            results_dict[entry['possession_id']] = entry
            save_results(results_dict, results_path)
            tagged = sum(1 for pp in possessions if possession_id(pp) in results_dict)
            print(f"  Saved {entry['possession_id']} -> {entry['result']}  ({tagged}/{total} tagged)")

        if action == 'next':
            idx += 1
        elif action == 'prev':
            idx = max(0, idx - 1)
        elif action == 'quit':
            break

    save_results(results_dict, results_path)
    tagged_count = sum(1 for p in possessions if possession_id(p) in results_dict)
    print(f"\nTagging session ended: {tagged_count}/{total} tagged overall")
    print(f"Saved to {results_path}")

    rerun_downstream_steps(possessions_path)


def rerun_downstream_steps(possessions_path):
    """Reruns Step 6.5 (apply corrections) -> Step 7 (formations) -> Step 8
    (cuts) automatically once tagging ends, so any 'X' corrections made
    this session are immediately reflected - no separate manual step or
    full pipeline rerun needed."""
    python_dir = Path(__file__).resolve().parent
    cameras = pc.get_cameras(GAME_CONFIG)
    tracks_paths = [f"{cam['label']}.tracks.json" for cam in cameras]

    env = os.environ.copy()
    env["LACROSSE_GAME_DIR"] = str(GAME_DIR)

    steps = [
        ("Apply possession corrections", "apply_possession_corrections.py",
         [str(possessions_path), "possession_results.json"]),
        ("Analyze formations", "analyze_formations.py",
         ["detected_possessions_final.json"] + tracks_paths),
        ("Detect cuts", "detect_cuts.py",
         ["detected_possessions_final.json"] + tracks_paths),
    ]

    print("\n" + "=" * 60)
    print("Rerunning formations/cuts with your tagging corrections applied...")
    print("=" * 60)
    for name, script, args in steps:
        print(f"\n[RUN ] {name}")
        cmd = [sys.executable, str(python_dir / script)] + args
        result = subprocess.run(cmd, cwd=str(GAME_DIR), env=env)
        if result.returncode != 0:
            print(f"[FAIL] {name} exited with code {result.returncode} - "
                  f"formation_analysis.json/cuts_analysis.json may be stale. "
                  f"Check the error above and rerun manually if needed.")
            return
        print(f"[OK  ] {name}")

    print("\n\u2713 formation_analysis.json and cuts_analysis.json are up to date "
          "with your tagging corrections.")


if __name__ == '__main__':
    main()
