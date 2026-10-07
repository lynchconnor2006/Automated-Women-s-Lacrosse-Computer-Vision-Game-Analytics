"""
run_pipeline.py
Master pipeline: runs Steps 4 (tracking) through 8 (cuts) in order for ONE
game, inside its own folder (Summer Attempt/games/<GameName>/), skipping
any step whose expected output already exists there. Stops immediately on
the first failure.

FULLY GAME-CONFIG DRIVEN: camera labels/video filenames, the play-by-play
filename, team names, quarter-frame boundaries, and field calibration all
come from that game's own game_config.json — nothing about a specific game
is hardcoded in this file or in any of the scripts it calls. See
create_game_config.py to generate a template for a brand new game.

WORKFLOW for a new game:
  1. python create_game_config.py "NewGameName"   (creates the folder +
     a template config.json for you to fill in)
  2. Put that game's raw inputs in Summer Attempt/games/NewGameName/
     (videos, .detections.json from Colab, the play-by-play .txt)
  3. Fill in game_config.json with that game's real team names, camera
     labels/videos, quarter frames, calibration, and jersey colors
  4. python run_pipeline.py --game "NewGameName"

Usage:
    python run_pipeline.py --game "SYR_UNC_20260213" [--force]

    --force : ignore cached outputs and re-run every step regardless.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parent
BASE_DIR = PYTHON_DIR.parent
GAMES_ROOT = BASE_DIR / "games"


def tracks_path(cam):
    return f"{cam['label']}.tracks.json"


def detections_path(cam):
    return f"{cam['label']}.detections.json"


def goalie_team_path(cam):
    return f"goalie_team_{cam['label']}.json"


def goalie_team_reconciled_path(cam):
    return f"goalie_team_reconciled_{cam['label']}.json"


def run_step(name, script, args, expected_outputs, force, game_dir):
    out_paths = [game_dir / o for o in expected_outputs]
    if not force and all(p.exists() for p in out_paths):
        print(f"[SKIP] {name} - output already exists: {expected_outputs}")
        return

    print(f"\n[RUN ] {name}")
    cmd = [sys.executable, str(PYTHON_DIR / script)] + args
    print(f"       {' '.join(cmd)}")

    env = os.environ.copy()
    env["LACROSSE_GAME_DIR"] = str(game_dir)
    result = subprocess.run(cmd, cwd=str(game_dir), env=env)

    if result.returncode != 0:
        print(f"\n[FAIL] {name} exited with code {result.returncode}. Stopping pipeline.")
        sys.exit(1)

    missing = [o for o, p in zip(expected_outputs, out_paths) if not p.exists()]
    if missing:
        print(f"\n[FAIL] {name} completed but expected output(s) not found: {missing}. "
              f"Stopping pipeline.")
        sys.exit(1)

    print(f"[OK  ] {name}")


def parse_args():
    if "--game" not in sys.argv:
        print("Usage: python run_pipeline.py --game \"GameFolderName\" [--force]")
        print(f"\nExisting game folders in {GAMES_ROOT}:")
        if GAMES_ROOT.exists():
            for d in sorted(GAMES_ROOT.iterdir()):
                if d.is_dir():
                    print(f"  {d.name}")
        else:
            print("  (none yet)")
        sys.exit(1)
    game_name = sys.argv[sys.argv.index("--game") + 1]
    force = "--force" in sys.argv
    return game_name, force


def main():
    game_name, force = parse_args()
    game_dir = GAMES_ROOT / game_name

    if not game_dir.exists():
        print(f"Game folder does not exist yet: {game_dir}")
        print('Run: python create_game_config.py "%s"  to set one up.' % game_name)
        sys.exit(1)

    config_path = game_dir / "game_config.json"
    if not config_path.exists():
        print(f"No game_config.json found in {game_dir}")
        print('Run: python create_game_config.py "%s"  to generate a template, then fill it in.'
              % game_name)
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    cameras = config["cameras"]
    playbyplay_txt = config["playbyplay_txt"]

    print("=" * 70)
    print(f"LACROSSE PIPELINE - game: {game_name}")
    print(f"Game folder: {game_dir}")
    print(f"Teams: {config['team_a']} vs {config['team_b']}")
    print(f"Cameras: {[c['label'] for c in cameras]}")
    if force:
        print("(--force: ignoring cached outputs, re-running everything)")
    print("=" * 70)

    for cam in cameras:
        run_step(
            f"Step 4 tracking [{cam['label']}]",
            "Step1_track_players.py",
            [detections_path(cam)],
            [tracks_path(cam)],
            force, game_dir,
        )

    for cam in cameras:
        interp_marker = game_dir / f"{tracks_path(cam)}.interpolated_marker"
        if not force and interp_marker.exists():
            print(f"[SKIP] Step 4.5 gap interpolation [{cam['label']}] - already done")
        else:
            print(f"\n[RUN ] Step 4.5 gap interpolation [{cam['label']}]")
            cmd = [sys.executable, str(PYTHON_DIR / "interpolate_track_gaps.py"), tracks_path(cam)]
            print(f"       {' '.join(cmd)}")
            env = os.environ.copy()
            env["LACROSSE_GAME_DIR"] = str(game_dir)
            result = subprocess.run(cmd, cwd=str(game_dir), env=env)
            if result.returncode != 0:
                print(f"\n[FAIL] Step 4.5 gap interpolation [{cam['label']}] exited with code "
                      f"{result.returncode}. Stopping pipeline.")
                sys.exit(1)
            interp_marker.write_text("done")
            print(f"[OK  ] Step 4.5 gap interpolation [{cam['label']}]")

    for cam in cameras:
        run_step(
            f"Goalie team ID [{cam['label']}]",
            "identify_goalie_team.py",
            [cam["video"], tracks_path(cam)],
            [goalie_team_path(cam)],
            force, game_dir,
        )

    run_step(
        "Reconcile goalie teams",
        "reconcile_goalie_teams.py",
        [goalie_team_path(cameras[0]), goalie_team_path(cameras[1])],
        [goalie_team_reconciled_path(cameras[0]), goalie_team_reconciled_path(cameras[1])],
        force, game_dir,
    )

    run_step(
        "Parse play-by-play",
        "parse_playbyplay.py",
        [playbyplay_txt],
        ["official_possessions.json", "stoppage_markers.json", "goal_markers.json"],
        force, game_dir,
    )

    run_step(
        "Align clocks",
        "align_clocks.py",
        [tracks_path(cameras[0]), tracks_path(cameras[1]),
         "official_possessions.json", "stoppage_markers.json", "goal_markers.json"],
        ["official_possessions_aligned.json"],
        force, game_dir,
    )

    run_step(
        "Detect and match possessions",
        "detect_and_match_possessions.py",
        ["official_possessions_aligned.json",
         tracks_path(cameras[0]), tracks_path(cameras[1]),
         goalie_team_reconciled_path(cameras[0]), goalie_team_reconciled_path(cameras[1])],
        ["detected_possessions.json"],
        force, game_dir,
    )

    # Always runs (never cached) - its correctness depends on manual tagging
    # progress (possession_results.json) which changes independently of the
    # rest of this automated pipeline, so a simple "does the output already
    # exist" check would silently miss newly-added corrections.
    print("\n[RUN ] Apply possession corrections")
    cmd = [sys.executable, str(PYTHON_DIR / "apply_possession_corrections.py"),
           "detected_possessions.json", "possession_results.json"]
    print(f"       {' '.join(cmd)}")
    env = os.environ.copy()
    env["LACROSSE_GAME_DIR"] = str(game_dir)
    result = subprocess.run(cmd, cwd=str(game_dir), env=env)
    if result.returncode != 0:
        print(f"\n[FAIL] Apply possession corrections exited with code {result.returncode}. "
              f"Stopping pipeline.")
        sys.exit(1)
    print("[OK  ] Apply possession corrections")

    run_step(
        "Analyze formations",
        "analyze_formations.py",
        ["detected_possessions_final.json", tracks_path(cameras[0]), tracks_path(cameras[1])],
        ["formation_analysis.json"],
        force, game_dir,
    )

    run_step(
        "Detect cuts",
        "detect_cuts.py",
        ["detected_possessions_final.json", tracks_path(cameras[0]), tracks_path(cameras[1])],
        ["cuts_analysis.json"],
        force, game_dir,
    )

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE for '{game_name}' - all steps ran successfully "
          f"(or were skipped as cached)")
    print("=" * 70)


if __name__ == "__main__":
    main()
