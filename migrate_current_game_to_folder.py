"""
migrate_current_game_to_folder.py
Moves the CURRENT SYR/UNC game's data files (videos, all .json outputs so
far, the play-by-play .txt) from Summer Attempt's root into
Summer Attempt/games/<GameName>/, so this existing game fits the new
per-game folder structure. Does NOT touch the python/ scripts folder or
any file not on the list below.

Run this ONCE, after upgrade_to_per_game_dirs.py, before using
run_pipeline.py with --game.

Usage:
    python migrate_current_game_to_folder.py "SYR_UNC_20260213"
"""
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(r"C:\Users\lynch\OneDrive\Syracuse\Computer Vision\Women's Lacrosse\Scheme Tracking\Summer Attempt")

# Every data file belonging to the current game, wherever it currently sits
# directly in Summer Attempt's root.
GAME_FILES = [
    "SYR UNC MVC Side.mp4",
    "SYR UNC LR Side.mp4",
    "SYR UNC MVC Side.detections.json",
    "SYR UNC LR Side.detections.json",
    "SYR UNC MVC Side.tracks.json",
    "SYR UNC LR Side.tracks.json",
    "SYR UNC MVC Side.player_clicks.json",
    "SYR UNC LR Side.player_clicks.json",
    "SYR UNC MVC Side.exclude_zones.json",
    "SYR UNC LR Side.exclude_zones.json",
    "playbyplay_UNC_CUSE_20260213.txt",
    "official_possessions.json",
    "stoppage_markers.json",
    "official_possessions_aligned.json",
    "goalie_team_SYR UNC MVC Side.json",
    "goalie_team_SYR UNC LR Side.json",
    "goalie_team_reconciled_SYR UNC MVC Side.json",
    "goalie_team_reconciled_SYR UNC LR Side.json",
    "detected_possessions.json",
    "formation_analysis.json",
    "cuts_analysis.json",
]


def main():
    if len(sys.argv) < 2:
        print('Usage: python migrate_current_game_to_folder.py "GameFolderName"')
        sys.exit(1)

    game_name = sys.argv[1]
    game_dir = BASE_DIR / "games" / game_name
    game_dir.mkdir(parents=True, exist_ok=True)

    print(f"Migrating current game's files into: {game_dir}\n")
    moved, missing = 0, []
    for fname in GAME_FILES:
        src = BASE_DIR / fname
        if src.exists():
            shutil.move(str(src), str(game_dir / fname))
            print(f"  [OK] moved {fname}")
            moved += 1
        else:
            missing.append(fname)

    if missing:
        print(f"\n  [--] not found (skipped, may not exist yet): {missing}")

    print(f"\nDone. Moved {moved}/{len(GAME_FILES)} files into {game_dir}")
    print(f'You can now run: python python\\run_pipeline.py --game "{game_name}"')


if __name__ == "__main__":
    main()
