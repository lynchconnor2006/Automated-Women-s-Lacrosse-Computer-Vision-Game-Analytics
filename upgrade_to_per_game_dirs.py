"""
upgrade_to_per_game_dirs.py
Upgrades the path-fix already inserted into each script (by
patch_paths_for_subfolder.py) so it resolves against a PER-GAME folder
(Summer Attempt/games/<GameName>/) when the LACROSSE_GAME_DIR environment
variable is set, falling back to the old Summer Attempt-root behavior if
it isn't set (so nothing breaks if you run a script manually without it).

Run this once, on the scripts already living in Summer Attempt/python/.
Safe to run multiple times — already-upgraded files are detected and
skipped.

Usage:
    python upgrade_to_per_game_dirs.py
"""
from pathlib import Path

BASE_DIR = Path(r"C:\Users\lynch\OneDrive\Syracuse\Computer Vision\Women's Lacrosse\Scheme Tracking\Summer Attempt")
PYTHON_DIR = BASE_DIR / "python"

ALL_NEEDED = [
    "Step1_track_players.py",
    "parse_playbyplay.py",
    "align_clocks.py",
    "identify_goalie_team.py",
    "reconcile_goalie_teams.py",
    "detect_and_match_possessions.py",
    "analyze_formations.py",
    "detect_cuts.py",
    "pre_colab_pick_players.py",
    "pre_colab_pick_exclusion_zones.py",
    "validate_detected_possessions.py",
    "visualize_possession.py",
    "verify_cut.py",
    "check_other_camera.py",
    "compare_raw_vs_aligned.py",
    "review_possessions.py",
]

OLD_BLOCK = "_BASE_DIR = Path(__file__).resolve().parent.parent\nos.chdir(_BASE_DIR)\n"
NEW_MARKER = "_ENV_GAME_DIR = os.environ.get"
NEW_BLOCK = (
    "_ENV_GAME_DIR = os.environ.get('LACROSSE_GAME_DIR')\n"
    "if _ENV_GAME_DIR:\n"
    "    _BASE_DIR = Path(_ENV_GAME_DIR)\n"
    "else:\n"
    "    _BASE_DIR = Path(__file__).resolve().parent.parent\n"
    "os.chdir(_BASE_DIR)\n"
)


def upgrade_file(path):
    text = path.read_text()
    if NEW_MARKER in text:
        return "already upgraded"
    if OLD_BLOCK not in text:
        return "old block not found - check manually"
    new_text = text.replace(OLD_BLOCK, NEW_BLOCK)
    path.write_text(new_text)
    return "upgraded"


def main():
    print(f"Upgrading scripts in: {PYTHON_DIR}\n")
    for fname in ALL_NEEDED:
        path = PYTHON_DIR / fname
        if not path.exists():
            print(f"  [--] not found (skipped): {fname}")
            continue
        result = upgrade_file(path)
        tag = "OK" if result == "upgraded" else ".."
        print(f"  [{tag}] {fname}: {result}")

    print("\nDone. Scripts now use LACROSSE_GAME_DIR when set, falling back to the old")
    print("Summer Attempt-root behavior otherwise.")


if __name__ == "__main__":
    main()
