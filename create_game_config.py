"""
create_game_config.py
Creates Summer Attempt/games/<GameName>/ and drops a TEMPLATE
game_config.json in it for you to fill in with that game's real values.
This is the starting point for adding a genuinely new game to the pipeline.

What you'll need to fill in (all currently placeholder/example values):
  - team_a / team_b: the two team abbreviations exactly as they appear in
    BOTH the play-by-play text AND however your Colab detection pipeline
    labeled teams in the .tracks.json files.
  - team_aliases: only needed if the play-by-play text uses a different
    spelling than the tracks.json team labels (e.g. play-by-play says
    "SYRACUSE" but tracks.json says "SYR" -> {"SYRACUSE": "SYR"}).
  - cameras: one entry per camera (this pipeline is built for exactly 2).
    For each: label (must match the .tracks.json filename before
    ".tracks.json"), video filename, quarter_frames (frame number ranges
    per quarter - from your Colab quarter-marking step), calibration_y
    (goal line / 8-yard line / 30-yard line pixel Y-coordinates - from
    Colab's field calibration), and team_colors (HSV jersey color ranges
    per team - from Colab's jersey classifier calibration).
  - playbyplay_txt: the play-by-play text filename.
  - trusted_camera: which camera's goalie-color reading to trust more when
    the two disagree (pick whichever one showed a cleaner Q1/Q3/Q2/Q4
    alternating pattern during setup - see identify_goalie_team.py output).

Usage:
    python create_game_config.py "NewGameName"
"""
import json
import sys
from pathlib import Path

BASE_DIR = Path(r"C:\Users\lynch\OneDrive\Syracuse\Computer Vision\Women's Lacrosse\Scheme Tracking\Summer Attempt")

TEMPLATE = {
    "team_a": "TEAMA",
    "team_b": "TEAMB",
    "team_aliases": {
        "TEAMA_FULL_NAME_IF_DIFFERENT_IN_PLAYBYPLAY": "TEAMA"
    },
    "playbyplay_txt": "playbyplay_TEMPLATE.txt",
    "period_length_sec": 900,
    "fps": 30.0,
    "trusted_camera": "CAMERA_1_LABEL",
    "cameras": [
        {
            "label": "CAMERA_1_LABEL",
            "video": "CAMERA_1_LABEL.mp4",
            "quarter_frames": {
                "1": [0, 0],
                "2": [0, 0],
                "3": [0, 0],
                "4": [0, 0]
            },
            "calibration_y": {"goal_y": 0, "yard8_y": 0, "yard30_y": 0},
            "team_colors": {
                "TEAMA": [[0, 0, 0], [180, 70, 255]],
                "TEAMB": [[0, 0, 0], [180, 70, 255]]
            }
        },
        {
            "label": "CAMERA_2_LABEL",
            "video": "CAMERA_2_LABEL.mp4",
            "quarter_frames": {
                "1": [0, 0],
                "2": [0, 0],
                "3": [0, 0],
                "4": [0, 0]
            },
            "calibration_y": {"goal_y": 0, "yard8_y": 0, "yard30_y": 0},
            "team_colors": {
                "TEAMA": [[0, 0, 0], [180, 70, 255]],
                "TEAMB": [[0, 0, 0], [180, 70, 255]]
            }
        }
    ]
}


def main():
    if len(sys.argv) < 2:
        print('Usage: python create_game_config.py "NewGameName"')
        sys.exit(1)

    game_name = sys.argv[1]
    game_dir = BASE_DIR / "games" / game_name
    game_dir.mkdir(parents=True, exist_ok=True)

    config_path = game_dir / "game_config.json"
    if config_path.exists():
        print(f"game_config.json already exists at {config_path} - not overwriting.")
        sys.exit(0)

    with open(config_path, "w") as f:
        json.dump(TEMPLATE, f, indent=2)

    print(f"Created {game_dir}")
    print(f"Template config written to {config_path}")
    print("\nNext steps:")
    print(f"  1. Put this game's raw files (videos, .detections.json, play-by-play .txt) in {game_dir}")
    print(f"  2. Edit {config_path} with this game's real team names, camera labels, quarter")
    print("     frames, field calibration, and jersey colors")
    print(f'  3. python run_pipeline.py --game "{game_name}"')


if __name__ == "__main__":
    main()
