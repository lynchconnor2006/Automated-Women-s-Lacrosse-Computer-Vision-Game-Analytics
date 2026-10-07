"""
pre_colab_set_camera_side.py
"Side" (left/right) isn't saved anywhere as its own file - it's just the
inline 'camera': 'left'/'right' value you type directly into Colab's
CURRENT_VIDEO config dict (Cell 2) for each camera, and it's never read by
anything in the local pipeline. This is just a quick interactive reminder/
confirmation of which value to use for a given camera - nothing gets
written to game_config.json or anywhere else.

Usage:
    python pre_colab_set_camera_side.py <camera_label>
"""
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python pre_colab_set_camera_side.py <camera_label>")
        sys.exit(1)

    label = sys.argv[1]
    side = input(f"Which side of the field does '{label}' cover? (left/right): ").strip().lower()
    while side not in ("left", "right"):
        side = input("Please type 'left' or 'right': ").strip().lower()

    print(f"\nIn Colab's Cell 2, set:")
    print(f"    CURRENT_VIDEO = {{'label': '{label}', 'drive_path': ..., 'camera': '{side}'}}")


if __name__ == "__main__":
    main()
