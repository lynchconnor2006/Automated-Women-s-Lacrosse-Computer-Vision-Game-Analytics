"""
pre_colab_compute_team_colors.py
Fills the real gap between player_clicks.json (raw jersey click
coordinates + frame, from pre_colab_pick_players.py) and the actual HSV
color RANGE each team needs in game_config.json's "team_colors".

For each team, samples the jersey pixel color at every click (small patch
around the clicked point, on the exact frame it was clicked), then builds
an HSV range from the STATISTICAL SPREAD of all observed samples (mean +-
a few standard deviations), not raw min/max padded by a fixed constant.

Why this matters: a fixed-padding range works fine for an achromatic
jersey (white/gray - "low saturation, high brightness" stays fairly
stable across most lighting), but a specific-hue jersey (e.g. blue) shifts
meaningfully in saturation and value under shadow/lighting variation. A
handful of clicks under similar lighting will under-represent that real
spread, and a fixed pad on top of raw min/max doesn't compensate - it
silently produces a too-narrow range that undercounts that team all game,
with no error to flag it. A std-based range naturally widens as more
clicks reveal more of the true variation, and stays honest about how much
is really known when only a few clicks exist.

Usage:
    python pre_colab_compute_team_colors.py <video_path> <player_clicks.json>
    (run from inside the game's folder, so game_config.json is found)

Output:
    <video_stem>.team_colors.json
    Also auto-updates the matching camera's "team_colors" directly in
    game_config.json (backed up first, once, as game_config.json.backup) -
    no manual copy-paste needed.
"""
import sys
import json
import cv2
import numpy as np
from pathlib import Path

PATCH_HALF_SIZE = 8       # sample an 17x17 pixel patch around each click
STD_MULTIPLIER = 3.0      # range = mean +/- STD_MULTIPLIER * std, so it scales
                          # with how much real variation the clicks reveal
MIN_HUE_HALF_RANGE = 8    # floor on range half-width even with very few clicks
                          # (std alone is unreliable with only 1-2 samples)
MIN_SAT_HALF_RANGE = 35
MIN_VAL_HALF_RANGE = 35
RECOMMENDED_MIN_CLICKS = 8  # clicks per team, spread across different
                            # quarters/lighting/distances from camera
ACHROMATIC_SAT_THRESHOLD = 50  # if a team's mean saturation is below this,
                                # treat it as a white/gray-style jersey and
                                # ignore hue entirely (hue is mathematically
                                # unstable/noise for near-white pixels - a
                                # std-based hue range would narrow it based
                                # on that noise, which can reject real
                                # samples of the same jersey by chance)


def sample_hsv_patch(frame, x, y):
    h, w = frame.shape[:2]
    x0, x1 = max(0, x - PATCH_HALF_SIZE), min(w, x + PATCH_HALF_SIZE)
    y0, y1 = max(0, y - PATCH_HALF_SIZE), min(h, y + PATCH_HALF_SIZE)
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    return cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)


def main():
    if len(sys.argv) < 3:
        print("Usage: python pre_colab_compute_team_colors.py <video_path> <player_clicks.json>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    clicks_path = Path(sys.argv[2])

    with open(clicks_path) as f:
        clicks_data = json.load(f)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        sys.exit(1)

    samples_by_team = {}
    clicks_by_team = {}
    for click in clicks_data["clicks"]:
        team = click["team"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, click["frame"])
        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: could not read frame {click['frame']} for a {team} click, skipping")
            continue
        hsv_patch = sample_hsv_patch(frame, click["x"], click["y"])
        if hsv_patch is None or hsv_patch.size == 0:
            continue
        pixels = hsv_patch.reshape(-1, 3)
        samples_by_team.setdefault(team, []).append(pixels)
        clicks_by_team[team] = clicks_by_team.get(team, 0) + 1
        print(f"  Sampled {team} click at frame {click['frame']} ({click['x']},{click['y']}): "
              f"median HSV = {tuple(int(v) for v in np.median(pixels, axis=0))}")

    cap.release()

    team_colors = {}
    for team, patches in samples_by_team.items():
        all_pixels = np.concatenate(patches, axis=0).astype(np.float64)
        mean = all_pixels.mean(axis=0)
        std = all_pixels.std(axis=0)

        h_half = max(STD_MULTIPLIER * std[0], MIN_HUE_HALF_RANGE)
        s_half = max(STD_MULTIPLIER * std[1], MIN_SAT_HALF_RANGE)
        v_half = max(STD_MULTIPLIER * std[2], MIN_VAL_HALF_RANGE)

        is_achromatic = mean[1] < ACHROMATIC_SAT_THRESHOLD
        if is_achromatic:
            # hue is noise for near-white/gray pixels - don't constrain it
            h_lo, h_hi = 0, 180
        else:
            h_lo = max(0, int(round(mean[0] - h_half)))
            h_hi = min(180, int(round(mean[0] + h_half)))

        lo = [
            h_lo,
            max(0, int(round(mean[1] - s_half))),
            max(0, int(round(mean[2] - v_half))),
        ]
        hi = [
            h_hi,
            min(255, int(round(mean[1] + s_half))),
            min(255, int(round(mean[2] + v_half))),
        ]
        team_colors[team] = [lo, hi]
        if is_achromatic:
            print(f"  {team}: mean saturation {mean[1]:.0f} is low - treating as a "
                  f"white/gray-style jersey, hue left unconstrained (full 0-180)")

    out_path = video_path.parent / f"{video_path.stem}.team_colors.json"
    with open(out_path, "w") as f:
        json.dump(team_colors, f, indent=2)
    print(f"\nSaved {out_path}")

    for team, n in clicks_by_team.items():
        if n < RECOMMENDED_MIN_CLICKS:
            print(f"  \u26a0 {team}: only {n} click(s) - recommend at least "
                  f"{RECOMMENDED_MIN_CLICKS}, spread across different quarters, "
                  f"lighting, and distances from the camera. Too few clicks means "
                  f"this range is built mostly from the floor values, not real "
                  f"observed variation - it may not cover shadowed/distant players.")

    config_path = Path("game_config.json")
    if not config_path.exists():
        print(f"\n\u26a0 game_config.json not found in the current directory - "
              f"couldn't auto-update it. Here's the snippet to paste in manually:")
        print(json.dumps({"team_colors": team_colors}, indent=2))
        return

    with open(config_path) as f:
        config = json.load(f)

    camera_label = video_path.stem
    matched_cam = None
    for cam in config.get("cameras", []):
        if cam.get("label") == camera_label or cam.get("video") == video_path.name:
            matched_cam = cam
            break

    if matched_cam is None:
        print(f"\n\u26a0 No camera in game_config.json matched '{camera_label}' "
              f"(video: {video_path.name}) - couldn't auto-update it. "
              f"Here's the snippet to paste in manually:")
        print(json.dumps({"team_colors": team_colors}, indent=2))
        return

    backup_path = config_path.with_suffix(".json.backup")
    if not backup_path.exists():
        with open(backup_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nBacked up original config to {backup_path.name}")

    old_colors = matched_cam.get("team_colors")
    matched_cam["team_colors"] = team_colors
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n\u2713 Auto-updated game_config.json - camera '{matched_cam.get('label', camera_label)}' "
          f"team_colors:")
    if old_colors is not None:
        print(f"  Old: {old_colors}")
    print(f"  New: {team_colors}")


if __name__ == "__main__":
    main()

