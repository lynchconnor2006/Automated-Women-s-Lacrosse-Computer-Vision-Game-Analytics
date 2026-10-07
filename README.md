# Lacrosse Computer Vision Pipeline

A multi-camera computer vision system that turns raw women's lacrosse game footage into structured analytics — possession detection, formations, and player-movement patterns — without relying on manual film review or ball-tracking hardware.

The pipeline fine-tunes a YOLO object detection model on in-game footage, tracks players and goalies across two broadcast camera angles, reconstructs real field geometry from manual calibration, and independently detects possession changes directly from player positioning — cross-validated against (but never dependent on) the official play-by-play record.

---

## What it does

- **Detects and tracks players/goalies** across two fixed camera angles using a custom fine-tuned YOLO model
- **Reconstructs field geometry** from 8 manually-calibrated reference points, including a true curved key-arc (not a simplified box), to classify player position into real field zones
- **Detects possessions directly from video signal** — player headcounts and positioning over time — rather than inheriting timing from the official game log
- **Validates against the official play-by-play** as an independent check (does a detected possession exist near each logged one?) without ever letting the log override video-derived boundaries
- **Analyzes formations and cuts** per possession, broken out by team and by possession outcome
- **Supports manual possession tagging** (goal, save, turnover, etc.) via an interactive video-review tool
- **Exports a coach-ready Excel scouting report** and an annotated highlight clip of the pipeline's most reliably-tracked possession

---

## Why build it this way

Most of the hard problems in this project came from two real-world constraints that don't show up in a toy dataset:

1. **The official game clock and the video are two independent, imperfectly-synced timelines.** A hand-logged game clock stops on every whistle; the video doesn't. The pipeline includes a clock-alignment system (anchored on real, visually-detectable events like goal celebrations and stoppages) with automatic anchor validation to catch and discard individual bad anchor points before they distort a whole quarter's timing.

2. **A player-count-based possession detector has a real, information-theoretic limit**: it knows *who's on the field*, not *who has the ball*. A fast shot/rebound sequence where neither team's players actually leave the attacking zone is genuinely invisible to this method. Rather than pretend otherwise, the pipeline explicitly flags these cases (`boundary_source` tags, zero-cut warnings) so downstream analysis can weight them with appropriate confidence instead of treating every possession as equally certain.

---

## Repository structure

```
Summer Attempt/
├── games/
│   └── <GameName>/                 # one folder per game
│       ├── game_config.json        # all per-game settings
│       ├── <Camera1>.mp4
│       ├── <Camera2>.mp4
│       ├── playbyplay_....txt
│       └── (everything the pipeline produces)
│
└── python/
    ├── run_pipeline.py                      # master runner, Steps 4-8
    ├── tag_possessions.py                   # Step 9 — manual tagging
    ├── export_to_excel.py                   # Step 10 — scouting report
    ├── render_best_possession_video.py      # Step 10 — annotated clip
    ├── pipeline_common.py                   # shared config/geometry helpers
    │
    ├── pre_colab/
    │   ├── setup_only.py                    # field calibration + quarter timestamps
    │   ├── pre_colab_pick_players.py         # jersey color click-sampling
    │   ├── pre_colab_compute_team_colors.py  # HSV range computation
    │   ├── pre_colab_pick_exclusion_zones.py # false-positive zone marking
    │   └── pre_colab_set_camera_side.py
    │
    └── (diagnostic/validation tools — see Troubleshooting below)
```

Colab notebooks (`tracker_gpu.ipynb` and the model fine-tuning notebook) run detection and training separately and aren't part of the local repo's execution path — see **Workflow** below for where they fit in.

---

## Workflow

Three phases per game: **pre-Colab setup** (local, no GPU) → **Colab detection** (GPU, once per camera) → **local pipeline** (tracking through final report).

A full phase-by-phase guide with every exact terminal command is in [`Lacrosse_Pipeline_Quick_Reference.pdf`](./Lacrosse_Pipeline_Quick_Reference.pdf). Condensed version:

| Phase | What happens |
|---|---|
| 0 | `create_game_config.py` scaffolds a new game folder |
| 1 (pre-Colab) | Field calibration, quarter timestamps, jersey-color sampling, exclusion zones |
| 2 (Colab) | YOLO detection + team classification, per camera |
| 3 (local) | `run_pipeline.py` — tracking → goalie ID → clock alignment → possession detection → formations → cuts |
| 4 | `tag_possessions.py` — manual outcome tagging per possession |
| 5 | `export_to_excel.py` + `render_best_possession_video.py` — final deliverables |

Re-run any stage with `--force` to ignore cached outputs (needed after changing team colors, re-running Colab, or adjusting detection thresholds).

---

## Model

The detector is a fine-tuned YOLOv8 model (3 classes: Player, Referee, Goalie), fine-tuned from a general base model using annotated frames from real game footage. Fine-tuning used a frozen-backbone transfer-learning approach appropriate for a small, growing dataset, with low-detection-count frames specifically targeted for annotation to improve recall on the model's known weak spots (`find_obscure_frames.py` + `annotate_players.py`).

Current validated performance (held-out validation set): **mAP50 ≈ 0.99**, up from ≈ 0.65-0.70 on the original base model, with goalie recall improving from ≈ 0.14 to 1.00 — the class that mattered most for downstream zone classification.

---

## Known limitations

- **Possession detection cannot see fast, same-zone turnovers.** If both teams' player counts stay elevated through a shot/save/clear sequence, the detector has no signal to split the possession. The pipeline recovers what it can by splitting abnormally-long detected possessions against the official record where a hidden opposing-team possession is confirmed to exist inside them (`boundary_source: official_split_within_long_video_block`), but this isn't a substitute for a true ball-tracking signal.
- **Clock alignment is approximate, not exact**, especially in quarters with few recognizable stoppage/goal events. An automatic anchor-validation step discards individual anchors that would distort timing, but alignment precision still varies by quarter.
- **Both cameras are fixed, wide-angle setups.** The pipeline's field-geometry model assumes a static camera; it does not currently support a moving/zooming sideline camera (a planned future direction, requiring a different geometric approach).

---

## Troubleshooting / diagnostic tools

A set of standalone scripts for investigating specific issues (not part of the required pipeline path) — each run the same way, with `--game "GameName"` appended:

| Script | Use case |
|---|---|
| `visualize_possession.py` | Frame-by-frame count time series + sample annotated frames for one possession |
| `check_missing_possession_both_cameras.py` | Side-by-side per-team counts on both cameras for a possession the detector missed |
| `check_official_log_accuracy.py` | Checks whether a "missing" possession traces back to a raw play-by-play transcription issue |
| `check_alignment_scale_factors.py` | Flags quarters/segments where clock alignment pace looks distorted |
| `check_fragment_gaps.py` | Flags short possessions that may really be fragments of an adjacent one |
| `check_full_zone_breakdown.py` | Full per-zone (including Beyond-30) player counts for a time window |
| `find_obscure_frames.py` | Finds low-detection-count stretches for targeted annotation |
| `diff_detected_possessions.py` | Diffs two `detected_possessions.json` snapshots (e.g. before/after a model swap) |

---

## Requirements

- Python 3.x, OpenCV, NumPy
- Ultralytics YOLO (for fine-tuning and local-side zone-aware re-detection logic)
- Google Colab (GPU) for the detection step
- `openpyxl` for Excel report generation

---

## Status

Validated on one complete game end-to-end, including real possession tagging and report generation. Next steps: validating generalization on a second game with the same two-camera setup, and scoping a separate approach for a single moving/zooming sideline camera.
