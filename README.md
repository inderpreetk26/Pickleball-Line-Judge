# Pickleball Line Judge (MVP)

This repository now contains a **starter computer-vision app** that:

1. Maps a pickleball court from camera coordinates into real-world court coordinates.
2. Tracks the ball in each frame.
3. Labels each detected bounce candidate as **IN** or **OUT** based on court bounds.

> ⚠️ This is an MVP baseline intended to get you from idea to a working prototype quickly. For tournament-grade accuracy, you'll want multiple cameras, lens calibration, and a learned detector.

## Project layout

- `src/pickleball_line_judge/geometry.py` – homography + court boundary logic
- `src/pickleball_line_judge/tracking.py` – basic HSV ball detector and frame annotation pipeline
- `src/pickleball_line_judge/main.py` – CLI entrypoint
- `tests/test_geometry.py` – unit tests for mapping and line-judge logic

## Quick start

### 1) Create environment and install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2) Run on a video

```bash
pickleball-line-judge --video path/to/video.mp4 --interactive-corners
```

You will click the four court corners on the first frame in this order:

1. Near left
2. Near right
3. Far right
4. Far left

The app writes:

- Annotated video: `output/annotated.mp4`
- Frame-by-frame decisions: `output/detections.csv`

## CLI options

```bash
pickleball-line-judge --video input.mp4 --output-dir output --interactive-corners
```

Optional flags:

- `--corners "x1,y1 x2,y2 x3,y3 x4,y4"` to pass corners without clicking.
- `--hsv-lower "20,70,70"` and `--hsv-upper "40,255,255"` to tune yellow ball detection.
- `--min-contour-area 10` to tune noise filtering.

## How decisions are made

- Ball center is detected in image space.
- Center is projected into court coordinates (feet) with a homography.
- If mapped point is inside court rectangle `[0,20] x [0,44]`, the call is `IN`; otherwise `OUT`.

## Next improvements (recommended)

- Replace color-threshold detector with YOLOv8/RT-DETR model fine-tuned on pickleballs.
- Detect true bounce frames (vertical acceleration sign-change + court proximity).
- Add camera calibration (intrinsics + distortion correction).
- Add temporal filtering (Kalman/ByteTrack).
- Support doubles sidelines / kitchen / service rules with configurable line sets.
