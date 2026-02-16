from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv

import numpy as np

from .geometry import CourtDimensions, compute_homography, is_in_bounds, map_point_to_court


@dataclass
class DetectorConfig:
    hsv_lower: tuple[int, int, int] = (20, 70, 70)
    hsv_upper: tuple[int, int, int] = (40, 255, 255)
    min_contour_area: int = 10


def detect_ball_center(frame: np.ndarray, cfg: DetectorConfig) -> tuple[int, int] | None:
    """Detect likely ball center with simple HSV threshold + largest contour."""
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(cfg.hsv_lower), np.array(cfg.hsv_upper))

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < cfg.min_contour_area:
        return None

    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return cx, cy


def _draw_overlay(
    frame: np.ndarray,
    center: tuple[int, int] | None,
    court_point: tuple[float, float] | None,
    call: str | None,
) -> np.ndarray:
    import cv2

    out = frame.copy()
    if center is not None:
        cv2.circle(out, center, 8, (0, 255, 255), 2)
    if court_point is not None and call is not None:
        text = f"{call}  court=({court_point[0]:.2f}, {court_point[1]:.2f})"
        color = (0, 200, 0) if call == "IN" else (0, 0, 255)
        cv2.putText(out, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


def run_video_line_judge(
    video_path: Path,
    image_corners: Iterable[tuple[float, float]],
    output_dir: Path,
    detector_cfg: DetectorConfig | None = None,
    court: CourtDimensions | None = None,
) -> tuple[Path, Path]:
    """Process a video and emit annotated video + CSV detections."""
    import cv2

    detector_cfg = detector_cfg or DetectorConfig()
    court = court or CourtDimensions()
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = output_dir / "annotated.mp4"
    output_csv = output_dir / "detections.csv"

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    H = compute_homography(image_corners, court=court)

    with output_csv.open("w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["frame", "pixel_x", "pixel_y", "court_x_ft", "court_y_ft", "call"])

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            center = detect_ball_center(frame, detector_cfg)
            court_point = None
            call = None
            if center is not None:
                court_point = map_point_to_court(center, H)
                call = "IN" if is_in_bounds(court_point, court=court) else "OUT"
                csv_writer.writerow(
                    [
                        frame_idx,
                        center[0],
                        center[1],
                        f"{court_point[0]:.4f}",
                        f"{court_point[1]:.4f}",
                        call,
                    ]
                )

            writer.write(_draw_overlay(frame, center, court_point, call))
            frame_idx += 1

    cap.release()
    writer.release()
    return output_video, output_csv
