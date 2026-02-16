from __future__ import annotations

import argparse
from pathlib import Path

from .tracking import DetectorConfig, run_video_line_judge


def _parse_corner_string(corner_str: str) -> list[tuple[float, float]]:
    points = []
    for pair in corner_str.strip().split():
        x_str, y_str = pair.split(",")
        points.append((float(x_str), float(y_str)))
    if len(points) != 4:
        raise ValueError("Expected exactly four corners: 'x1,y1 x2,y2 x3,y3 x4,y4'")
    return points


def _parse_hsv(hsv_str: str) -> tuple[int, int, int]:
    values = tuple(int(v.strip()) for v in hsv_str.split(","))
    if len(values) != 3:
        raise ValueError("HSV must contain 3 comma-separated integers.")
    return values


def _collect_corners_interactively(video_path: Path) -> list[tuple[float, float]]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read first video frame for corner selection.")

    clicks: list[tuple[float, float]] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((float(x), float(y)))

    window = "Click corners: near-left, near-right, far-right, far-left"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        display = frame.copy()
        for i, pt in enumerate(clicks):
            cv2.circle(display, (int(pt[0]), int(pt[1])), 7, (0, 255, 0), -1)
            cv2.putText(display, str(i + 1), (int(pt[0]) + 8, int(pt[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(
            display,
            "Click 4 corners in order. Press ENTER to confirm.",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.imshow(window, display)
        key = cv2.waitKey(10) & 0xFF
        if key in (13, 10) and len(clicks) == 4:
            break
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Corner selection canceled.")

    cv2.destroyAllWindows()
    return clicks


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pickleball line judge MVP")
    p.add_argument("--video", type=Path, required=True, help="Input video path")
    p.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory to write outputs")
    p.add_argument("--corners", type=str, default=None, help='Corner list: "x1,y1 x2,y2 x3,y3 x4,y4"')
    p.add_argument("--interactive-corners", action="store_true", help="Pick corners by clicking first frame")
    p.add_argument("--hsv-lower", type=str, default="20,70,70", help="HSV lower threshold")
    p.add_argument("--hsv-upper", type=str, default="40,255,255", help="HSV upper threshold")
    p.add_argument("--min-contour-area", type=int, default=10, help="Minimum contour area")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.interactive_corners:
        corners = _collect_corners_interactively(args.video)
    elif args.corners:
        corners = _parse_corner_string(args.corners)
    else:
        raise ValueError("Provide --corners or use --interactive-corners.")

    cfg = DetectorConfig(
        hsv_lower=_parse_hsv(args.hsv_lower),
        hsv_upper=_parse_hsv(args.hsv_upper),
        min_contour_area=args.min_contour_area,
    )

    video_out, csv_out = run_video_line_judge(
        video_path=args.video,
        image_corners=corners,
        output_dir=args.output_dir,
        detector_cfg=cfg,
    )
    print(f"Done. Annotated video: {video_out}")
    print(f"Detections CSV: {csv_out}")


if __name__ == "__main__":
    main()
