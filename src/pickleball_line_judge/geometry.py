from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CourtDimensions:
    """Pickleball court dimensions in feet.

    Width: 20 ft
    Length: 44 ft
    """

    width_ft: float = 20.0
    length_ft: float = 44.0


def _to_float32_points(points: Iterable[tuple[float, float]]) -> np.ndarray:
    arr = np.array(list(points), dtype=np.float32)
    if arr.shape != (4, 2):
        raise ValueError("Expected exactly 4 points with shape (4, 2).")
    return arr


def compute_homography(
    image_corners: Iterable[tuple[float, float]],
    court: CourtDimensions | None = None,
) -> np.ndarray:
    """Compute homography mapping image plane -> court plane.

    Corner ordering must be:
    1) near-left, 2) near-right, 3) far-right, 4) far-left.
    """
    court = court or CourtDimensions()
    img = _to_float32_points(image_corners)
    world = np.array(
        [
            [0.0, 0.0],
            [court.width_ft, 0.0],
            [court.width_ft, court.length_ft],
            [0.0, court.length_ft],
        ],
        dtype=np.float32,
    )

    import cv2  # Local import to keep geometry tests lightweight if cv2 isn't installed.

    H = cv2.getPerspectiveTransform(img, world)
    if H is None or H.shape != (3, 3):
        raise RuntimeError("Failed to compute homography matrix.")
    return H


def map_point_to_court(image_point: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    """Project an image-space point into court coordinates."""
    x, y = image_point
    p = np.array([x, y, 1.0], dtype=np.float64)
    mapped = homography @ p
    if np.isclose(mapped[2], 0.0):
        raise ZeroDivisionError("Invalid homography projection; w ~= 0.")
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])


def is_in_bounds(
    court_point: tuple[float, float],
    court: CourtDimensions | None = None,
    margin_ft: float = 0.0,
) -> bool:
    """Return True if court point lies inside the full singles/doubles boundary rectangle."""
    court = court or CourtDimensions()
    x, y = court_point
    return (
        -margin_ft <= x <= court.width_ft + margin_ft
        and -margin_ft <= y <= court.length_ft + margin_ft
    )
