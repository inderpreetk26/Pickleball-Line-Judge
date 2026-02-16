import numpy as np
from pickleball_line_judge.geometry import CourtDimensions, is_in_bounds, map_point_to_court


def test_map_point_identity_homography():
    H = np.eye(3)
    x, y = map_point_to_court((5.5, 7.25), H)
    assert x == 5.5
    assert y == 7.25


def test_in_bounds_basic():
    court = CourtDimensions(width_ft=20.0, length_ft=44.0)
    assert is_in_bounds((10.0, 22.0), court=court)
    assert is_in_bounds((0.0, 0.0), court=court)
    assert is_in_bounds((20.0, 44.0), court=court)
    assert not is_in_bounds((-0.01, 10.0), court=court)
    assert not is_in_bounds((10.0, 44.01), court=court)


def test_in_bounds_margin():
    court = CourtDimensions()
    assert is_in_bounds((-0.2, 10.0), court=court, margin_ft=0.25)
    assert not is_in_bounds((-0.3, 10.0), court=court, margin_ft=0.25)
