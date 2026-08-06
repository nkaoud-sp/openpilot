"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Tests for the device-frame route path sampling used by Nav Path Assist (mode B).
"""
import math

import pytest

from openpilot.sunnypilot.nkaoud_nav.geometry import Coordinate, local_xy, sample_route_ahead

# ~metres per degree near the equator, for building synthetic routes.
M_PER_DEG_LAT = math.radians(1.0) * 6371007.2


def _north_of(origin: Coordinate, metres: float) -> Coordinate:
  return Coordinate(origin.latitude + metres / M_PER_DEG_LAT, origin.longitude)


def _east_of(origin: Coordinate, metres: float) -> Coordinate:
  dlon = metres / (M_PER_DEG_LAT * math.cos(math.radians(origin.latitude)))
  return Coordinate(origin.latitude, origin.longitude + dlon)


# ---- local_xy ----

def test_local_xy_point_ahead():
  o = Coordinate(0.0, 0.0)
  # 10 m north, heading north (bearing 0) -> straight ahead (x=10, y=0).
  x, y = local_xy(o, _north_of(o, 10.0), 0.0)
  assert x == pytest.approx(10.0, abs=0.1)
  assert y == pytest.approx(0.0, abs=0.1)


def test_local_xy_east_is_right_when_heading_north():
  o = Coordinate(0.0, 0.0)
  # 10 m east, heading north -> to the right, so y (left) is negative.
  x, y = local_xy(o, _east_of(o, 10.0), 0.0)
  assert x == pytest.approx(0.0, abs=0.1)
  assert y == pytest.approx(-10.0, abs=0.1)


def test_local_xy_rotates_with_heading():
  o = Coordinate(0.0, 0.0)
  # 10 m north, but heading east (bearing 90) -> the point is now to the left.
  x, y = local_xy(o, _north_of(o, 10.0), 90.0)
  assert x == pytest.approx(0.0, abs=0.1)
  assert y == pytest.approx(10.0, abs=0.1)


# ---- sample_route_ahead ----

def test_sample_route_ahead_straight_north():
  o = Coordinate(0.0, 0.0)
  geom = [o, _north_of(o, 20.0), _north_of(o, 60.0)]
  pts = sample_route_ahead(geom, o, 0.0, horizon_m=40.0, spacing_m=2.0)
  assert len(pts) >= 5
  # All roughly on the +x axis (straight ahead), y ~ 0, x increasing.
  assert all(abs(y) < 0.5 for _, y in pts)
  xs = [x for x, _ in pts]
  assert xs == sorted(xs)
  assert xs[-1] == pytest.approx(40.0, abs=3.0)


def test_sample_route_ahead_needs_heading():
  o = Coordinate(0.0, 0.0)
  geom = [o, _north_of(o, 20.0)]
  assert sample_route_ahead(geom, o, None) == []


def test_sample_route_ahead_empty_geometry():
  assert sample_route_ahead([], Coordinate(0.0, 0.0), 0.0) == []


def test_sample_route_ahead_right_turn_bends_negative_y():
  # An L: 30 m north then 30 m east. Heading north, the eastward leg is a right
  # turn, so far points must go to negative y (right).
  o = Coordinate(0.0, 0.0)
  corner = _north_of(o, 30.0)
  end = _east_of(corner, 30.0)
  pts = sample_route_ahead([o, corner, end], o, 0.0, horizon_m=60.0, spacing_m=2.0)
  assert pts[-1][1] < -1.0
