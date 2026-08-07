"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Tests for the device-frame route path sampling used by Nav Path Assist (mode B).
"""
import math

import pytest

from openpilot.sunnypilot.nkaoud_nav.geometry import (
  Coordinate, local_xy, maneuver_passed, sample_points_ahead, sample_route_ahead,
)

# ~metres per degree near the equator, for building synthetic routes.
M_PER_DEG_LAT = math.radians(1.0) * 6371007.2


def _north_of(origin: Coordinate, metres: float) -> Coordinate:
  return Coordinate(origin.latitude + metres / M_PER_DEG_LAT, origin.longitude)


def _east_of(origin: Coordinate, metres: float) -> Coordinate:
  dlon = metres / (M_PER_DEG_LAT * math.cos(math.radians(origin.latitude)))
  return Coordinate(origin.latitude, origin.longitude + dlon)


def _offset(origin: Coordinate, north_m: float, east_m: float) -> Coordinate:
  return _east_of(_north_of(origin, north_m), east_m)


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


# ---- maneuver_passed (step-transition detector) ----

CAPTURE = 40.0
HYST = 10.0


def _run_pass(distances):
  """Feed a sequence of straight-line distances (m) to the maneuver point and
  return the tick index where `passed` first fires, or None."""
  mp = Coordinate(0.0, 0.0)
  mn = float("inf")
  for i, d in enumerate(distances):
    passed, mn = maneuver_passed(mp, _north_of(mp, d), mn, CAPTURE, HYST)
    if passed:
      return i
  return None


def test_maneuver_passed_on_departure():
  # Approach to 5 m then drive away. min=5; d=15 is +10 (== HYST, no), d=20 is
  # +15 (> HYST) -> fires at index 5.
  assert _run_pass([50, 30, 15, 5, 15, 20]) == 5


def test_maneuver_passed_is_permissive_on_dip_then_rise():
  # By design maneuver_passed only sees distances, so a min-latching dip followed
  # by a > HYST rise DOES fire even if the vehicle has not truly passed the
  # maneuver (e.g. GPS multipath, or a jog in the approach). navd constrains this
  # by additionally requiring the next step's geometry to be the closer one
  # (on_next) before advancing; this test pins the raw detector's contract so a
  # CAPTURE/HYST tuning change is caught.
  assert _run_pass([38, 25, 38]) is not None


def test_maneuver_passed_cut_corner_floor():
  # Log scenario: closest straight-line approach ~18 m (corner cut wide, never
  # within the old 10 m along-track threshold), then departs. Must still fire.
  assert _run_pass([60, 40, 25, 18, 22, 30, 45]) is not None


def test_maneuver_passed_requires_capture():
  # Never comes within capture (closest 55 m) -> never fires even while leaving.
  assert _run_pass([80, 70, 60, 55, 70, 90]) is None


def test_maneuver_passed_ignores_jitter():
  # Sitting near the maneuver with sub-hysteresis GPS jitter must not fire.
  assert _run_pass([12, 10, 11, 9, 12, 10, 13]) is None


def test_sample_route_ahead_snaps_to_nearest_earlier_segment():
  # Interchange shape: an "on-ramp" segment 5 m ahead running east, then the
  # route folds to an "off-ramp" continuing straight ahead 20 m out. The vehicle
  # is nearest the on-ramp, so sampling the WHOLE route starts there and the near
  # path bends east (toward the wrong ramp) -- the backward-snap navd avoids by
  # anchoring to the current step.
  o = Coordinate(0.0, 0.0)
  on_ramp = [_north_of(o, 5.0), _offset(o, 5.0, 30.0)]         # 5 m ahead, runs east
  off_ramp = [_north_of(o, 20.0), _north_of(o, 60.0)]          # straight ahead, far
  whole = on_ramp + off_ramp

  pts_whole = sample_route_ahead(whole, o, 0.0, horizon_m=15.0, spacing_m=2.0)
  # near path bends to the right (east -> negative y) toward the on-ramp
  assert min(y for _, y in pts_whole) < -5.0

  pts_step = sample_route_ahead(off_ramp, o, 0.0, horizon_m=40.0, spacing_m=2.0)
  # current-step-onward: straight ahead, no eastward excursion
  assert all(abs(y) < 1.0 for _, y in pts_step)
  assert pts_step[-1][0] > pts_step[0][0]                      # advances forward


def test_sample_points_ahead_resists_downstream_fold():
  # Cloverleaf shape: the anchored forward polyline starts just ahead, runs north,
  # then a downstream leg folds back to within a few metres of the vehicle and
  # heads east. sample_route_ahead's nearest search snaps the start to that fold
  # (near path bends east); sample_points_ahead is anchored to world[0] and does
  # no search, so it stays forward.
  o = Coordinate(0.0, 0.0)
  world = [
    _north_of(o, 5.0),            # anchored start, just ahead
    _north_of(o, 30.0),           # forward leg (north)
    _offset(o, 2.0, 3.0),         # folds back to ~3.6 m from the vehicle
    _offset(o, 2.0, 40.0),        # then runs east
  ]
  pts_search = sample_route_ahead(world, o, 0.0, horizon_m=40.0, spacing_m=2.0)
  assert min(y for _, y in pts_search) < -3.0                  # snapped to the fold, bends east

  pts_anchor = sample_points_ahead(world, o, 0.0, horizon_m=40.0, spacing_m=2.0)
  assert all(abs(y) < 1.0 for _, y in pts_anchor[:10])         # stays forward, no snap
  assert pts_anchor[-1][0] > pts_anchor[0][0]


def test_sample_points_ahead_needs_two_points():
  o = Coordinate(0.0, 0.0)
  assert sample_points_ahead([_north_of(o, 5.0)], o, 0.0) == []
  assert sample_points_ahead([], o, 0.0) == []


def test_sample_route_ahead_right_turn_bends_negative_y():
  # An L: 30 m north then 30 m east. Heading north, the eastward leg is a right
  # turn, so far points must go to negative y (right).
  o = Coordinate(0.0, 0.0)
  corner = _north_of(o, 30.0)
  end = _east_of(corner, 30.0)
  pts = sample_route_ahead([o, corner, end], o, 0.0, horizon_m=60.0, spacing_m=2.0)
  assert pts[-1][1] < -1.0
