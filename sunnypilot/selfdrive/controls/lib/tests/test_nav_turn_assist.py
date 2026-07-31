"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pytest

from cereal import log
from openpilot.common.params import Params
from openpilot.sunnypilot.selfdrive.controls.lib.nav_turn_assist import (
  NavTurnAssist, _angle_scale, KAPPA_HARD_MAX, MIN_TURN_ANGLE, UTURN_ANGLE,
)

Desire = log.Desire


def _make(enabled=True):
  Params().put_bool("NkaoudNavTurnAssist", enabled)
  a = NavTurnAssist()
  a.enabled = enabled
  return a


def _settle(a, desire, angle, v_ego, lat_active=True, n=15):
  out = 0.0
  for _ in range(n):
    out = a.update(desire, angle, v_ego, lat_active)
  return out


# ---- _angle_scale ----

@pytest.mark.parametrize("angle,expected", [
  (0.0, 0.0),
  (10.0, 0.0),                       # below MIN
  (MIN_TURN_ANGLE - 0.1, 0.0),
  (45.0, (45.0 - 20.0) / (120.0 - 20.0)),
  (120.0, 1.0),
  (140.0, 1.0),                      # saturates, still under u-turn
  (UTURN_ANGLE, 0.0),                # u-turn excluded
  (180.0, 0.0),
  (-90.0, (90.0 - 20.0) / (120.0 - 20.0)),  # magnitude only
])
def test_angle_scale(angle, expected):
  assert _angle_scale(angle) == pytest.approx(expected)


# ---- gating ----

def test_disabled_returns_zero():
  a = _make(enabled=False)
  assert _settle(a, Desire.turnLeft, -90.0, 12.0) == 0.0


def test_no_turn_desire_returns_zero():
  a = _make()
  assert _settle(a, Desire.none, -90.0, 12.0) == 0.0
  assert _settle(a, Desire.laneChangeLeft, -90.0, 12.0) == 0.0


def test_lat_inactive_zeroes_immediately():
  a = _make()
  _settle(a, Desire.turnLeft, -90.0, 12.0)     # build up a bias
  assert a._bias != 0.0
  out = a.update(Desire.turnLeft, -90.0, 12.0, lat_active=False)
  assert out == 0.0
  assert a._bias == 0.0                         # not merely slewing -- hard zero


def test_shallow_turn_returns_zero():
  a = _make()
  assert _settle(a, Desire.turnLeft, -10.0, 12.0) == 0.0


def test_uturn_excluded():
  a = _make()
  assert _settle(a, Desire.turnLeft, -170.0, 12.0) == 0.0
  assert a.reason == "uturn"


def test_below_min_speed_returns_zero():
  a = _make()
  assert _settle(a, Desire.turnLeft, -90.0, 1.0) == 0.0


# ---- direction ----

def test_turn_left_is_negative():
  a = _make()
  out = _settle(a, Desire.turnLeft, -90.0, 12.0)
  assert out < 0.0
  assert a.active


def test_turn_right_is_positive():
  a = _make()
  out = _settle(a, Desire.turnRight, 90.0, 12.0)
  assert out > 0.0
  assert a.active


def test_sign_mismatch_suppressed():
  # turnLeft desire but the route angle says right -> ambiguous, don't push.
  a = _make()
  assert _settle(a, Desire.turnLeft, 90.0, 12.0) == 0.0
  assert a.reason == "sign mismatch"


# ---- geometry / speed scaling ----

def test_sharper_turn_pushes_harder():
  # Same speed, high enough that neither saturates the hard cap.
  shallow = _make()
  sharp = _make()
  out_shallow = abs(_settle(shallow, Desire.turnLeft, -45.0, 15.0))
  out_sharp = abs(_settle(sharp, Desire.turnLeft, -120.0, 15.0))
  assert out_sharp > out_shallow > 0.0


def test_lower_speed_pushes_harder():
  # a_lat / v^2 => same turn is a tighter curvature at lower speed.
  fast = _make()
  slow = _make()
  out_fast = abs(_settle(fast, Desire.turnRight, 90.0, 20.0))
  out_slow = abs(_settle(slow, Desire.turnRight, 90.0, 8.0))
  assert out_slow > out_fast > 0.0


def test_hard_cap_respected():
  a = _make()
  # Low speed + sharp turn drives the budget past the hard cap.
  out = abs(_settle(a, Desire.turnRight, 120.0, 4.0, n=40))
  assert out <= KAPPA_HARD_MAX + 1e-9


def test_ramp_out_on_desire_clear():
  a = _make()
  peak = abs(_settle(a, Desire.turnLeft, -90.0, 12.0))
  assert peak > 0.0
  # Desire clears (turn done) while lateral stays active -> slews back to 0.
  out = a.update(Desire.none, 0.0, 12.0, lat_active=True)
  assert abs(out) < peak
