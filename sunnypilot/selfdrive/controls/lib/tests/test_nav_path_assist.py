"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pytest

from cereal import log
from openpilot.common.params import Params
from openpilot.sunnypilot.selfdrive.controls.lib.nav_path_assist import (
  NavPathAssist, _proximity_ramp, W_MAX, KAPPA_PATH_MAX, ENGAGE_DIST_M, K_TRUST,
)

Desire = log.Desire

# A model curvature that agrees in sign so the safety veto never trips in the
# generic cases (turnLeft => negative curvature, turnRight => positive).
AGREE = {Desire.turnLeft: -0.01, Desire.turnRight: 0.01}


def _make(enabled=True):
  Params().put_bool("NkaoudNavPathAssist", enabled)
  a = NavPathAssist()
  a.enabled = enabled
  return a


def _settle(a, desire, angle, dist, v_ego, kappa_model=None, lat_active=True, n=30):
  if kappa_model is None:
    kappa_model = AGREE.get(desire, 0.0)
  out = (0.0, 0.0)
  for _ in range(n):
    out = a.update(desire, angle, dist, v_ego, kappa_model, lat_active)
  return out  # (curvature, weight)


# ---- _proximity_ramp ----

@pytest.mark.parametrize("dist,expected", [
  (ENGAGE_DIST_M, 0.0),
  (ENGAGE_DIST_M + 10.0, 0.0),
  (0.0, 1.0),
  (-5.0, 1.0),                       # past the maneuver point still full
  (ENGAGE_DIST_M / 2.0, 0.5),
])
def test_proximity_ramp(dist, expected):
  assert _proximity_ramp(dist) == pytest.approx(expected)


# ---- gating (weight must be 0) ----

def test_disabled_returns_zero():
  a = _make(enabled=False)
  assert _settle(a, Desire.turnLeft, -90.0, 5.0, 12.0) == (0.0, 0.0)


def test_no_turn_desire_returns_zero():
  a = _make()
  assert _settle(a, Desire.none, -90.0, 5.0, 12.0)[1] == 0.0
  assert _settle(a, Desire.laneChangeLeft, -90.0, 5.0, 12.0)[1] == 0.0


def test_lat_inactive_zeroes_immediately():
  a = _make()
  _settle(a, Desire.turnLeft, -90.0, 5.0, 12.0)
  assert a._weight != 0.0
  curv, w = a.update(Desire.turnLeft, -90.0, 5.0, 12.0, AGREE[Desire.turnLeft], lat_active=False)
  assert (curv, w) == (0.0, 0.0)
  assert a._weight == 0.0            # hard zero, not merely slewing


def test_shallow_turn_returns_zero():
  a = _make()
  assert _settle(a, Desire.turnLeft, -10.0, 5.0, 12.0)[1] == 0.0


def test_uturn_excluded():
  a = _make()
  assert _settle(a, Desire.turnLeft, -170.0, 5.0, 12.0)[1] == 0.0
  assert a.reason == "uturn"


def test_below_min_speed_returns_zero():
  a = _make()
  assert _settle(a, Desire.turnLeft, -90.0, 5.0, 1.0)[1] == 0.0


def test_far_maneuver_contributes_nothing():
  # Beyond ENGAGE_DIST there is no path authority even mid-turn-desire.
  a = _make()
  curv, w = _settle(a, Desire.turnLeft, -90.0, ENGAGE_DIST_M + 5.0, 12.0)
  assert (curv, w) == (0.0, 0.0)
  assert a.reason == "far"


# ---- direction & geometry ----

def test_turn_left_is_negative():
  a = _make()
  curv, w = _settle(a, Desire.turnLeft, -90.0, 2.0, 12.0)
  assert curv < 0.0
  assert w > 0.0
  assert a.active


def test_turn_right_is_positive():
  a = _make()
  curv, w = _settle(a, Desire.turnRight, 90.0, 2.0, 12.0)
  assert curv > 0.0
  assert w > 0.0
  assert a.active


def test_sign_mismatch_suppressed():
  # turnLeft desire but the route angle says right -> ambiguous, don't blend.
  a = _make()
  assert _settle(a, Desire.turnLeft, 90.0, 5.0, 12.0)[1] == 0.0
  assert a.reason == "sign mismatch"


def test_curvature_speed_independent():
  # Same corner, different speed -> same geometric curvature target (unlike the
  # a_lat/v^2 mode-A nudge). Weight/curvature depend on proximity, not speed.
  slow = _make()
  fast = _make()
  curv_slow, _ = _settle(slow, Desire.turnRight, 90.0, 2.0, 8.0)
  curv_fast, _ = _settle(fast, Desire.turnRight, 90.0, 2.0, 20.0)
  assert curv_slow == pytest.approx(curv_fast)


def test_sharper_turn_has_more_curvature():
  # Angles chosen below the KAPPA_PATH_MAX cap so neither saturates.
  shallow = _make()
  sharp = _make()
  curv_shallow, _ = _settle(shallow, Desire.turnLeft, -30.0, 2.0, 12.0)
  curv_sharp, _ = _settle(sharp, Desire.turnLeft, -90.0, 2.0, 12.0)
  assert abs(curv_sharp) > abs(curv_shallow) > 0.0


def test_closer_maneuver_more_authority():
  near = _make()
  far = _make()
  _, w_near = _settle(near, Desire.turnRight, 90.0, 3.0, 12.0)
  _, w_far = _settle(far, Desire.turnRight, 90.0, ENGAGE_DIST_M * 0.75, 12.0)
  assert w_near > w_far > 0.0


def test_hard_cap_respected():
  a = _make()
  curv, _ = _settle(a, Desire.turnRight, 140.0, 0.0, 12.0)
  assert abs(curv) <= KAPPA_PATH_MAX + 1e-9


def test_weight_capped():
  a = _make()
  _, w = _settle(a, Desire.turnRight, 120.0, 0.0, 12.0, n=60)
  assert w <= W_MAX + 1e-9


# ---- safety veto ----

def test_model_disagreement_vetoes_weight():
  # turnRight route, but the model is steering hard left (avoiding something the
  # route can't see) -> weight collapses to 0, model keeps authority.
  a = _make()
  curv, w = _settle(a, Desire.turnRight, 90.0, 2.0, 12.0, kappa_model=-0.05)
  assert w == 0.0
  assert a.reason == "model disagrees"


def test_small_opposing_model_curvature_not_vetoed():
  # A tiny opposing model curvature (below K_TRUST) is ~straight; a mild lean is
  # still allowed.
  a = _make()
  _, w = _settle(a, Desire.turnRight, 90.0, 2.0, 12.0, kappa_model=-K_TRUST / 2.0)
  assert w > 0.0


# ---- handoff ----

def test_ramp_out_on_desire_clear():
  a = _make()
  _, peak = _settle(a, Desire.turnLeft, -90.0, 2.0, 12.0)
  assert peak > 0.0
  _, w = a.update(Desire.none, 0.0, 2.0, 12.0, 0.0, lat_active=True)
  assert w < peak
