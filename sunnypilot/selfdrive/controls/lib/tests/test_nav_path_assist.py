"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

import pytest

from cereal import log
from openpilot.common.params import Params
from openpilot.sunnypilot.selfdrive.controls.lib.nav_path_assist import (
  NavPathAssist, curvature_from_path, _proximity_ramp,
  W_MAX, KAPPA_PATH_MAX, ENGAGE_DIST_M, K_TRUST, NAV_PATH_SPACING_M, TURN_LENGTH_M,
)

Desire = log.Desire


def _arc_path(radius_m, sign, n=30, spacing=NAV_PATH_SPACING_M):
  """Device-frame points along a constant-radius arc from the origin heading
  +x. sign=+1 curves right (-y), sign=-1 curves left (+y). Curvature = 1/R."""
  xs, ys = [], []
  for i in range(n):
    phi = (i * spacing) / radius_m
    xs.append(radius_m * math.sin(phi))
    ys.append(-sign * radius_m * (1.0 - math.cos(phi)))
  return xs, ys

# A model curvature that agrees in sign so the safety veto never trips in the
# generic cases (turnLeft => negative curvature, turnRight => positive).
AGREE = {Desire.turnLeft: -0.01, Desire.turnRight: 0.01}


def _make(enabled=True, trajectory=False):
  Params().put_bool("NkaoudNavPathAssist", enabled)
  Params().put_bool("NkaoudNavPathTrajectory", trajectory)
  a = NavPathAssist()
  a.enabled = enabled
  a.trajectory = trajectory
  return a


def _settle(a, desire, angle, dist, v_ego, kappa_model=None, lat_active=True, n=30,
            path_x=None, path_y=None, path_valid=False):
  if kappa_model is None:
    kappa_model = AGREE.get(desire, 0.0)
  out = (0.0, 0.0)
  for _ in range(n):
    out = a.update(desire, angle, dist, v_ego, kappa_model, lat_active,
                   path_x, path_y, path_valid)
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


# ---- curvature_from_path ----

def test_curvature_from_path_straight_is_zero():
  xs = [float(i * NAV_PATH_SPACING_M) for i in range(30)]
  ys = [0.0] * 30
  assert curvature_from_path(xs, ys, center_m=20.0, margin_m=25.0) == pytest.approx(0.0, abs=1e-6)


def test_curvature_from_path_recovers_radius():
  # A 20 m radius arc -> curvature 0.05 /m, measured within the window.
  xs, ys = _arc_path(20.0, sign=+1)
  k = curvature_from_path(xs, ys, center_m=20.0, margin_m=25.0)
  assert k == pytest.approx(0.05, rel=0.1)


def test_curvature_from_path_too_short():
  assert curvature_from_path([0.0, 1.0], [0.0, 0.0], center_m=5.0) == 0.0


def test_curvature_from_path_window_excludes_far_bend():
  # Straight for the near window, a sharp bend only far away -> a near window
  # sees nothing.
  xs = [float(i * NAV_PATH_SPACING_M) for i in range(30)]
  ys = [0.0] * 20 + [0.5 * (i - 19) ** 2 for i in range(20, 30)]
  assert curvature_from_path(xs, ys, center_m=6.0, margin_m=8.0) == pytest.approx(0.0, abs=1e-6)


# ---- path magnitude vs angle fallback ----

def test_uses_measured_path_when_valid():
  # turnRight, gentle 10 deg angle (angle synthesis would be tiny) but the real
  # path is a tight 12 m arc -> the measured curvature must dominate.
  a = _make()
  xs, ys = _arc_path(12.0, sign=+1)
  curv, w = _settle(a, Desire.turnRight, 30.0, 6.0, 12.0,
                    path_x=xs, path_y=ys, path_valid=True)
  assert a.source == "path"
  assert w > 0.0
  # measured ~1/12 = 0.083, well above the angle synthesis for 30 deg (~0.026).
  assert abs(curv) > math.radians(30.0) / TURN_LENGTH_M


def test_falls_back_to_angle_without_path():
  a = _make()
  _settle(a, Desire.turnRight, 90.0, 2.0, 12.0)
  assert a.source == "angle"


def test_falls_back_to_angle_when_path_straight():
  a = _make()
  xs = [float(i * NAV_PATH_SPACING_M) for i in range(30)]
  ys = [0.0] * 30
  _settle(a, Desire.turnRight, 90.0, 2.0, 12.0, path_x=xs, path_y=ys, path_valid=True)
  assert a.source == "angle"


def test_measured_path_direction_still_from_desire():
  # Path curves right, but the desire says left -> sign follows the desire, and
  # the sign-vs-angle gate is unaffected (angle -90 agrees with turnLeft).
  a = _make()
  xs, ys = _arc_path(15.0, sign=+1)   # geometry curves right
  curv, w = _settle(a, Desire.turnLeft, -90.0, 6.0, 12.0,
                    path_x=xs, path_y=ys, path_valid=True)
  assert w > 0.0
  assert curv < 0.0                    # left, from the desire, not the path shape


# ---- trajectory (pure pursuit) mode ----

def test_trajectory_mode_selected():
  a = _make(trajectory=True)
  xs, ys = _arc_path(20.0, sign=+1)
  curv, w = _settle(a, Desire.turnRight, 60.0, 6.0, 12.0, path_x=xs, path_y=ys, path_valid=True)
  assert a.source == "trajectory"
  assert w > 0.0
  assert curv > 0.0                    # right


def test_trajectory_left_negative():
  a = _make(trajectory=True)
  xs, ys = _arc_path(20.0, sign=-1)    # curves left
  curv, w = _settle(a, Desire.turnLeft, -60.0, 6.0, 12.0, path_x=xs, path_y=ys, path_valid=True)
  assert a.source == "trajectory"
  assert curv < 0.0
  assert w > 0.0


def test_trajectory_recovers_arc_curvature():
  # Pure pursuit on a constant-radius arc recovers ~1/R independent of speed.
  a = _make(trajectory=True)
  xs, ys = _arc_path(20.0, sign=+1, n=40)
  curv, _ = _settle(a, Desire.turnRight, 60.0, 2.0, 12.0, path_x=xs, path_y=ys, path_valid=True)
  assert abs(curv) == pytest.approx(0.05, rel=0.2)


def test_trajectory_side_gate_stays_off():
  # Path bends right but desire says left -> pure-pursuit side gate rejects, and
  # trajectory mode does NOT fall back to the proximity arc; the assist stays off
  # rather than steering the wrong way.
  a = _make(trajectory=True)
  xs, ys = _arc_path(20.0, sign=+1)   # curves right
  curv, w = _settle(a, Desire.turnLeft, -60.0, 6.0, 12.0, path_x=xs, path_y=ys, path_valid=True)
  assert w == 0.0
  assert a.source != "trajectory"


def test_trajectory_far_corner_does_not_engage():
  # Straight for the first 24 m, corner only beyond that. The lookahead point is
  # still straight, so pure pursuit stays below the engage threshold and the
  # assist commands nothing -- this is the fix for leading the turn early.
  a = _make(trajectory=True)
  n = 40
  xs = [float(i * 2.0) for i in range(n)]
  ys = [0.0] * 12 + [-0.15 * (i - 11) ** 2 for i in range(12, n)]  # bends right past ~24 m
  curv, w = _settle(a, Desire.turnRight, 60.0, 26.0, 4.0, path_x=xs, path_y=ys, path_valid=True)
  assert w == 0.0
  assert a.reason == "waiting for bend"


def test_trajectory_no_fallback_on_path_dropout():
  # With trajectory on but the path invalid (GPS dropout), the assist stays off
  # rather than falling through to the proximity arc/angle modes -- which would
  # re-introduce the early turn.
  a = _make(trajectory=True)
  curv, w = _settle(a, Desire.turnRight, 90.0, 20.0, 12.0, path_valid=False)
  assert w == 0.0
  assert a.reason == "no path"
  assert a.source != "angle"


def test_trajectory_engages_when_bend_enters_lookahead():
  # Same geometry, but now the bend is close enough to fall inside the lookahead
  # -> pure pursuit engages. Contrast with the far case above (same path, no
  # engage), which is the early-turn fix.
  a = _make(trajectory=True)
  xs, ys = _arc_path(15.0, sign=+1)   # curves right from the start
  curv, w = _settle(a, Desire.turnRight, 60.0, 4.0, 12.0, path_x=xs, path_y=ys, path_valid=True)
  assert a.source == "trajectory"
  assert w > 0.0
  assert curv > 0.0


def test_trajectory_capped():
  a = _make(trajectory=True)
  xs, ys = _arc_path(6.0, sign=+1)    # very tight -> large curvature
  curv, _ = _settle(a, Desire.turnRight, 120.0, 0.0, 6.0, path_x=xs, path_y=ys, path_valid=True)
  assert abs(curv) <= KAPPA_PATH_MAX + 1e-9
