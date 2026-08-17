"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Nav Turn Assist.

A feedforward nudge that helps the driving model execute a route-commanded
turn. It does NOT replace the turn desire: navd still injects turnLeft/turnRight
into the model (the trained maneuver), and this only adds a small, capped
curvature bias in the SAME direction to lean the model through the turn when a
given model executes the desire weakly or inconsistently.

Two design choices make it safe to layer on top of the model:

  * Feedforward, not feedback. The direction comes from the applied nav desire
    and the magnitude from the route's turn angle -- never from the model's own
    (possibly wrong) lane lines/path. So it still helps precisely when the
    model is the thing misbehaving. This is the opposite of Lane Center Assist,
    which is a feedback centering trim and deliberately bails during maneuvers.

  * Consistent with the turn. The route knows a turn is 45 vs 90 vs 120 deg
    (from the Mapbox maneuver bearings), so the assist scales with that angle:
    gentle for a shallow turn, firmer for a sharp one. u-turns are excluded --
    that is too much authority for a feedforward nudge; the model + driver own
    those.

The nudge is a *lateral-accel budget* converted to curvature (a_lat / v^2) so
the feel is speed-consistent, angle-scaled, slew-limited on engage, and hard
capped. It is emitted as a curvature bias for controlsd to add downstream,
where clip_curvature still bounds the total. Off unless NkaoudNavTurnAssist is
enabled.
"""
from __future__ import annotations

import numpy as np

from cereal import log
from openpilot.common.params import Params

Desire = log.Desire

# The nav turn desires this assist reacts to. Direction is taken from here, not
# from any model output.
TURN_DESIRES = (Desire.turnLeft, Desire.turnRight)

# Curvature sign convention matches desiredCurvature: positive steers right (see
# latcontrol_angle.py / lane_center_assist.py). turnRight -> +, turnLeft -> -.
CURV_SIGN = {Desire.turnLeft: -1.0, Desire.turnRight: 1.0}

# Angle band (deg, magnitude). Below MIN the model handles it unaided; the scale
# ramps linearly to 1.0 at FULL so a 120 deg turn gets more lean than a 90. At
# or above UTURN the maneuver is excluded (u-turns are the model's + driver's).
MIN_TURN_ANGLE = 20.0
FULL_TURN_ANGLE = 120.0
UTURN_ANGLE = 150.0

# Lateral-accel budget for the nudge (m/s^2). Deliberately small: this is an
# assist on top of the model's own turn, not the turn itself.
ASSIST_MAX_ACCEL = 0.5

# Hard curvature cap (1/m) regardless of speed -- a floor on turn radius the
# assist can request on its own. clip_curvature downstream is the real bound.
KAPPA_HARD_MAX = 0.02

V_EGO_FLOOR = 5.0            # m/s, floor in a_lat / v^2 so the bias can't blow up
MIN_SPEED = 2.0             # m/s, below this we don't nudge at all (parking/creep)

# Slew toward the target so engagement is smooth. Model runs at ~20 Hz.
BIAS_SLEW = 2.0e-3          # 1/m per frame => ~0.5 s to reach the hard cap

PARAM_READ_PERIOD_FRAMES = 20  # ~1 Hz at the model rate


def _angle_scale(turn_angle_deg: float) -> float:
  """0..1 magnitude scale from the turn angle. 0 below MIN or in the u-turn
  band; ramps linearly MIN->FULL and saturates at 1.0."""
  mag = abs(turn_angle_deg)
  if mag < MIN_TURN_ANGLE or mag >= UTURN_ANGLE:
    return 0.0
  return float(np.clip((mag - MIN_TURN_ANGLE) / (FULL_TURN_ANGLE - MIN_TURN_ANGLE), 0.0, 1.0))


class NavTurnAssist:
  def __init__(self):
    self.params = Params()
    self._frame = 0
    self.enabled = False

    self._bias = 0.0          # slew-limited curvature bias currently applied (1/m)
    self.active = False
    self.reason = "off"
    self.turn_angle = 0.0     # last turn angle seen (deg, signed)

    self._read_params()

  def _read_params(self) -> None:
    self.enabled = self.params.get_bool("NkaoudNavTurnAssist")

  def _slew_to(self, target: float) -> float:
    self._bias = float(np.clip(target, self._bias - BIAS_SLEW, self._bias + BIAS_SLEW))
    return self._bias

  def reset(self) -> None:
    """Hard-clear state when a navigation provider that does not own this
    target-specific curvature assist becomes active."""
    self._bias = 0.0
    self.active = False
    self.reason = "provider"
    self.turn_angle = 0.0

  def update(self, desire, turn_angle_deg: float, v_ego: float, lat_active: bool) -> float:
    """Return a feedforward curvature bias (1/m) to add to desiredCurvature for
    a route-commanded turn. 0 unless enabled, lateral is active, a turn desire
    is applied, and the turn angle is in the assisted band."""
    self._frame += 1
    if self._frame % PARAM_READ_PERIOD_FRAMES == 0:
      self._read_params()

    self.active = False
    self.turn_angle = float(turn_angle_deg)

    # Hard, immediate ramp-out: whenever lateral is not active the assist must
    # not linger. clip_curvature can't save us if we keep commanding a bias
    # while disengaged, so zero it outright rather than slewing.
    if not lat_active:
      self.reason = "not active"
      self._bias = 0.0
      return 0.0

    if not self.enabled:
      self.reason = "off"
      return self._slew_to(0.0)

    if desire not in TURN_DESIRES:
      self.reason = "no turn desire"
      return self._slew_to(0.0)

    if v_ego < MIN_SPEED:
      self.reason = "speed"
      return self._slew_to(0.0)

    scale = _angle_scale(turn_angle_deg)
    if scale <= 0.0:
      # Either a shallow turn the model handles alone, or a u-turn we exclude.
      self.reason = "uturn" if abs(turn_angle_deg) >= UTURN_ANGLE else "shallow"
      return self._slew_to(0.0)

    # Direction from the desire; if the route's angle sign disagrees with the
    # commanded desire (stale/racy data), don't push -- ambiguous direction is
    # never worth a feedforward steer.
    sign = CURV_SIGN[desire]
    if turn_angle_deg != 0.0 and np.sign(turn_angle_deg) != sign:
      self.reason = "sign mismatch"
      return self._slew_to(0.0)

    v = max(v_ego, V_EGO_FLOOR)
    kappa_budget = ASSIST_MAX_ACCEL / (v * v)
    target = sign * scale * kappa_budget
    target = float(np.clip(target, -KAPPA_HARD_MAX, KAPPA_HARD_MAX))

    self.active = True
    self.reason = "active"
    return self._slew_to(target)
