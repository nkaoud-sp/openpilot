"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Nav Path Assist (mode B).

An alternative to Nav Turn Assist. Instead of *adding* a fixed lateral-accel
nudge (mode A), this expresses the upcoming route turn as its own *curvature
target* -- the curvature the road geometry implies -- and asks controlsd to
*blend* the model's curvature toward it by a weight. This mirrors comma's old
navigation model, which output a route path (NavModelData.position) rather than
a scalar bias.

The curvature target's magnitude has three sources, in preference order:

  * trajectory (NkaoudNavPathTrajectory): a pure-pursuit path-following target --
    aim at the point on the route a speed-scaled lookahead ahead and command the
    curvature that reaches it (2*y / Ld^2). Unlike the Menger measurement this
    uses the path's lateral *position*, so it follows the route line (correcting
    heading and cross-track), and the lookahead anticipates the corner instead of
    reacting at the controller's short lookahead. Produced as a blend target, so
    it reuses the same safe pipeline (weight, veto, clip_curvature) as scalar mode.
  * path: the real route path (navPathX/Y in device frame) measured with a
    3-point/Menger curvature around the maneuver.
  * angle: geometric synthesis from the Mapbox turn angle (theta / L), the
    fallback when no valid path is available.

Direction always comes from the trusted turn desire, never from the GPS-derived
path, so a pose or bearing error can change the shape we blend toward but never
the side.

Why a blend instead of an add:

  * Degrades to the model. The weight ramps in with proximity and collapses to
    0 the instant the route disagrees in sign with the model (see the veto
    below). So when the model swerves for something the route can't see -- an
    obstacle, a blocked lane -- nav authority *vanishes* instead of fighting it.
    An additive nudge is at its most dangerous exactly when it is most wrong;
    this is the opposite.

  * Geometry, not a lateral-accel heuristic. The path/angle targets come from the
    road's actual radius (a corner's radius does not change with how fast you take
    it), and the trajectory target from a pure-pursuit lookahead that *does* scale
    with speed (a faster car aims further ahead, the standard path-tracking
    behaviour). Either way the magnitude is a real turn radius, not an a_lat
    budget; clip_curvature downstream still bounds lateral accel.

Safety envelope (all still in force):
  * weight is hard-capped below 1 (W_MAX) -- the model always keeps authority.
  * the blend may only *strengthen* the model's turn, never straighten it
    (enforced in controlsd).
  * clip_curvature remains the final bound on the blended curvature.
  * off unless NkaoudNavPathAssist is enabled.
"""
from __future__ import annotations

import numpy as np

from cereal import log
from openpilot.common.params import Params
# Share the angle band and sign convention with mode A so the two trigger under
# the same conditions and are directly comparable on the road.
from openpilot.sunnypilot.selfdrive.controls.lib.nav_turn_assist import (
  TURN_DESIRES, CURV_SIGN, MIN_TURN_ANGLE, UTURN_ANGLE,
)

Desire = log.Desire

# Characteristic along-path length (m) over which the maneuver's heading change
# is completed. kappa_turn = theta_rad / TURN_LENGTH_M, so a 90 deg turn implies
# ~R = 12.7 m and a 45 deg turn ~R = 25 m -- typical city-corner radii.
TURN_LENGTH_M = 20.0

# Cap on the route curvature *target* (1/m => R = 10 m floor). Unlike mode A's
# small nudge cap, this represents the actual corner, so it is a real turn
# radius, not a bias limit -- authority is bounded by the blend weight below and
# by clip_curvature (lateral accel/jerk) downstream, not by shrinking the target.
KAPPA_PATH_MAX = 0.10

# Proximity ramp: authority (and commanded curvature) grow from 0 at ENGAGE_DIST
# to full at the maneuver point, then fall away as the desire clears.
ENGAGE_DIST_M = 30.0

# Max blend weight. Strictly below 1 so the model never loses lateral authority.
W_MAX = 0.6

# Model curvature magnitude (1/m) above which a route/model sign disagreement
# vetoes the blend outright. Below it the model is ~straight and a mild route
# lean is safe.
K_TRUST = 0.008

MIN_SPEED = 2.0             # m/s; below this we don't blend (parking/creep)

# Slew the weight so engagement/handoff is smooth. Model runs at ~20 Hz.
WEIGHT_SLEW = 4.0e-2        # per frame => ~0.75 s to reach W_MAX

PARAM_READ_PERIOD_FRAMES = 20  # ~1 Hz at the model rate

# Route-path curvature measurement. Must match navd's NAV_PATH_SPACING_M so an
# index maps to arc length. Curvature is measured only around the maneuver
# (center +/- BEND_MARGIN) so a sharp corner further along the horizon can't
# hijack the assist for the gentle one we're actually taking.
NAV_PATH_SPACING_M = 2.0
TRIPLE_SPACING_M = 6.0      # gap between the three points of each curvature sample
BEND_MARGIN_M = 12.0        # search window half-width around dist_to_maneuver

# Trajectory (pure-pursuit) mode. The lookahead distance is speed-scaled so the
# assist anticipates the corner instead of reacting at the controller's short
# lookahead (a corner tens of metres away is straight at 3 m, so evaluating the
# route heading there is ~0 -- pure pursuit aims at a point that reaches into the
# turn). Clamped to a sane window.
LD_TIME_S = 1.0
LD_MIN_M = 6.0
LD_MAX_M = 30.0


def _proximity_ramp(dist_to_maneuver_m: float) -> float:
  """0 at or beyond ENGAGE_DIST, ramping linearly to 1 at the maneuver point."""
  d = max(float(dist_to_maneuver_m), 0.0)
  return float(np.clip((ENGAGE_DIST_M - d) / ENGAGE_DIST_M, 0.0, 1.0))


def curvature_from_path(nav_path_x, nav_path_y, center_m: float, margin_m: float = BEND_MARGIN_M) -> float:
  """Max unsigned curvature (1/m) of a device-frame path via 3-point (Menger)
  curvature, restricted to samples whose middle point lies within center_m +/-
  margin_m of arc length. 0 if the path is too short or effectively straight."""
  xs = np.asarray(list(nav_path_x), dtype=np.float64)
  ys = np.asarray(list(nav_path_y), dtype=np.float64)
  n = min(len(xs), len(ys))
  if n < 3:
    return 0.0
  pts = np.stack([xs[:n], ys[:n]], axis=1)
  step = max(1, int(round(TRIPLE_SPACING_M / NAV_PATH_SPACING_M)))
  lo, hi = center_m - margin_m, center_m + margin_m
  kappa_max = 0.0
  for i in range(0, n - 2 * step):
    mid_arc = (i + step) * NAV_PATH_SPACING_M
    if mid_arc < lo or mid_arc > hi:
      continue
    a, b, c = pts[i], pts[i + step], pts[i + 2 * step]
    d_ab = np.hypot(*(b - a))
    d_bc = np.hypot(*(c - b))
    d_ca = np.hypot(*(a - c))
    if d_ab < 1e-3 or d_bc < 1e-3 or d_ca < 1e-3:
      continue
    # 2 * triangle area via the cross product of two edges.
    area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
    kappa_max = max(kappa_max, 2.0 * area2 / (d_ab * d_bc * d_ca))  # Menger: 4A/(abc)
  return float(kappa_max)


class NavPathAssist:
  def __init__(self):
    self.params = Params()
    self._frame = 0
    self.enabled = False

    self._weight = 0.0        # slew-limited blend weight currently applied
    self.kappa = 0.0          # last route curvature target (1/m, signed)
    self.active = False
    self.reason = "off"
    self.turn_angle = 0.0
    self.trajectory = False   # NkaoudNavPathTrajectory: use the heading-profile target
    self.source = "none"      # "trajectory" | "path" (measured) | "angle" (fallback)

    self._read_params()

  def _read_params(self) -> None:
    self.enabled = self.params.get_bool("NkaoudNavPathAssist")
    self.trajectory = self.params.get_bool("NkaoudNavPathTrajectory")

  def _slew_weight(self, target: float) -> float:
    self._weight = float(np.clip(target, self._weight - WEIGHT_SLEW, self._weight + WEIGHT_SLEW))
    return self._weight

  def _off(self, reason: str, hard: bool = False) -> tuple[float, float]:
    """Return (curvature, weight). hard=True zeroes the weight immediately
    (used when lateral is not active); otherwise it slews out."""
    self.reason = reason
    self.active = False
    self.kappa = 0.0
    self.source = "none"
    if hard:
      self._weight = 0.0
      return 0.0, 0.0
    return 0.0, self._slew_weight(0.0)

  def _pure_pursuit_curvature(self, nav_path_x, nav_path_y, sign: float, v_ego: float,
                              dist_to_maneuver_m: float) -> float | None:
    """Pure-pursuit path-following curvature magnitude (1/m): aim at the point on
    the route a speed-scaled lookahead ahead and command the curvature that
    reaches it (kappa = 2*y / Ld^2). Unlike the Menger measurement this uses the
    path's lateral *position*, so it corrects heading and cross-track toward the
    route line, and the lookahead anticipates the corner. None if the path is too
    short or the lookahead point is on the opposite side to the commanded turn.
    Magnitude only; the caller applies the desire's sign."""
    xs = np.asarray(list(nav_path_x), dtype=np.float64)
    ys = np.asarray(list(nav_path_y), dtype=np.float64)
    n = min(len(xs), len(ys))
    if n < 2:
      return None
    arc = np.arange(n) * NAV_PATH_SPACING_M
    ld = float(np.clip(LD_TIME_S * v_ego, LD_MIN_M, LD_MAX_M))
    # Don't aim well past the maneuver apex onto post-corner geometry (matters on
    # tight corners / S-curves at high speed); keep the target on the corner we
    # are taking, but never below the minimum lookahead.
    ld = min(ld, max(LD_MIN_M, float(dist_to_maneuver_m) + BEND_MARGIN_M))
    if arc[-1] >= ld:
      x_l = float(np.interp(ld, arc, xs[:n]))
      y_l = float(np.interp(ld, arc, ys[:n]))
    else:                                   # path shorter than Ld: use its end
      x_l, y_l = float(xs[n - 1]), float(ys[n - 1])
    ld_eff = float(np.hypot(x_l, y_l))
    if ld_eff < 1.0:
      return None
    # Side gate: the lookahead point must be on the commanded turn side. y is
    # left; a right turn (sign +1) puts the point at y < 0 -> side_sign +1.
    side_sign = -np.sign(y_l)
    if side_sign != 0.0 and side_sign != sign:
      return None
    return abs(2.0 * y_l / (ld_eff * ld_eff))

  def _target_magnitude(self, mag_deg: float, dist_to_maneuver_m: float,
                        nav_path_x, nav_path_y, nav_path_valid: bool, sign: float,
                        v_ego: float) -> float:
    """Unsigned curvature target (1/m). Prefers the pure-pursuit trajectory
    target, then the measured route-path curvature, then the turn-angle
    synthesis."""
    have_path = nav_path_valid and nav_path_x is not None and nav_path_y is not None
    if self.trajectory and have_path:
      kappa = self._pure_pursuit_curvature(nav_path_x, nav_path_y, sign, v_ego, dist_to_maneuver_m)
      if kappa is not None and kappa > 0.0:
        self.source = "trajectory"
        return kappa
    if have_path:
      kappa = curvature_from_path(nav_path_x, nav_path_y, dist_to_maneuver_m)
      if kappa > 0.0:
        self.source = "path"
        return kappa
    self.source = "angle"
    return np.radians(mag_deg) / TURN_LENGTH_M

  def update(self, desire, turn_angle_deg: float, dist_to_maneuver_m: float,
             v_ego: float, kappa_model: float, lat_active: bool,
             nav_path_x=None, nav_path_y=None, nav_path_valid: bool = False) -> tuple[float, float]:
    """Return (navPathCurvature, navPathWeight) for controlsd to blend. Weight
    is 0 (no blend) unless enabled, lateral is active, a turn desire is applied,
    the angle is in the assisted band, and the route agrees in sign with the
    model. The magnitude is measured from nav_path_x/y when a valid route path
    is supplied, else synthesized from turn_angle_deg."""
    self._frame += 1
    if self._frame % PARAM_READ_PERIOD_FRAMES == 0:
      self._read_params()

    self.turn_angle = float(turn_angle_deg)

    # Whenever lateral is not active the assist must not linger.
    if not lat_active:
      return self._off("not active", hard=True)
    if not self.enabled:
      return self._off("off")
    if desire not in TURN_DESIRES:
      return self._off("no turn desire")
    if v_ego < MIN_SPEED:
      return self._off("speed")

    mag = abs(turn_angle_deg)
    if mag < MIN_TURN_ANGLE:
      return self._off("shallow")
    if mag >= UTURN_ANGLE:
      return self._off("uturn")

    sign = CURV_SIGN[desire]
    # Route angle sign must agree with the commanded desire (stale/racy data).
    if turn_angle_deg != 0.0 and np.sign(turn_angle_deg) != sign:
      return self._off("sign mismatch")

    # Route curvature target: measured from the real path when available, else
    # synthesized from the angle. Direction from the desire, magnitude capped.
    kappa_mag = self._target_magnitude(mag, dist_to_maneuver_m, nav_path_x, nav_path_y,
                                       nav_path_valid, sign, v_ego)
    kappa_turn = sign * float(np.clip(kappa_mag, 0.0, KAPPA_PATH_MAX))

    ramp = _proximity_ramp(dist_to_maneuver_m)
    if ramp <= 0.0:
      # Maneuver still too far to lend authority; blending toward 0 here would
      # only straighten the model, so contribute nothing.
      return self._off("far")

    self.kappa = kappa_turn * ramp

    # Safety veto: if the model is meaningfully steering the *other* way, it is
    # reacting to something the route can't see -- yield entirely.
    if kappa_model != 0.0 and np.sign(kappa_model) != sign and abs(kappa_model) > K_TRUST:
      self.reason = "model disagrees"
      self.active = False
      return self.kappa, self._slew_weight(0.0)

    self.active = True
    self.reason = "active"
    return self.kappa, self._slew_weight(W_MAX * ramp)
