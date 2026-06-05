#!/usr/bin/env python3
"""
Experimental Mapbox-based navigation daemon for nkaoud-sp fork.

Phase 3 (this revision): full route fetching + maneuver/route publishing.

- Watches NkaoudNavDestination (written by the onroad NAV button).
- Fetches a Mapbox driving-traffic route in a worker thread (non-blocking).
- Publishes navRoute (polyline) on every new route.
- Publishes navInstruction continuously (current maneuver + distance + ETA).
- Publishes nkaoudNavigationSP with active/onRoute/rerouting/routeId.
- Detects bearing-misalignment off-route and triggers a reroute
  (>95 deg, min speed 5 m/s, 3-tick counter).

maneuverTargetSpeed is still 0.0 here -- phase 6 fills that in.
"""
from __future__ import annotations

import math
import os
import threading
import time

import cereal.messaging as messaging
from cereal import custom
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.nkaoud_nav.geometry import (
  Coordinate, closest_segment_index, distance_along_geometry,
  route_bearing_at, total_geometry_length,
)
from openpilot.sunnypilot.nkaoud_nav.route_client import (
  Banner, LaneOption, RouteData, RouteFetchError, fetch_route,
)
from openpilot.sunnypilot.selfdrive.controls.lib.lane_position import LanePositionEstimator

NavDesire = custom.NkaoudNavigationSP.NavDesire


# Reroute thresholds (ported from old fork's bearing-misalignment detector).
BEARING_MISALIGN_THRESHOLD_DEG = 95.0
BEARING_MISALIGN_MIN_SPEED_MS = 5.0
BEARING_MISALIGN_COUNTER_MIN = 3

ARRIVAL_DISTANCE_M = 25.0          # consider the destination reached within this
MIN_REROUTE_INTERVAL_S = 8.0       # back off so reroutes don't spam the API

# Turn-slowdown target speed (ported from old fork's
# navigation_test_maneuver_target_speed / TURN_SLOWDOWN_MIN_SPEED_MS).
TURN_SLOWDOWN_SPEED_MS = 25.0 / 3.6   # ~6.94 m/s (25 km/h)
TURN_SLOWDOWN_RANGE_M = 150.0         # only apply within this distance to maneuver
TURN_MANEUVER_MODIFIERS = ("left", "right", "uturn", "sharpLeft", "sharpRight")

# Lateral influence (phase 7) -- distance windows to the upcoming maneuver.
# When closer than DESIRE_TURN_RANGE_M we emit turnLeft/turnRight as a direct
# pre-turn cue. Between TURN_RANGE and LANE_KEEP_RANGE we emit keepLeft/
# keepRight if the lane-position estimator says we're not on the correct
# half of the road for the upcoming maneuver.
DESIRE_TURN_RANGE_M = 50.0
DESIRE_LANE_KEEP_RANGE_M = 200.0
LEFT_TURN_MODIFIERS = ("left", "sharpLeft", "uturn")
RIGHT_TURN_MODIFIERS = ("right", "sharpRight")


def _read_destination(params: Params) -> Coordinate | None:
  # JSON-typed params come back as the parsed object directly (dict here), not a string.
  d = params.get("NkaoudNavDestination")
  if not isinstance(d, dict):
    return None
  if "latitude" not in d or "longitude" not in d:
    return None
  return Coordinate(float(d["latitude"]), float(d["longitude"]))


def _read_token(params: Params) -> str:
  return (params.get("NkaoudNavMapboxToken") or "").strip()


# Drop-in token import: place the token in any of these files (one of which is
# reachable via the copyparty web UI when EnableCopyparty is on) and navd will
# read it, write it to the param, and delete the file. Avoids typing a long
# token on the on-screen keyboard.
TOKEN_DROP_PATHS = (
  "/data/openpilot/nkaoud_mapbox_token.txt",
  "/data/openpilot/sunnypilot/nkaoud_nav/mapbox_token.txt",
)


def _maybe_import_token_file(params: Params) -> None:
  for path in TOKEN_DROP_PATHS:
    if not os.path.exists(path):
      continue
    try:
      with open(path) as f:
        token = f.read().strip()
      os.remove(path)
    except OSError as e:
      cloudlog.warning(f"nkaoud_navd: token-file import failed at {path}: {e}")
      continue
    if not token:
      cloudlog.warning(f"nkaoud_navd: token-file at {path} was empty, removed")
      continue
    params.put("NkaoudNavMapboxToken", token)
    cloudlog.info(f"nkaoud_navd: imported Mapbox token from {path}, file removed")
    return


def _location_from_llk(llk) -> tuple[Coordinate | None, float | None, float]:
  """Returns (position, bearing_deg, v_ego_ms_estimate). Bearing/speed may be None."""
  if not llk.gpsOK:
    return None, None, 0.0
  geo = llk.positionGeodetic
  if not geo.valid or len(geo.value) < 2:
    return None, None, 0.0
  pos = Coordinate(geo.value[0], geo.value[1])

  bearing: float | None = None
  ori = llk.calibratedOrientationNED
  if ori.valid and len(ori.value) == 3:
    yaw_rad = ori.value[2]
    bearing = (math.degrees(yaw_rad) + 360.0) % 360.0

  v_ego = 0.0
  vel = llk.velocityCalibrated
  if vel.valid and len(vel.value) >= 1:
    v_ego = float(vel.value[0])

  return pos, bearing, v_ego


class RouteFetcher:
  """Runs Mapbox fetches off the main loop. One in-flight request at a time."""

  def __init__(self) -> None:
    self._thread: threading.Thread | None = None
    self._result: RouteData | None = None
    self._error: str | None = None
    self._request_id: int = 0
    self._lock = threading.Lock()

  def in_flight(self) -> bool:
    return self._thread is not None and self._thread.is_alive()

  def submit(self, origin: Coordinate, destination: Coordinate, token: str,
             bearing: float | None) -> int:
    with self._lock:
      self._request_id += 1
      rid = self._request_id
      self._result = None
      self._error = None
    self._thread = threading.Thread(
      target=self._run, args=(rid, origin, destination, token, bearing),
      name="nkaoud_navd_fetch", daemon=True,
    )
    self._thread.start()
    return rid

  def _run(self, rid: int, origin: Coordinate, destination: Coordinate,
           token: str, bearing: float | None) -> None:
    try:
      result = fetch_route(origin, destination, token, bearing_deg=bearing)
      with self._lock:
        if rid == self._request_id:
          self._result = result
    except RouteFetchError as e:
      with self._lock:
        if rid == self._request_id:
          self._error = str(e)

  def take_result(self) -> tuple[RouteData | None, str | None]:
    with self._lock:
      r, e = self._result, self._error
      self._result = None
      self._error = None
      return r, e


class NkaoudNavd:
  def __init__(self) -> None:
    self.params = Params()
    self.sm = messaging.SubMaster(['liveLocationKalman', 'modelV2'])
    self.pm = messaging.PubMaster(['nkaoudNavigationSP', 'navRoute', 'navInstruction'])
    self.rk = Ratekeeper(5.0)

    self.fetcher = RouteFetcher()
    self.route: RouteData | None = None
    self.destination: Coordinate | None = None
    self.step_idx: int = 0
    self.bearing_misalign_counter: int = 0
    self.last_route_fetch_t: float = 0.0
    self.rerouting: bool = False
    self.last_pos: Coordinate | None = None
    self.last_bearing: float | None = None
    self.last_v_ego: float = 0.0
    self.last_distance_along: float = 0.0
    self.arrived: bool = False
    self.lane_position_est = LanePositionEstimator()
    self.lane_current: int = 0
    self.lane_total: int = 0
    self.lane_conf: str = "unknown"

  # ---- core loop ----
  def step(self) -> None:
    self.sm.update(0)

    _maybe_import_token_file(self.params)

    pos, bearing, v_ego = _location_from_llk(self.sm['liveLocationKalman'])
    if pos is not None:
      self.last_pos = pos
    if bearing is not None:
      self.last_bearing = bearing
    self.last_v_ego = v_ego

    if self.sm.updated['modelV2']:
      self.lane_current, self.lane_total, self.lane_conf = self.lane_position_est.update(self.sm['modelV2'])

    self._maybe_drain_fetcher()

    new_dest = _read_destination(self.params)
    if not self._same_destination(new_dest):
      self.destination = new_dest
      self.route = None
      self.step_idx = 0
      self.arrived = False
      self.bearing_misalign_counter = 0
      self._try_fetch_initial()

    self._update_progress()
    self._maybe_reroute()
    self._publish()

  def run(self) -> None:
    while True:
      self.step()
      self.rk.keep_time()

  # ---- helpers ----
  def _same_destination(self, d: Coordinate | None) -> bool:
    if d is None and self.destination is None:
      return True
    if d is None or self.destination is None:
      return False
    return (abs(d.latitude - self.destination.latitude) < 1e-7
            and abs(d.longitude - self.destination.longitude) < 1e-7)

  def _try_fetch_initial(self) -> None:
    if self.destination is None:
      return
    if self.last_pos is None:
      cloudlog.warning("nkaoud_navd: have destination but no GPS yet, deferring fetch")
      return
    token = _read_token(self.params)
    if not token:
      cloudlog.warning("nkaoud_navd: NkaoudNavMapboxToken is empty, cannot fetch route")
      return
    if self.fetcher.in_flight():
      return
    cloudlog.info(f"nkaoud_navd: fetching route to {self.destination.latitude:.5f},{self.destination.longitude:.5f}")
    self.fetcher.submit(self.last_pos, self.destination, token, self.last_bearing)
    self.last_route_fetch_t = time.monotonic()
    self.rerouting = self.route is not None  # mark only if replacing an existing route

  def _maybe_drain_fetcher(self) -> None:
    if self.fetcher.in_flight():
      return
    result, error = self.fetcher.take_result()
    if result is not None:
      cloudlog.info(f"nkaoud_navd: route received ({result.distance_total:.0f} m, {len(result.steps)} steps)")
      self.route = result
      self.step_idx = 0
      self.rerouting = False
    elif error is not None:
      cloudlog.warning(f"nkaoud_navd: route fetch failed: {error}")
      self.rerouting = False

  def _update_progress(self) -> None:
    if self.route is None or self.last_pos is None or not self.route.geometry:
      return

    # Arrival check
    if self.destination is not None:
      dist_to_dest = self.last_pos.distance_to(self.destination)
      if dist_to_dest < ARRIVAL_DISTANCE_M:
        self.arrived = True
        # Clear destination so the user has to pick a new one (matches old fork)
        self.params.remove("NkaoudNavDestination")
        return

    # Advance step_idx to whichever step contains the closest segment.
    cumulative = self.route.cumulative_step_distance
    self.last_distance_along = distance_along_geometry(self.route.geometry, self.last_pos)
    # Find largest step whose cumulative start <= last_distance_along
    new_idx = 0
    for i, c in enumerate(cumulative):
      if c <= self.last_distance_along:
        new_idx = i
      else:
        break
    self.step_idx = new_idx

  def _maybe_reroute(self) -> None:
    if self.route is None or self.last_pos is None:
      return
    if self.fetcher.in_flight():
      return
    if time.monotonic() - self.last_route_fetch_t < MIN_REROUTE_INTERVAL_S:
      return

    geom = self.route.geometry
    if self.last_bearing is None or self.last_v_ego < BEARING_MISALIGN_MIN_SPEED_MS:
      self.bearing_misalign_counter = 0
      return

    route_bearing = route_bearing_at(geom, self.last_pos)
    if route_bearing is None:
      self.bearing_misalign_counter = 0
      return

    diff = abs(((self.last_bearing - route_bearing) + 540.0) % 360.0 - 180.0)
    if diff > BEARING_MISALIGN_THRESHOLD_DEG:
      self.bearing_misalign_counter += 1
    else:
      self.bearing_misalign_counter = 0

    if self.bearing_misalign_counter > BEARING_MISALIGN_COUNTER_MIN:
      self.bearing_misalign_counter = 0
      self._try_fetch_initial()

  # ---- publishing ----
  def _publish(self) -> None:
    self._publish_sp()
    self._publish_nav_route()
    self._publish_nav_instruction()

  def _publish_sp(self) -> None:
    msg = messaging.new_message('nkaoudNavigationSP')
    msg.valid = bool(self.sm['liveLocationKalman'].gpsOK)
    nav = msg.nkaoudNavigationSP
    nav.enabled = self.params.get_bool("NkaoudNavEnabled")
    nav.active = self.route is not None and self.destination is not None
    nav.onRoute = nav.active and not self.rerouting
    nav.routeId = self.route.route_id if self.route is not None else ""
    nav.rerouting = self.rerouting or self.fetcher.in_flight()
    nav.maneuverTargetSpeed = self._maneuver_target_speed()
    nav.distanceToManeuver = self._distance_to_maneuver()
    cur_step = self._current_step()
    nav.maneuverType = cur_step.maneuver_type if cur_step is not None else ""
    nav.maneuverModifier = cur_step.maneuver_modifier if cur_step is not None else ""
    nav.recommendedDesire = self._recommended_desire()
    self.pm.send('nkaoudNavigationSP', msg)

  def _publish_nav_route(self) -> None:
    msg = messaging.new_message('navRoute')
    if self.route is not None:
      coords_msg = msg.navRoute.init('coordinates', len(self.route.geometry))
      for i, c in enumerate(self.route.geometry):
        coords_msg[i].latitude = c.latitude
        coords_msg[i].longitude = c.longitude
    self.pm.send('navRoute', msg)

  def _publish_nav_instruction(self) -> None:
    msg = messaging.new_message('navInstruction')
    if self.route is None:
      self.pm.send('navInstruction', msg)
      return

    msg.valid = True
    inst = msg.navInstruction

    distance_remaining = max(0.0, self.route.distance_total - self.last_distance_along)
    inst.distanceRemaining = distance_remaining
    fraction_remaining = (distance_remaining / max(self.route.distance_total, 1.0))
    inst.timeRemaining = self.route.duration_total * fraction_remaining
    inst.timeRemainingTypical = inst.timeRemaining

    cur_step = self._current_step()
    if cur_step is not None:
      dist_to_man = self._distance_to_maneuver()
      inst.maneuverDistance = dist_to_man
      inst.maneuverType = cur_step.maneuver_type
      inst.maneuverModifier = cur_step.maneuver_modifier

      # Pick the highest-detail banner whose distance threshold has passed.
      banner = self._select_banner(cur_step.banners, dist_to_man)
      if banner is not None:
        inst.maneuverPrimaryText = banner.primary_text
        inst.maneuverSecondaryText = banner.secondary_text
        inst.showFull = dist_to_man < banner.distance_along_geometry

    self.pm.send('navInstruction', msg)

  def _current_step(self):
    if self.route is None:
      return None
    if not self.route.steps:
      return None
    idx = min(self.step_idx, len(self.route.steps) - 1)
    return self.route.steps[idx]

  def _distance_to_maneuver(self) -> float:
    if self.route is None or not self.route.cumulative_step_distance:
      return 0.0
    idx = min(self.step_idx, len(self.route.cumulative_step_distance) - 1)
    step_start = self.route.cumulative_step_distance[idx]
    step = self.route.steps[idx]
    step_end = step_start + step.distance
    return max(0.0, step_end - self.last_distance_along)

  def _recommended_desire(self):
    """Phase 7 lateral influence.

    Gated by NkaoudNavControlSteer. Falls back to none unless:
      - We have an active route and an upcoming maneuver.
      - We're inside DESIRE_TURN_RANGE_M -> emit turnLeft/turnRight matching
        the upcoming modifier.
      - Otherwise inside DESIRE_LANE_KEEP_RANGE_M -> consult Mapbox banner
        lanes plus the lane-position estimator. If we're not on the half
        of the road that has active lanes, emit keepLeft/keepRight.

    Returns a custom.NkaoudNavigationSP.NavDesire enum value.
    """
    if not self.params.get_bool("NkaoudNavControlSteer"):
      return NavDesire.none
    cur_step = self._current_step()
    if cur_step is None:
      return NavDesire.none
    dist = self._distance_to_maneuver()
    if dist <= 0.0 or dist > DESIRE_LANE_KEEP_RANGE_M:
      return NavDesire.none

    modifier = cur_step.maneuver_modifier
    # Direct turn cue close to the maneuver.
    if dist <= DESIRE_TURN_RANGE_M:
      if modifier in LEFT_TURN_MODIFIERS:
        return NavDesire.turnLeft
      if modifier in RIGHT_TURN_MODIFIERS:
        return NavDesire.turnRight
      return NavDesire.none

    # Otherwise look at the active banner lanes and our current lane to
    # decide whether we need to drift left or right before the turn.
    side = self._banner_active_side(cur_step, dist)
    if side == "left" and self._need_to_move("left"):
      return NavDesire.keepLeft
    if side == "right" and self._need_to_move("right"):
      return NavDesire.keepRight
    return NavDesire.none

  @staticmethod
  def _banner_active_side(step, dist_to_maneuver: float) -> str:
    """Returns 'left' / 'right' / '' based on which half of the road the
    active lanes are in. Picks the closest banner whose distance threshold
    we've already crossed (same selector NavManeuverBanner uses)."""
    if not step.banners:
      return ""
    current = step.banners[0]
    for b in step.banners:
      if dist_to_maneuver < b.distance_along_geometry:
        current = b
    lanes = current.lanes
    if not lanes:
      return ""
    active_idx = [i for i, ln in enumerate(lanes) if ln.active]
    if not active_idx:
      return ""
    n = len(lanes)
    # All active lanes strictly in the left half -> route wants left side.
    if all(i < n / 2 for i in active_idx):
      return "left"
    if all(i >= n / 2 for i in active_idx):
      return "right"
    return ""

  def _need_to_move(self, side: str) -> bool:
    """Use the lane-position estimator to decide whether the car is already
    on the desired half of the road. If the estimator is uncertain we
    return False (don't move) -- better to do nothing than to nudge into
    the wrong lane."""
    if self.lane_conf in ("unknown", "low") or self.lane_total <= 0:
      return False
    if self.lane_current <= 0:
      return False
    half = (self.lane_total + 1) / 2
    if side == "left":
      return self.lane_current > half
    if side == "right":
      return self.lane_current < half
    return False

  def _maneuver_target_speed(self) -> float:
    """Turn-slowdown target speed (m/s).

    Returns 0.0 when no constraint applies. The longitudinal planner treats
    0.0 / negative as "ignore this source". Mirrors the old fork's
    navigation_test_maneuver_target_speed: a fixed slow speed when a sharp
    turn / u-turn is the next maneuver and we're within range.
    """
    if self.route is None:
      return 0.0
    # Look at the NEXT maneuver (the end of the current step is the
    # upcoming maneuver), not the previous one.
    cur_step = self._current_step()
    if cur_step is None or cur_step.maneuver_modifier not in TURN_MANEUVER_MODIFIERS:
      return 0.0
    dist = self._distance_to_maneuver()
    if dist <= 0.0 or dist > TURN_SLOWDOWN_RANGE_M:
      return 0.0
    return TURN_SLOWDOWN_SPEED_MS

  @staticmethod
  def _select_banner(banners: list[Banner], distance_to_maneuver: float) -> Banner | None:
    if not banners:
      return None
    current = banners[0]
    for b in banners:
      if distance_to_maneuver < b.distance_along_geometry:
        current = b
    return current


def main() -> None:
  NkaoudNavd().run()


if __name__ == "__main__":
  main()
