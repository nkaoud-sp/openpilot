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

LEFT_TURN_MODIFIERS = ("left", "sharpLeft", "uturn")
RIGHT_TURN_MODIFIERS = ("right", "sharpRight")

# Phase 8: maneuver-type-aware ranges. Highway exits/forks need much earlier
# lane positioning than a surface-street turn. (lane_keep_m, turn_cue_m).
MANEUVER_RANGES = {
  "off ramp":   (1500.0, 200.0),
  "on ramp":    (800.0, 150.0),
  "fork":       (500.0, 100.0),
  "roundabout": (300.0, 80.0),
  "rotary":     (300.0, 80.0),
  "merge":      (400.0, 100.0),
}
DEFAULT_RANGES = (200.0, 50.0)   # surface streets / "turn" / unknown

# Phase 8: cross-track reroute -- catches gradual drift off-route that
# bearing misalignment won't see because the heading stays roughly correct.
CROSS_TRACK_THRESHOLD_M = 30.0
CROSS_TRACK_COUNTER_MIN = 25     # ~5 s at 5 Hz

# Phase 8: missed-maneuver detection. After step_idx advances past a step
# whose upcoming maneuver was a left/right/uturn, expect a meaningful
# heading change. If we didn't turn, trip the counter.
MISS_HEADING_CHANGE_DEG = 30.0   # required absolute heading delta after the maneuver
MISS_OBSERVATION_S = 2.5         # how long we wait before evaluating
MISS_COUNTER_MIN = 1

# Phase 9: active lane-change cooldown -- once a nav-triggered lane change
# finishes, give the lane-position estimator and the model time to settle
# before we consider another one. Otherwise the (just-old) "wrong lane"
# reading would chain into a second lane change.
NAV_LC_COOLDOWN_S = 4.0
AUTO_LANE_CHANGE_OFF = -1        # AutoLaneChangeTimer param value meaning "off"

# Phase 10: highway-cruising default lane. When the next maneuver is far
# enough away that the lane-keep window hasn't opened, target the center
# lane on motorway-class roads so passes and either-side exits stay
# reachable.
HIGHWAY_CLASSES = ("motorway", "motorway_link", "trunk")
HIGHWAY_DEFAULT_MIN_SPEED_MS = 60.0 / 3.6      # ~60 km/h; below this we don't auto-position
HIGHWAY_DEFAULT_MIN_DIST_M = 1500.0            # only target center when next maneuver is at least this far


def _ranges_for(maneuver_type: str) -> tuple[float, float]:
  return MANEUVER_RANGES.get((maneuver_type or "").strip(), DEFAULT_RANGES)


def _bearing_delta(a: float, b: float) -> float:
  d = (a - b + 540.0) % 360.0 - 180.0
  return abs(d)


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
    self.cross_track_m: float = 0.0
    self.cross_track_counter: int = 0
    self.missed_counter: int = 0
    self._prev_step_idx: int = 0
    self._missed_watch_until_t: float = 0.0
    self._missed_watch_bearing: float | None = None
    self._missed_watch_modifier: str = ""
    self._last_lane_change_state: str = "off"
    self._lc_cooldown_until_t: float = 0.0
    self._last_logged_desire: str = "none"
    self._last_logged_modifier: str = ""

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
      mv2 = self.sm['modelV2']
      self.lane_current, self.lane_total, self.lane_conf = self.lane_position_est.update(mv2)
      # Track DesireHelper's lane-change state via modelV2.meta so we don't
      # ask for a fresh lane change while one is already running (or right
      # after one ends).
      lcs = str(mv2.meta.laneChangeState)
      if self._last_lane_change_state != "off" and lcs == "off":
        self._lc_cooldown_until_t = time.monotonic() + NAV_LC_COOLDOWN_S
      self._last_lane_change_state = lcs

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
    # Also record perpendicular distance to the route for cross-track reroute.
    _idx, perp, _t = closest_segment_index(self.route.geometry, self.last_pos)
    self.cross_track_m = perp
    # Find largest step whose cumulative start <= last_distance_along
    new_idx = 0
    for i, c in enumerate(cumulative):
      if c <= self.last_distance_along:
        new_idx = i
      else:
        break

    # On step boundary, snapshot bearing if the step we just entered came
    # from a turn maneuver. After MISS_OBSERVATION_S we'll compare against
    # current bearing -- no significant change = we drove straight through
    # the turn point = missed.
    if new_idx != self._prev_step_idx and new_idx < len(self.route.steps):
      entered_step = self.route.steps[new_idx]
      if entered_step.maneuver_modifier in TURN_MANEUVER_MODIFIERS:
        self._missed_watch_until_t = time.monotonic() + MISS_OBSERVATION_S
        self._missed_watch_bearing = self.last_bearing
        self._missed_watch_modifier = entered_step.maneuver_modifier
    self._prev_step_idx = new_idx
    self.step_idx = new_idx

  def _maybe_reroute(self) -> None:
    if self.route is None or self.last_pos is None:
      return
    if self.fetcher.in_flight():
      return
    if time.monotonic() - self.last_route_fetch_t < MIN_REROUTE_INTERVAL_S:
      # Still update counters so we don't false-trigger the instant the
      # backoff expires.
      self._update_cross_track_counter()
      self._update_missed_counter()
      return

    geom = self.route.geometry
    if self.last_bearing is not None and self.last_v_ego >= BEARING_MISALIGN_MIN_SPEED_MS:
      route_bearing = route_bearing_at(geom, self.last_pos)
      if route_bearing is None:
        self.bearing_misalign_counter = 0
      else:
        diff = _bearing_delta(self.last_bearing, route_bearing)
        if diff > BEARING_MISALIGN_THRESHOLD_DEG:
          self.bearing_misalign_counter += 1
        else:
          self.bearing_misalign_counter = 0
    else:
      self.bearing_misalign_counter = 0

    self._update_cross_track_counter()
    self._update_missed_counter()

    if self.bearing_misalign_counter > BEARING_MISALIGN_COUNTER_MIN:
      cloudlog.info("nkaoud_navd: reroute trigger -- bearing misaligned")
      self.bearing_misalign_counter = 0
      self._try_fetch_initial()
    elif self.cross_track_counter > CROSS_TRACK_COUNTER_MIN:
      cloudlog.info(f"nkaoud_navd: reroute trigger -- cross-track {self.cross_track_m:.1f} m")
      self.cross_track_counter = 0
      self._try_fetch_initial()
    elif self.missed_counter > MISS_COUNTER_MIN:
      cloudlog.info(f"nkaoud_navd: reroute trigger -- missed {self._missed_watch_modifier} turn")
      self.missed_counter = 0
      self._missed_watch_until_t = 0.0
      self._try_fetch_initial()

  def _update_cross_track_counter(self) -> None:
    if self.cross_track_m > CROSS_TRACK_THRESHOLD_M and self.last_v_ego >= BEARING_MISALIGN_MIN_SPEED_MS:
      self.cross_track_counter += 1
    else:
      self.cross_track_counter = 0

  def _update_missed_counter(self) -> None:
    # Wait until the observation window closes, then judge by heading change.
    if self._missed_watch_until_t <= 0.0:
      return
    if time.monotonic() < self._missed_watch_until_t:
      return
    pre = self._missed_watch_bearing
    cur = self.last_bearing
    self._missed_watch_until_t = 0.0
    if pre is None or cur is None:
      return
    delta = _bearing_delta(cur, pre)
    if delta < MISS_HEADING_CHANGE_DEG:
      self.missed_counter += 1
    else:
      self.missed_counter = 0

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
    upcoming = self._upcoming_step()
    cur_step = self._current_step()
    nav.maneuverType = upcoming.maneuver_type if upcoming is not None else ""
    nav.maneuverModifier = upcoming.maneuver_modifier if upcoming is not None else ""
    nav.recommendedDesire = self._recommended_desire()

    # Phase 8 fields.
    lane_keep_m, _ = _ranges_for(upcoming.maneuver_type if upcoming else "")
    upcoming_modifier = upcoming.maneuver_modifier if upcoming is not None else ""
    side = self._route_side(cur_step, self._distance_to_maneuver(), upcoming_modifier) if upcoming else ""
    if side == "left":
      nav.recommendedLaneSide = "left"
    elif side == "right":
      nav.recommendedLaneSide = "right"
    else:
      nav.recommendedLaneSide = "none"
    nav.laneKeepDistance = float(lane_keep_m if side else 0.0)
    nav.currentRoadClasses = ",".join(cur_step.road_classes) if cur_step else ""
    nav.upcomingRoadClasses = ",".join(upcoming.road_classes) if upcoming else ""
    nav.crossTrackDistance = float(self.cross_track_m)
    nav.missedManeuverCount = int(self.missed_counter)

    # Log when the upcoming-maneuver modifier or our recommendation changes,
    # rate-limited to once per change so the swaglog isn't flooded.
    mod_str = nav.maneuverModifier
    if mod_str != self._last_logged_modifier:
      cloudlog.info(f"nkaoud_navd: upcoming modifier={mod_str!r} dist={nav.distanceToManeuver:.1f}m")
      self._last_logged_modifier = mod_str
    desire_str = str(nav.recommendedDesire)
    if desire_str != self._last_logged_desire:
      cloudlog.info(f"nkaoud_navd: recommendedDesire={desire_str} "
                    f"(dist={nav.distanceToManeuver:.1f}m, lane={self.lane_current}/{self.lane_total} "
                    f"conf={self.lane_conf})")
      self._last_logged_desire = desire_str

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
    upcoming = self._upcoming_step()
    if cur_step is not None:
      dist_to_man = self._distance_to_maneuver()
      inst.maneuverDistance = dist_to_man
      # Type/modifier describe the UPCOMING maneuver (start of the next
      # step), not the one that started this step. Banners are on this
      # step but describe the upcoming maneuver -- prefer the banner's
      # modifier when present so the arrow always matches the text.
      banner = self._select_banner(cur_step.banners, dist_to_man)
      if banner is not None:
        inst.maneuverType = banner.maneuver_type or (upcoming.maneuver_type if upcoming else "")
        inst.maneuverModifier = banner.maneuver_modifier or (upcoming.maneuver_modifier if upcoming else "")
        inst.maneuverPrimaryText = banner.primary_text
        inst.maneuverSecondaryText = banner.secondary_text
        inst.showFull = dist_to_man < banner.distance_along_geometry
      elif upcoming is not None:
        inst.maneuverType = upcoming.maneuver_type
        inst.maneuverModifier = upcoming.maneuver_modifier

    self.pm.send('navInstruction', msg)

  def _current_step(self):
    if self.route is None:
      return None
    if not self.route.steps:
      return None
    idx = min(self.step_idx, len(self.route.steps) - 1)
    return self.route.steps[idx]

  def _upcoming_step(self):
    """The step whose start IS the next maneuver. In Mapbox each step's
    maneuver is at its START, so the upcoming maneuver lives at
    steps[step_idx + 1].maneuver_*. The bannerInstructions describing
    that same upcoming maneuver are attached to steps[step_idx] (the
    step we're currently driving), which is why _publish_nav_instruction
    pulls banners from _current_step() but everything else (modifier,
    desire gating, slowdown) reads from _upcoming_step()."""
    if self.route is None or not self.route.steps:
      return None
    nxt = self.step_idx + 1
    if nxt >= len(self.route.steps):
      return None
    return self.route.steps[nxt]

  def _distance_to_maneuver(self) -> float:
    if self.route is None or not self.route.cumulative_step_distance:
      return 0.0
    idx = min(self.step_idx, len(self.route.cumulative_step_distance) - 1)
    step_start = self.route.cumulative_step_distance[idx]
    step = self.route.steps[idx]
    step_end = step_start + step.distance
    return max(0.0, step_end - self.last_distance_along)

  def _recommended_desire(self):
    """Phase 7-10 lateral influence.

    Gated by NkaoudNavControlSteer. Order of precedence (only one fires):
      1. Direct turn cue within turn_cue_m -> turnLeft/turnRight.
      2. Lane-positioning for an upcoming maneuver within lane_keep_m.
         Target lane is incremental -- loose at the far edge of the
         window (any lane on the correct side), strict at the near edge
         (outermost lane).
      3. Highway-cruise default -- when no maneuver is imminent and the
         current step is motorway-class, target the center lane so we
         keep our options open.
    """
    if not self.params.get_bool("NkaoudNavControlSteer"):
      return NavDesire.none
    cur_step = self._current_step()
    upcoming = self._upcoming_step()
    if cur_step is None:
      return NavDesire.none
    dist = self._distance_to_maneuver()

    # Imminent maneuver path
    if upcoming is not None and dist > 0.0:
      lane_keep_m, turn_cue_m = _ranges_for(upcoming.maneuver_type)
      if dist <= lane_keep_m:
        modifier = upcoming.maneuver_modifier
        if dist <= turn_cue_m:
          if modifier in LEFT_TURN_MODIFIERS:
            return NavDesire.turnLeft
          if modifier in RIGHT_TURN_MODIFIERS:
            return NavDesire.turnRight
          return NavDesire.none

        side = self._route_side(cur_step, dist, modifier)
        if side:
          target = self._target_lane(side, dist, lane_keep_m, turn_cue_m)
          if target is not None and self._need_to_move(side, target):
            return self._lc_or_keep(side)
        return NavDesire.none

    # Highway-cruise default path -- only reached when no imminent maneuver
    # tweaks lateral. Target the center lane on motorway-class roads.
    return self._highway_default_desire(cur_step, dist)

  def _lc_or_keep(self, side: str):
    """Pick laneChange* (auto-execute) or keep* (bias) based on whether
    the user has AutoLaneChange enabled + we're outside the cooldown."""
    alc_timer = self.params.get("AutoLaneChangeTimer", return_default=True)
    auto_lc_allowed = (alc_timer is not None and int(alc_timer) != AUTO_LANE_CHANGE_OFF
                       and self._last_lane_change_state == "off"
                       and time.monotonic() >= self._lc_cooldown_until_t)
    if auto_lc_allowed:
      return NavDesire.laneChangeLeft if side == "left" else NavDesire.laneChangeRight
    return NavDesire.keepLeft if side == "left" else NavDesire.keepRight

  def _target_lane(self, side: str, dist: float, lane_keep_m: float, turn_cue_m: float) -> int | None:
    """1-indexed target lane. Lerps from 'any lane on the correct side'
    at lane_keep_m to 'outermost' at turn_cue_m so the requirement
    tightens as we approach the maneuver."""
    if side not in ("left", "right") or self.lane_total <= 0:
      return None
    n = self.lane_total
    if n == 1:
      return 1
    span = lane_keep_m - turn_cue_m
    progress = 1.0 if span <= 0 else max(0.0, min(1.0, 1.0 - (dist - turn_cue_m) / span))
    half_tolerance = math.ceil(n / 2)
    if side == "right":
      loose = n - half_tolerance + 1   # smallest acceptable on right (e.g. lane 3 of 4)
      strict = n                       # outermost lane
      return int(0.5 + loose + (strict - loose) * progress)
    # side == "left"
    loose = half_tolerance             # largest acceptable on left (e.g. lane 2 of 4)
    strict = 1
    return int(0.5 + loose - (loose - strict) * progress)

  def _highway_default_desire(self, cur_step, dist: float):
    """When cruising on a motorway with no imminent maneuver, drift to
    the center lane. ceil(N/2) means: 3-lane -> 2 (center), 4-lane ->
    2 (center-left), 5-lane -> 3 (center)."""
    if cur_step is None or not cur_step.road_classes:
      return NavDesire.none
    if not any(c in HIGHWAY_CLASSES for c in cur_step.road_classes):
      return NavDesire.none
    # Don't fight the imminent-maneuver logic at the boundary.
    if dist > 0.0 and dist < HIGHWAY_DEFAULT_MIN_DIST_M:
      return NavDesire.none
    if self.last_v_ego < HIGHWAY_DEFAULT_MIN_SPEED_MS:
      return NavDesire.none
    if self.lane_conf in ("unknown", "low") or self.lane_total <= 1 or self.lane_current <= 0:
      return NavDesire.none
    target = math.ceil(self.lane_total / 2)
    if self.lane_current == target:
      return NavDesire.none
    side = "left" if self.lane_current > target else "right"
    return self._lc_or_keep(side)

  def _route_side(self, cur_step, dist_to_maneuver: float, modifier: str) -> str:
    """Which half of the road the route wants for the upcoming maneuver.

    Tries Mapbox's explicit banner.sub.components lane guidance first; for
    surface-street turns that data is usually absent, so falls back to the
    maneuver modifier itself ('left' / 'sharpLeft' / 'uturn' -> 'left';
    'right' / 'sharpRight' -> 'right'). Returns '' when there is no
    actionable preference.
    """
    if cur_step is not None:
      side = self._banner_active_side(cur_step, dist_to_maneuver)
      if side:
        return side
    if modifier in LEFT_TURN_MODIFIERS:
      return "left"
    if modifier in RIGHT_TURN_MODIFIERS:
      return "right"
    return ""

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

  def _need_to_move(self, side: str, target_lane: int) -> bool:
    """Whether we should move toward `side` to reach target_lane. Stays
    conservative -- only triggers on a high-confidence lane read, never
    on an unknown/low one (better to do nothing than nudge into the
    wrong lane)."""
    if self.lane_conf in ("unknown", "low") or self.lane_total <= 0:
      return False
    if self.lane_current <= 0:
      return False
    if side == "right":
      return self.lane_current < target_lane
    if side == "left":
      return self.lane_current > target_lane
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
    # Look at the upcoming maneuver -- the one at the START of the next
    # step, which is the action we'll execute at the END of the current
    # step's geometry.
    upcoming = self._upcoming_step()
    if upcoming is None or upcoming.maneuver_modifier not in TURN_MANEUVER_MODIFIERS:
      return 0.0
    dist = self._distance_to_maneuver()
    # Slowdown range scales with maneuver type (highway exits start earlier).
    _lane_keep_m, slowdown_range = _ranges_for(upcoming.maneuver_type)
    slowdown_range = max(slowdown_range, TURN_SLOWDOWN_RANGE_M)
    if dist <= 0.0 or dist > slowdown_range:
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
