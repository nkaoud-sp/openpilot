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
  Coordinate, closest_segment_index, distance_along_geometry, route_bearing_at,
)
from openpilot.sunnypilot.nkaoud_nav.route_client import (
  Banner, RouteData, RouteFetchError, fetch_route,
)
from openpilot.sunnypilot.nkaoud_nav.share_client import (
  ShareFetchError, fetch_share_destination,
)
from openpilot.sunnypilot.selfdrive.controls.lib.lane_position import FILTER_MODE_BOTH_OR, LanePositionEstimator

NavDesire = custom.NkaoudNavigationSP.NavDesire


# Reroute thresholds (ported from old fork's bearing-misalignment detector).
BEARING_MISALIGN_THRESHOLD_DEG = 95.0
BEARING_MISALIGN_MIN_SPEED_MS = 5.0
BEARING_MISALIGN_COUNTER_MIN = 3

ARRIVAL_DISTANCE_M = 25.0          # consider the destination reached within this
MIN_REROUTE_INTERVAL_S = 8.0       # back off so reroutes don't spam the API
PARAM_REFRESH_S = 0.5              # throttle for per-tick param reads

# Turn-slowdown target speed (ported from old fork's
# navigation_test_maneuver_target_speed / TURN_SLOWDOWN_MIN_SPEED_MS).
TURN_SLOWDOWN_SPEED_MS = 25.0 / 3.6   # ~6.94 m/s (25 km/h)
TURN_SLOWDOWN_RANGE_M = 150.0         # only apply within this distance to maneuver
TURN_MANEUVER_MODIFIERS = ("left", "right", "uturn", "sharpLeft", "sharpRight")

LEFT_TURN_MODIFIERS = ("left", "sharpLeft", "uturn")
RIGHT_TURN_MODIFIERS = ("right", "sharpRight")
# Broader sets used for "which half of the road do we want" lane
# positioning. slight* doesn't warrant a turnLeft/Right cue (gentle
# enough that the model handles it) but absolutely warrants getting
# into the correct lane beforehand -- highway off-ramps in particular
# come back from Mapbox with modifier="slightRight" most of the time.
LEFT_SIDE_MODIFIERS = LEFT_TURN_MODIFIERS + ("slightLeft",)
RIGHT_SIDE_MODIFIERS = RIGHT_TURN_MODIFIERS + ("slightRight",)

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
# (Also covers the old missed-maneuver detector: driving past a turn departs
# the post-turn geometry, which shows up as cross-track within seconds.)
CROSS_TRACK_THRESHOLD_M = 30.0
CROSS_TRACK_COUNTER_MIN = 25     # ~5 s at 5 Hz

# Steering keep* intent is restricted to the maneuver types where advance
# lane positioning actually pays off (long approach, hard-to-recover miss).
# Surface-street turns are left to the model + the turn cue; the UI advisory
# arrow still covers every maneuver type.
LANE_POSITION_MANEUVERS = ("off ramp", "on ramp", "fork", "merge")
# Lane-fix confidence required to INITIATE a steering bias. Any reading may
# stop one (that fails safe), but a low-confidence read must never start one.
STEER_LANE_CONFS = ("high", "medium")

# Phase 11: "Share" destination -- HTTP fetch when the picker writes a new
# NkaoudNavShareTrigger token. Retries on failure are bounded so a wrong
# URL doesn't hammer the endpoint forever.
SHARE_FETCH_RETRY_S = 15.0
SHARE_FETCH_MAX_ATTEMPTS = 4


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


def _sanitize_url(url: str) -> str:
  """Strip user:pass@ from a URL so we can put it in swaglog without leaking
  credentials. Used only for logging."""
  if "@" not in url:
    return url
  scheme, _, after = url.partition("://")
  _creds, _, rest = after.partition("@")
  return f"{scheme}://<creds>@{rest}" if scheme else f"<creds>@{rest}"


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


class ThreadedFetcher:
  """Runs a blocking fetch function off the main loop. One in-flight request
  at a time; a newer submit invalidates any older in-flight result."""

  def __init__(self, fetch_fn, error_cls: type[Exception], name: str) -> None:
    self._fetch_fn = fetch_fn
    self._error_cls = error_cls
    self._name = name
    self._thread: threading.Thread | None = None
    self._result = None
    self._error: str | None = None
    self._request_id: int = 0
    self._lock = threading.Lock()

  def in_flight(self) -> bool:
    return self._thread is not None and self._thread.is_alive()

  def submit(self, *args) -> int:
    with self._lock:
      self._request_id += 1
      rid = self._request_id
      self._result = None
      self._error = None
    self._thread = threading.Thread(
      target=self._run, args=(rid, args), name=self._name, daemon=True,
    )
    self._thread.start()
    return rid

  def _run(self, rid: int, args: tuple) -> None:
    try:
      result = self._fetch_fn(*args)
      with self._lock:
        if rid == self._request_id:
          self._result = result
    except self._error_cls as e:
      with self._lock:
        if rid == self._request_id:
          self._error = str(e)

  def take_result(self):
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

    self.fetcher = ThreadedFetcher(fetch_route, RouteFetchError, "nkaoud_navd_fetch")
    self.share_fetcher = ThreadedFetcher(fetch_share_destination, ShareFetchError, "nkaoud_navd_share_fetch")
    self._last_share_trigger: str | None = None
    self._share_next_retry_t: float = 0.0
    self._share_attempts: int = 0
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
    self._enabled: bool = False
    self._next_param_t: float = 0.0
    self._last_logged_desire: str = "none"
    self._last_logged_modifier: str = ""
    self._last_logged_advisory: str = "none"

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
      self.lane_current, self.lane_total, self.lane_conf = self.lane_position_est.update(self.sm['modelV2'], filter_mode=FILTER_MODE_BOTH_OR)

    now = time.monotonic()
    if now >= self._next_param_t:
      self._next_param_t = now + PARAM_REFRESH_S
      self._enabled = self.params.get_bool("NkaoudNavEnabled")

    self._maybe_drain_fetcher()
    self._handle_share_trigger()

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

  def _handle_share_trigger(self) -> None:
    """The UI bumps NkaoudNavShareTrigger when the user taps the Share
    preset. We pick that up here, GET the configured endpoint, and write
    the result into NkaoudNavDestination so the normal route flow takes
    over. On failure we retry SHARE_FETCH_MAX_ATTEMPTS times with a
    SHARE_FETCH_RETRY_S backoff, then give up until the next trigger."""
    trigger = (self.params.get("NkaoudNavShareTrigger") or "").strip()
    if self._last_share_trigger is None:
      # Seed at startup so a stale trigger from a previous boot doesn't fire.
      self._last_share_trigger = trigger
      return

    new_trigger = trigger != self._last_share_trigger
    if new_trigger:
      self._last_share_trigger = trigger
      self._share_attempts = 0
      self._share_next_retry_t = 0.0
      if not trigger:
        return  # user cleared the trigger; nothing to do
      cloudlog.info(f"nkaoud_navd: share trigger changed to {trigger!r}, will fetch")

    if self.share_fetcher.in_flight():
      return
    result, error = self.share_fetcher.take_result()
    if not trigger:
      # User cleared the trigger after we'd submitted -- drop whatever
      # came back so we don't re-instate the destination they just cleared.
      if result is not None or error is not None:
        cloudlog.info("nkaoud_navd: share fetch completed but trigger was cleared, discarding")
      return
    if result is not None:
      cloudlog.info(f"nkaoud_navd: share fetch OK -> {result.get('place_name')!r} "
                    f"lat={result.get('latitude'):.5f} lon={result.get('longitude'):.5f}; "
                    f"writing NkaoudNavDestination")
      self.params.put("NkaoudNavDestination", result)
      # Mark this trigger as fully handled so we don't re-submit on every
      # subsequent tick (which would overwrite a user-initiated "Clear
      # destination" with the same coordinates). A fresh Share tap bumps
      # NkaoudNavShareTrigger which resets attempts to 0 above.
      self._share_attempts = SHARE_FETCH_MAX_ATTEMPTS
      self._share_next_retry_t = 0.0
      return
    if error is not None:
      self._share_attempts += 1
      self._share_next_retry_t = time.monotonic() + SHARE_FETCH_RETRY_S
      cloudlog.warning(f"nkaoud_navd: share fetch FAILED (attempt {self._share_attempts}/{SHARE_FETCH_MAX_ATTEMPTS}): {error}")

    if not trigger:
      return
    if self._share_attempts >= SHARE_FETCH_MAX_ATTEMPTS:
      return
    if time.monotonic() < self._share_next_retry_t:
      return
    url = (self.params.get("NkaoudNavShareEndpoint") or "").strip()
    if not url:
      if new_trigger:
        cloudlog.warning("nkaoud_navd: NkaoudNavShareEndpoint is empty, cannot fetch")
      self._share_attempts = SHARE_FETCH_MAX_ATTEMPTS  # short-circuit until URL is set
      return
    cloudlog.info(f"nkaoud_navd: submitting share fetch ({_sanitize_url(url)})")
    self.share_fetcher.submit(url)

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

    if self.bearing_misalign_counter > BEARING_MISALIGN_COUNTER_MIN:
      cloudlog.info("nkaoud_navd: reroute trigger -- bearing misaligned")
      self.bearing_misalign_counter = 0
      self._try_fetch_initial()
    elif self.cross_track_counter > CROSS_TRACK_COUNTER_MIN:
      cloudlog.info(f"nkaoud_navd: reroute trigger -- cross-track {self.cross_track_m:.1f} m")
      self.cross_track_counter = 0
      self._try_fetch_initial()

  def _update_cross_track_counter(self) -> None:
    if self.cross_track_m > CROSS_TRACK_THRESHOLD_M and self.last_v_ego >= BEARING_MISALIGN_MIN_SPEED_MS:
      self.cross_track_counter += 1
    else:
      self.cross_track_counter = 0

  # ---- publishing ----
  def _publish(self) -> None:
    self._publish_sp()
    self._publish_nav_route()
    self._publish_nav_instruction()

  def _publish_sp(self) -> None:
    msg = messaging.new_message('nkaoudNavigationSP')
    msg.valid = bool(self.sm['liveLocationKalman'].gpsOK)
    nav = msg.nkaoudNavigationSP
    nav.enabled = self._enabled
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
    nav.advisoryLaneChange = self._advisory_lane_side() or "none"
    nav.currentRoadClasses = ",".join(cur_step.road_classes) if cur_step else ""
    nav.upcomingRoadClasses = ",".join(upcoming.road_classes) if upcoming else ""
    nav.crossTrackDistance = float(self.cross_track_m)
    # missedManeuverCount is retired (cross-track covers missed turns); the
    # schema field stays and defaults to 0.

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
    advisory_str = str(nav.advisoryLaneChange)
    if advisory_str != self._last_logged_advisory:
      cloudlog.info(f"nkaoud_navd: advisoryLaneChange={advisory_str} (dist={nav.distanceToManeuver:.1f}m, lane={self.lane_current}/{self.lane_total})")
      self._last_logged_advisory = advisory_str

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
    """Route lateral INTENT -- what the route wants, with no permission
    gating. Every gate (NkaoudNavControlSteer, AutoLaneChange timer, lane
    change state, post-change cooldown, BSM/visual clearance, episode
    timeout) lives in desire_helper, which sees that state fresh at 20 Hz.

      * turnLeft/turnRight inside the turn-cue window.
      * keepLeft/keepRight -- a cautious lane-change bias -- inside the
        lane-keep window, only for ramp-class maneuvers and only when a
        medium+ confidence lane fix says we actually need to move.
    """
    cur_step = self._current_step()
    upcoming = self._upcoming_step()
    if cur_step is None or upcoming is None:
      return NavDesire.none
    dist = self._distance_to_maneuver()
    if dist <= 0.0:
      return NavDesire.none

    lane_keep_m, turn_cue_m = _ranges_for(upcoming.maneuver_type)
    modifier = upcoming.maneuver_modifier
    if dist <= turn_cue_m:
      if modifier in LEFT_TURN_MODIFIERS:
        return NavDesire.turnLeft
      if modifier in RIGHT_TURN_MODIFIERS:
        return NavDesire.turnRight
      return NavDesire.none

    if dist > lane_keep_m or upcoming.maneuver_type not in LANE_POSITION_MANEUVERS:
      return NavDesire.none
    if self.lane_conf not in STEER_LANE_CONFS or self.lane_total <= 0 or self.lane_current <= 0:
      return NavDesire.none
    side = self._route_side(cur_step, dist, modifier)
    if not side or self._positioned_for(side, dist, lane_keep_m, turn_cue_m):
      return NavDesire.none
    return NavDesire.keepLeft if side == "left" else NavDesire.keepRight

  def _advisory_lane_side(self) -> str:
    """Side of a lane move to cue with the UI's flashing arrow. Broader than
    the steering intent: covers every maneuver type, and falls back to the
    route's side when there is no usable lane fix (we can't tell whether
    we're already positioned, so cue the side maps-style). Empty inside the
    turn-cue window -- the turn arrow owns that zone."""
    cur_step = self._current_step()
    upcoming = self._upcoming_step()
    if cur_step is None or upcoming is None:
      return ""
    dist = self._distance_to_maneuver()
    if dist <= 0.0:
      return ""
    lane_keep_m, turn_cue_m = _ranges_for(upcoming.maneuver_type)
    if not (turn_cue_m < dist <= lane_keep_m):
      return ""
    side = self._route_side(cur_step, dist, upcoming.maneuver_modifier)
    if not side:
      return ""
    if self.lane_conf == "unknown" or self.lane_total <= 0 or self.lane_current <= 0:
      return side
    return "" if self._positioned_for(side, dist, lane_keep_m, turn_cue_m) else side

  def _positioned_for(self, side: str, dist: float, lane_keep_m: float, turn_cue_m: float) -> bool:
    """True when the current lane already satisfies the route's `side`
    requirement. Two fixed rules (replacing the old loose->strict lerp,
    which downstream could only ever consume as this boolean): the outer
    half of the window requires being in the correct half of the road, the
    inner half requires the outermost lane. Callers must have checked that
    a usable lane fix exists."""
    n = self.lane_total
    cur = self.lane_current
    if dist > (lane_keep_m + turn_cue_m) / 2:
      # Correct half; the middle lane of an odd count satisfies either side.
      return cur > n / 2 if side == "right" else cur <= math.ceil(n / 2)
    return cur == n if side == "right" else cur == 1

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
    if modifier in LEFT_SIDE_MODIFIERS:
      return "left"
    if modifier in RIGHT_SIDE_MODIFIERS:
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
