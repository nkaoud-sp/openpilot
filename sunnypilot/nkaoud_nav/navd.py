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

import csv
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
  Coordinate, closest_segment_index, distance_along_geometry, route_bearing_at, sample_route_ahead,
  total_geometry_length,
)
from openpilot.sunnypilot.nkaoud_nav.route_client import (
  Banner, RouteData, RouteFetchError, fetch_route,
)
from openpilot.sunnypilot.nkaoud_nav.share_client import (
  ShareFetchError, fetch_share_destination,
)
from openpilot.sunnypilot.selfdrive.controls.lib.lane_position import FILTER_MODE_NONE, LanePositionEstimator

NavDesire = custom.NkaoudNavigationSP.NavDesire


# Reroute thresholds (ported from old fork's bearing-misalignment detector).
BEARING_MISALIGN_THRESHOLD_DEG = 95.0
BEARING_MISALIGN_MIN_SPEED_MS = 5.0
BEARING_MISALIGN_COUNTER_MIN = 3

ARRIVAL_DISTANCE_M = 25.0          # consider the destination reached within this
MIN_REROUTE_INTERVAL_S = 8.0       # back off so reroutes don't spam the API
PARAM_REFRESH_S = 0.5              # throttle for per-tick param reads

# Navigation maneuver logging. When NkaoudNavDriveLogging is on, navd appends a
# CSV row every NAV_LOG_INTERVAL seconds to a per-drive file for later analysis.
# The mailer daemon reads NkaoudNavCurrentLog at drive end and (optionally)
# emails it. Keep NAV_LOG_DIR in sync with mailer.py.
NAV_LOG_DIR = os.environ.get("NKAOUD_NAV_LOG_DIR", "/data/media/0/nkaoud_nav_logs")
NAV_LOG_INTERVAL = 0.5
NAV_LOG_FIELDS = [
  "timestamp",
  "monotonic",
  "v_ego_ms",
  "latitude",
  "longitude",
  "bearing_deg",
  "enabled",
  "active",
  "on_route",
  "rerouting",
  "arrived",
  "route_id",
  "step_idx",
  "maneuver_type",
  "maneuver_modifier",
  "distance_to_maneuver_m",
  "maneuver_target_speed_ms",
  "recommended_desire",
  "recommended_lane_side",
  "lane_keep_distance_m",
  "advisory_lane_change",
  "advisory_block_reason",
  "current_road_classes",
  "upcoming_road_classes",
  "cross_track_m",
  "lane_current",
  "lane_total",
  "lane_conf",
  "bearing_misalign",
]

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

# Advance to the next step once we are this far PAST the current step's
# maneuver (signed along-step distance goes negative). Ported from the old
# fork's MANEUVER_TRANSITION_THRESHOLD -- the piece that recovers step
# tracking when we drive past a turn instead of taking it.
MANEUVER_TRANSITION_THRESHOLD_M = 10.0

# Steering keep* intent is restricted to the maneuver types where advance
# lane positioning actually pays off (long approach, hard-to-recover miss).
# Surface-street turns are left to the model + the turn cue; the UI advisory
# arrow still covers every maneuver type.
LANE_POSITION_MANEUVERS = ("off ramp", "on ramp", "fork", "merge")
# Lane-fix confidence required to INITIATE a steering bias. Any reading may
# stop one (that fails safe), but a low-confidence read must never start one.
STEER_LANE_CONFS = ("high", "medium")

# Highway-cruise lane preference (NkaoudNavHighwayLanePref): while cruising
# with no imminent maneuver, bias toward the preferred lane. Pure intent, like
# everything else here -- desire_helper applies every safety gate (BSM,
# visual, cooldown, episode budget). This branch now accepts any Mapbox road
# class and trusts the same medium+ lane fix as maneuver positioning.
HIGHWAY_CRUISE_MIN_SPEED_MS = 60.0 / 3.6   # ~60 km/h; below this we don't auto-position
HIGHWAY_CRUISE_MIN_DIST_M = 1500.0         # only cruise-bias when the next maneuver is at least this far
HIGHWAY_LANE_PREF_LEFT = 0
HIGHWAY_LANE_PREF_CENTER = 1
HIGHWAY_LANE_PREF_RIGHT = 2
HIGHWAY_LANE_PREF_NONE = 3   # no cruise-lane bias at all (exit/fork positioning still applies)

# Without a usable lane fix the maneuver advisory can't tell whether we're
# already positioned, so cap how early that (possibly redundant) cue starts.
# With a fix the advisory runs the full lane-keep window and simply stops
# once we're positioned -- it must mirror the steering intent so a keep*
# bias is never active without the arrow explaining it.
ADVISORY_NO_FIX_CUE_M = 500.0

# Phase 11: "Share" destination -- HTTP fetch when the picker writes a new
# NkaoudNavShareTrigger token. Retries on failure are bounded so a wrong
# URL doesn't hammer the endpoint forever.
SHARE_FETCH_RETRY_S = 15.0
SHARE_FETCH_MAX_ATTEMPTS = 4


def _ranges_for(maneuver_type: str) -> tuple[float, float]:
  return MANEUVER_RANGES.get((maneuver_type or "").strip(), DEFAULT_RANGES)


# Device-frame route path published for mode B (NavPathAssist). Must be sampled
# at NAV_PATH_SPACING_M (NavPathAssist assumes this spacing to convert indices to
# arc length); needs at least a few points to measure curvature.
NAV_PATH_HORIZON_M = 60.0
NAV_PATH_SPACING_M = 2.0
NAV_PATH_MIN_POINTS = 5


def _bearing_delta(a: float, b: float) -> float:
  d = (a - b + 540.0) % 360.0 - 180.0
  return abs(d)


def _signed_turn_angle(before: float | None, after: float | None) -> float:
  """Signed turn angle (deg) from Mapbox maneuver bearings, normalized to
  [-180, 180]. Bearings are compass degrees (clockwise from north), so a
  positive result is a clockwise = right turn, negative a left turn. Returns
  0.0 when either bearing is missing (e.g. depart/arrive steps)."""
  if before is None or after is None:
    return 0.0
  return (after - before + 540.0) % 360.0 - 180.0


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
    self._highway_pref: int = HIGHWAY_LANE_PREF_CENTER
    self._lane_edge_filter_mode: int = FILTER_MODE_NONE
    self._next_param_t: float = 0.0
    self._last_logged_desire: str = "none"
    self._last_logged_modifier: str = ""
    self._last_logged_advisory: str = "none"
    self._drive_logging: bool = False
    self._nav_log_last_t: float = 0.0
    self._nav_log_dir: str = NAV_LOG_DIR
    self._nav_log_current_path: str | None = None

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
      self.lane_current, self.lane_total, self.lane_conf = self.lane_position_est.update(
        self.sm['modelV2'], filter_mode=self._lane_edge_filter_mode,
      )

    now = time.monotonic()
    if now >= self._next_param_t:
      self._next_param_t = now + PARAM_REFRESH_S
      self._enabled = self.params.get_bool("NkaoudNavEnabled")
      try:
        self._highway_pref = int(self.params.get("NkaoudNavHighwayLanePref", return_default=True))
      except (TypeError, ValueError):
        self._highway_pref = HIGHWAY_LANE_PREF_CENTER
      try:
        self._lane_edge_filter_mode = int(self.params.get("LaneEdgeFilterMode", return_default=True))
      except (TypeError, ValueError):
        self._lane_edge_filter_mode = FILTER_MODE_NONE
      self._drive_logging = self.params.get_bool("NkaoudNavDriveLogging")

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

    # Whole-route progress -- kept only for the remaining-distance readout.
    self.last_distance_along = distance_along_geometry(self.route.geometry, self.last_pos)
    # Perpendicular distance to the route for the cross-track reroute net.
    _idx, perp, _t = closest_segment_index(self.route.geometry, self.last_pos)
    self.cross_track_m = perp

    # Advance step_idx forward-only, driven by actually passing the maneuver
    # (per-step signed distance) rather than by where the global-nearest
    # projection lands. When we drive PAST a turn the whole-route projection
    # can pin to the pre-turn segment and freeze distance_to_maneuver; the
    # transition below still fires because it measures against the step's own
    # geometry and lets the along-step distance go negative.
    while self.step_idx + 1 < len(self.route.steps) and self._should_transition_to_next_step():
      self.step_idx += 1

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
    # Signed turn angle of the upcoming maneuver (deg, + = right, - = left).
    # Pure geometry -- no gating here; the turn assist downstream decides
    # whether/how to use it. 0 when there is no upcoming step or no bearings.
    nav.maneuverTurnAngle = float(_signed_turn_angle(
      upcoming.maneuver_bearing_before, upcoming.maneuver_bearing_after)) if upcoming is not None else 0.0

    # Route path ahead in device frame (mode B, NavPathAssist). Real geometry
    # from the Mapbox polyline at the current pose -- only when we are on-route
    # with a good fix and heading, so a stale/misaligned pose never steers.
    path_pts: list[tuple[float, float]] = []
    if self.route is not None and nav.onRoute and msg.valid and self.last_bearing is not None:
      path_pts = sample_route_ahead(self.route.geometry, self.last_pos, self.last_bearing,
                                    NAV_PATH_HORIZON_M, NAV_PATH_SPACING_M)
    nav.navPathValid = len(path_pts) >= NAV_PATH_MIN_POINTS
    if path_pts:
      xs = nav.init('navPathX', len(path_pts))
      ys = nav.init('navPathY', len(path_pts))
      for i, (px, py) in enumerate(path_pts):
        xs[i] = float(px)
        ys[i] = float(py)

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
    advisory_side, advisory_reason = self._advisory_lane_change()
    nav.advisoryLaneChange = advisory_side or "none"
    nav.advisoryLaneChangeBlockReason = advisory_reason
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

    self._maybe_log(nav)

    self.pm.send('nkaoudNavigationSP', msg)

  def _maybe_log(self, nav) -> None:
    """Append one CSV row for the current maneuver state, rate-limited to
    NAV_LOG_INTERVAL. Gated on the NkaoudNavDriveLogging param, and only while
    actively navigating a route (a destination + fetched route)."""
    if not self._drive_logging:
      return
    if not nav.active:
      return

    now = time.monotonic()
    if now - self._nav_log_last_t < NAV_LOG_INTERVAL:
      return
    self._nav_log_last_t = now

    path = self._nav_log_path()
    if path is None:
      return

    pos = self.last_pos
    row = {
      "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
      "monotonic": f"{now:.2f}",
      "v_ego_ms": f"{self.last_v_ego:.2f}",
      "latitude": f"{pos.latitude:.7f}" if pos is not None else "",
      "longitude": f"{pos.longitude:.7f}" if pos is not None else "",
      "bearing_deg": f"{self.last_bearing:.1f}" if self.last_bearing is not None else "",
      "enabled": nav.enabled,
      "active": nav.active,
      "on_route": nav.onRoute,
      "rerouting": nav.rerouting,
      "arrived": self.arrived,
      "route_id": nav.routeId,
      "step_idx": self.step_idx,
      "maneuver_type": nav.maneuverType,
      "maneuver_modifier": nav.maneuverModifier,
      "distance_to_maneuver_m": f"{nav.distanceToManeuver:.1f}",
      "maneuver_target_speed_ms": f"{nav.maneuverTargetSpeed:.2f}",
      "recommended_desire": str(nav.recommendedDesire),
      "recommended_lane_side": nav.recommendedLaneSide,
      "lane_keep_distance_m": f"{nav.laneKeepDistance:.1f}",
      "advisory_lane_change": nav.advisoryLaneChange,
      "advisory_block_reason": nav.advisoryLaneChangeBlockReason,
      "current_road_classes": nav.currentRoadClasses,
      "upcoming_road_classes": nav.upcomingRoadClasses,
      "cross_track_m": f"{nav.crossTrackDistance:.2f}",
      "lane_current": self.lane_current,
      "lane_total": self.lane_total,
      "lane_conf": self.lane_conf,
      "bearing_misalign": self.bearing_misalign_counter,
    }

    try:
      write_header = not os.path.exists(path) or os.path.getsize(path) == 0
      with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NAV_LOG_FIELDS)
        if write_header:
          writer.writeheader()
        writer.writerow(row)
    except OSError:
      cloudlog.exception("nkaoud_navd: nav log write failed")

  def _nav_log_path(self) -> str | None:
    """Per-drive CSV path. navd is a fresh process each drive, so the first
    call creates a new timestamped file and records it in the params the
    mailer reads at drive end."""
    if self._nav_log_current_path is not None:
      return self._nav_log_current_path

    try:
      os.makedirs(self._nav_log_dir, exist_ok=True)
    except OSError:
      cloudlog.exception(f"nkaoud_navd: cannot create nav log dir {self._nav_log_dir}")
      return None

    filename = f"nkaoud_nav_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.csv"
    path = os.path.join(self._nav_log_dir, filename)
    self._nav_log_current_path = path
    self.params.put("NkaoudNavCurrentLog", path)
    self.params.put("NkaoudNavLastDriveLog", path)
    return path

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

  def _along_step(self, idx: int) -> float:
    """Distance covered along step `idx`'s own geometry. Projecting onto the
    short per-step polyline (instead of the whole route) keeps the nearest
    point local, so it can't snap to a far-away segment and freeze."""
    step = self.route.steps[idx]
    if len(step.geometry) >= 2:
      return distance_along_geometry(step.geometry, self.last_pos)
    # No usable per-step polyline: fall back to whole-route cumulative progress.
    start = self.route.cumulative_step_distance[idx] if idx < len(self.route.cumulative_step_distance) else 0.0
    return max(0.0, self.last_distance_along - start)

  def _maneuver_ref_distance(self, idx: int) -> float:
    """Distance from the step's start to its maneuver point, measured along the
    step's OWN geometry. Mapbox's step.distance (routed length) can exceed the
    decimated polyline length -- notably on ramps -- and _along_step clamps to
    the polyline, so measuring against step.distance leaves a residual offset
    that never counts down (distance freezes a few metres short and the step
    never transitions). The geometry endpoint IS the maneuver, so reference
    its length."""
    step = self.route.steps[idx]
    if len(step.geometry) >= 2:
      return total_geometry_length(step.geometry)
    return step.distance

  def _path_min_distance(self, idx: int) -> float | None:
    """Perpendicular distance from the vehicle to step `idx`'s geometry, or
    None when that step has no usable polyline."""
    step = self.route.steps[idx]
    if len(step.geometry) < 2:
      return None
    _i, perp, _t = closest_segment_index(step.geometry, self.last_pos)
    return perp

  def _should_transition_to_next_step(self) -> bool:
    """Whether to advance from the current step to the next. Ported from the
    old fork: advance once we are more than MANEUVER_TRANSITION_THRESHOLD_M
    PAST the maneuver (signed along-step distance negative), and in the
    ambiguous zone right around the maneuver, once the next step's geometry is
    the closer one. Caller guarantees step_idx + 1 is in range."""
    signed = self._maneuver_ref_distance(self.step_idx) - self._along_step(self.step_idx)
    if signed < -MANEUVER_TRANSITION_THRESHOLD_M:
      return True
    if signed > MANEUVER_TRANSITION_THRESHOLD_M:
      return False
    cur_d = self._path_min_distance(self.step_idx)
    nxt_d = self._path_min_distance(self.step_idx + 1)
    return cur_d is not None and nxt_d is not None and nxt_d < cur_d

  def _distance_to_maneuver(self) -> float:
    if self.route is None or not self.route.steps or self.last_pos is None:
      return 0.0
    idx = min(self.step_idx, len(self.route.steps) - 1)
    return max(0.0, self._maneuver_ref_distance(idx) - self._along_step(idx))

  def _recommended_desire(self):
    """Route lateral INTENT -- what the route wants, with no permission
    gating. Every gate (NkaoudNavControlSteer, AutoLaneChange timer, lane
    change state, post-change cooldown, BSM/visual clearance, episode
    timeout) lives in desire_helper, which sees that state fresh at 20 Hz.

      * turnLeft/turnRight inside the turn-cue window.
      * keepLeft/keepRight -- a cautious lane-change bias -- inside the
        lane-keep window, only for ramp-class maneuvers and only when a
        medium+ confidence lane fix says we actually need to move.
      * keepLeft/keepRight toward the preferred highway lane while cruising
        with no imminent maneuver (high-confidence fix only).
    """
    cur_step = self._current_step()
    if cur_step is None:
      return NavDesire.none
    upcoming = self._upcoming_step()
    dist = self._distance_to_maneuver()

    if upcoming is not None and dist > 0.0:
      lane_keep_m, turn_cue_m = _ranges_for(upcoming.maneuver_type)
      modifier = upcoming.maneuver_modifier
      if dist <= turn_cue_m:
        if modifier in LEFT_TURN_MODIFIERS:
          return NavDesire.turnLeft
        if modifier in RIGHT_TURN_MODIFIERS:
          return NavDesire.turnRight
        return NavDesire.none
      if dist <= lane_keep_m:
        # Inside a maneuver window the maneuver owns lateral -- never
        # cruise-bias here, even when no positioning move is wanted.
        if (upcoming.maneuver_type in LANE_POSITION_MANEUVERS
            and self.lane_conf in STEER_LANE_CONFS and self.lane_total > 0 and self.lane_current > 0):
          side = self._route_side(cur_step, dist, modifier)
          if side and not self._positioned_for(side, dist, lane_keep_m, turn_cue_m):
            return NavDesire.keepLeft if side == "left" else NavDesire.keepRight
        return NavDesire.none

    side = self._cruise_lane_side(cur_step, dist)
    if side:
      return NavDesire.keepLeft if side == "left" else NavDesire.keepRight
    return NavDesire.none

  def _cruise_lane_side(self, cur_step, dist: float) -> str:
    """Side of a move toward the preferred highway cruising lane
    (NkaoudNavHighwayLanePref), or '' when none is wanted. Runs on any
    Mapbox road class, at speed, with no imminent maneuver, and with a
    medium+ lane fix. Center uses ceil(N/2): 3 lanes -> 2, 4 -> 2
    (center-left), 5 -> 3. "None" disables cruise-lane biasing entirely."""
    if self._highway_pref == HIGHWAY_LANE_PREF_NONE:
      return ""
    if 0.0 < dist < HIGHWAY_CRUISE_MIN_DIST_M:
      return ""
    if self.last_v_ego < HIGHWAY_CRUISE_MIN_SPEED_MS:
      return ""
    if self.lane_conf not in STEER_LANE_CONFS or self.lane_total <= 1 or self.lane_current <= 0:
      return ""
    if self._highway_pref == HIGHWAY_LANE_PREF_LEFT:
      target = 1
    elif self._highway_pref == HIGHWAY_LANE_PREF_RIGHT:
      target = self.lane_total
    else:
      target = math.ceil(self.lane_total / 2)
    if self.lane_current == target:
      return ""
    return "left" if self.lane_current > target else "right"

  def _cruise_lane_preference_advisory(self, cur_step, dist: float) -> tuple[str, str]:
    """Broader highway-preference cue for the UI. Returns (side, reason).
    `reason` is empty only when navd would also allow the keep* desire."""
    if self._highway_pref == HIGHWAY_LANE_PREF_NONE:
      return "", ""
    if self._highway_pref == HIGHWAY_LANE_PREF_LEFT:
      side = "left"
      target = 1
    elif self._highway_pref == HIGHWAY_LANE_PREF_RIGHT:
      side = "right"
      target = self.lane_total if self.lane_total > 0 else 0
    else:
      if self.lane_total <= 1 or self.lane_current <= 0:
        return "", ""
      target = math.ceil(self.lane_total / 2)
      if self.lane_current == target:
        return "", ""
      side = "left" if self.lane_current > target else "right"

    if target > 0 and self.lane_current == target:
      return "", ""
    if 0.0 < dist < HIGHWAY_CRUISE_MIN_DIST_M:
      return side, "Next maneuver"
    if self.last_v_ego < HIGHWAY_CRUISE_MIN_SPEED_MS:
      return side, "Speed"
    if self.lane_conf not in STEER_LANE_CONFS or self.lane_total <= 1 or self.lane_current <= 0:
      return side, "Lane confidence"
    return side, ""

  def _advisory_lane_change(self) -> tuple[str, str]:
    """Side of a lane move to cue with the UI's flashing arrow. Broader than
    the steering intent for maneuvers (covers every maneuver type, and falls
    back to the route's side when there is no usable lane fix). For highway
    cruising, returns a cue even when preference is currently blocked so the
    UI can explain why. Empty inside the turn-cue window -- the turn arrow owns
    that zone."""
    cur_step = self._current_step()
    if cur_step is None:
      return "", ""
    upcoming = self._upcoming_step()
    dist = self._distance_to_maneuver()

    if upcoming is not None and dist > 0.0:
      lane_keep_m, turn_cue_m = _ranges_for(upcoming.maneuver_type)
      # A maneuver owns lateral for its whole lane-keep window -- never fall
      # through to the highway lane-preference advisory here, which can point
      # opposite the maneuver (e.g. a left-lane cruise bias on the approach to
      # a right turn). Inside the turn-cue zone the turn arrow owns the
      # display, so emit nothing; in the outer lane-keep zone, cue the route's
      # side.
      if dist <= lane_keep_m:
        if dist <= turn_cue_m:
          return "", ""
        side = self._route_side(cur_step, dist, upcoming.maneuver_modifier)
        if side:
          if self.lane_conf == "unknown" or self.lane_total <= 0 or self.lane_current <= 0:
            return (side, "Lane confidence") if dist <= ADVISORY_NO_FIX_CUE_M else ("", "")
          if not self._positioned_for(side, dist, lane_keep_m, turn_cue_m):
            return side, ""
        return "", ""

    return self._cruise_lane_preference_advisory(cur_step, dist)

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
