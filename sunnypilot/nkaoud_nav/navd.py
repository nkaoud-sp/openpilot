#!/usr/bin/env python3
"""
Navigation daemon v2 — clean rewrite for nkaoud-sp fork.

Lateral logic — single priority chain, one desire per tick:
  TURN  (within turn_cue_m, sharp modifier, correct lane)  → turnLeft / turnRight
  POSITION  (within lane_keep_m)                           → keepLeft / keepRight
  HIGHWAY_DEFAULT  (motorway cruising, configurable)       → keepLeft / keepRight
  NONE                                                     → none

Design rules:
  - All nav lateral commands use keepLeft/keepRight (conservative lane change).
    laneChangeLeft/Right are never sent by this daemon.
  - turnLeft/turnRight only fires when lane position confirms the car is
    already in the correct (outermost) lane. If confidence is low/unknown
    and we are within half the turn-cue distance, fire anyway to avoid
    missing the maneuver entirely.
  - Highway default lane is configurable and suppressed by a driver blinker
    on the highway. It is re-armed when the next TURN or POSITION desire fires.
  - Driver blinker conflict: nav desire is suppressed when driver blinkers
    in the opposite direction.

Longitudinal logic:
  Geometry-based turn speed cap:
    v_target = sqrt(MAX_LAT_ACCEL * tolerance / curvature)
  Applied as a linear ramp starting at SLOW_START_FACTOR * turn_cue_m,
  reaching v_target at turn_cue_m. Gated by NkaoudNavControlSpeed.
  Tolerance is NkaoudNavTurnTolerance (50–150, default 100 = 1.0×).

Rerouting:
  - Bearing misalignment: >95° for 3 consecutive ticks at ≥5 m/s.
  - Cross-track: >30 m for CROSS_TRACK_COUNTER_ON ticks.
    Counter increments on violation, decrements on clear (hysteresis).
    This prevents false reroutes from momentary drift corrections.
"""
from __future__ import annotations

import math
import os
import threading
import time

import cereal.messaging as messaging
from cereal import custom, log
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.nkaoud_nav.geometry import (
  Coordinate, closest_segment_index, distance_along_geometry,
  route_bearing_at, total_geometry_length, bearing_between,
  EARTH_MEAN_RADIUS,
)
from openpilot.sunnypilot.nkaoud_nav.route_client import (
  Banner, RouteData, RouteFetchError, Step, fetch_route,
)
from openpilot.sunnypilot.nkaoud_nav.share_client import (
  ShareFetchError, fetch_share_destination,
)
from openpilot.sunnypilot.selfdrive.controls.lib.lane_position import (
  FILTER_MODE_WIDTH,
  LanePositionEstimator,
)

NavDesire = custom.NkaoudNavigationSP.NavDesire

# ---------------------------------------------------------------------------
# Modifier classification
# ---------------------------------------------------------------------------
SHARP_TURN_LEFT: frozenset[str] = frozenset({"left", "sharpLeft", "uturn"})
SHARP_TURN_RIGHT: frozenset[str] = frozenset({"right", "sharpRight"})
LEFT_SIDE: frozenset[str] = SHARP_TURN_LEFT | frozenset({"slightLeft"})
RIGHT_SIDE: frozenset[str] = SHARP_TURN_RIGHT | frozenset({"slightRight"})

# ---------------------------------------------------------------------------
# Zone thresholds by road class: (turn_cue_m, lane_keep_m)
#   turn_cue_m   — distance at which turnLeft/turnRight fires
#   lane_keep_m  — distance at which keepLeft/keepRight fires
# ---------------------------------------------------------------------------
ZONE_THRESHOLDS: dict[str, tuple[float, float]] = {
  "highway": (180.0, 1000.0),
  "ramp":    (120.0, 600.0),
  "surface": (60.0,  250.0),
}
SLOW_START_FACTOR = 4.0  # slow-down starts at this multiple of turn_cue_m

HIGHWAY_ROAD_CLASSES: frozenset[str] = frozenset({"motorway", "motorway_link", "trunk"})
RAMP_MANEUVER_TYPES: frozenset[str] = frozenset({"off ramp", "on ramp", "fork", "merge"})

# ---------------------------------------------------------------------------
# Longitudinal
# ---------------------------------------------------------------------------
MAX_LAT_ACCEL_MS2 = 2.5        # m/s² — default max lateral accel for speed calc
MIN_CURVATURE = 0.002          # 1/m — ignore near-straight geometry (radius > 500 m)
CURVATURE_LOOKAHEAD_M = 100.0  # only look at first 100 m of upcoming step

# UI multiple_button_item_sp stores a 0-based index. These tables convert
# index → actual value. Index 1 is the middle/normal option in each case.
_TOLERANCE_IDX_TO_PCT = [75, 100, 125]          # % of v_target; index default = 1 (100 %)
_ACCEL_IDX_TO_INT100 = [200, 250, 300]          # int*100 m/s²; index default = 1 (2.5 m/s²)
_LANE_CHANGE_COOLDOWN_IDX_TO_S = [0.0, 1.0, 2.0, 3.0, 5.0]

# ---------------------------------------------------------------------------
# Highway default
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Visual vehicle detector gate
# ---------------------------------------------------------------------------
# The adjacent lane must be clear for this many consecutive ticks before a
# keepLeft/keepRight desire is allowed. "Clear" means BOTH:
#   1. Blindspot on that side is false
#   2. VVD zone probability on that side is below VVD_CONF_THRESHOLD
# At 5 Hz loop rate: 20 ticks = 4 seconds.
SIDE_CLEAR_TICKS_REQUIRED = 20   # 4 s × 5 Hz
VVD_CONF_THRESHOLD      = 0.60  # vehicle confidence >= this → lane is blocked

HIGHWAY_DEFAULT_MIN_SPEED_MS = 60.0 / 3.6   # ~16.7 m/s

# Lane preference indices (NkaoudNavHighwayLanePref multiple_button_item_sp)
_LANE_PREF_RIGHTMOST = 0
_LANE_PREF_CENTER    = 1   # default
_LANE_PREF_LEFTMOST  = 2

# ---------------------------------------------------------------------------
# Rerouting
# ---------------------------------------------------------------------------
BEARING_MISALIGN_THRESHOLD_DEG = 95.0
BEARING_MISALIGN_MIN_SPEED_MS = 5.0
BEARING_MISALIGN_COUNTER_MIN = 3

CROSS_TRACK_THRESHOLD_M = 30.0
CROSS_TRACK_COUNTER_ON = 25    # ~5 s at 5 Hz to trigger reroute
CROSS_TRACK_COUNTER_OFF = 10   # ~2 s back on route before counter resets to 0

MIN_REROUTE_INTERVAL_S = 8.0
ARRIVAL_DISTANCE_M = 25.0

# ---------------------------------------------------------------------------
# Share fetch
# ---------------------------------------------------------------------------
SHARE_FETCH_RETRY_S = 15.0
SHARE_FETCH_MAX_ATTEMPTS = 4

# ---------------------------------------------------------------------------
# Param refresh
# ---------------------------------------------------------------------------
PARAMS_REFRESH_INTERVAL = 5    # ticks (5 Hz loop → refresh every 1 s)

# ---------------------------------------------------------------------------
# Token drop paths (copyparty / manual delivery)
# ---------------------------------------------------------------------------
TOKEN_DROP_PATHS = (
  "/data/openpilot/nkaoud_mapbox_token.txt",
  "/data/openpilot/sunnypilot/nkaoud_nav/mapbox_token.txt",
)


# ===========================================================================
# Geometry helpers
# ===========================================================================

def _road_class(step: Step | None) -> str:
  if step is None:
    return "surface"
  classes = set(step.road_classes)
  if classes & HIGHWAY_ROAD_CLASSES:
    return "highway"
  if (step.maneuver_type or "").strip() in RAMP_MANEUVER_TYPES:
    return "ramp"
  return "surface"


def _modifier_to_side(modifier: str) -> str:
  if modifier in LEFT_SIDE:
    return "left"
  if modifier in RIGHT_SIDE:
    return "right"
  return ""


def _menger_curvature(a: Coordinate, b: Coordinate, c: Coordinate) -> float:
  """Signed curvature (1/m) of the arc through three coordinates using Menger's formula."""
  cos_lat = math.cos(math.radians(b.latitude))
  dx1 = math.radians(b.longitude - a.longitude) * cos_lat * EARTH_MEAN_RADIUS
  dy1 = math.radians(b.latitude - a.latitude) * EARTH_MEAN_RADIUS
  dx2 = math.radians(c.longitude - b.longitude) * cos_lat * EARTH_MEAN_RADIUS
  dy2 = math.radians(c.latitude - b.latitude) * EARTH_MEAN_RADIUS
  cross = abs(dx1 * dy2 - dy1 * dx2)
  len_ab = math.sqrt(dx1 ** 2 + dy1 ** 2)
  len_bc = math.sqrt(dx2 ** 2 + dy2 ** 2)
  ac = a.distance_to(c)
  denom = len_ab * len_bc * ac
  if denom < 1e-6:
    return 0.0
  return 2.0 * cross / denom


def _max_curvature(geometry: list[Coordinate], max_dist_m: float = CURVATURE_LOOKAHEAD_M) -> float:
  """Peak curvature (1/m) over the first max_dist_m of the geometry."""
  if len(geometry) < 3:
    return 0.0
  best = 0.0
  traveled = 0.0
  for i in range(len(geometry) - 2):
    seg = geometry[i].distance_to(geometry[i + 1])
    if traveled + seg > max_dist_m:
      break
    traveled += seg
    c = _menger_curvature(geometry[i], geometry[i + 1], geometry[i + 2])
    if c > best:
      best = c
  return best


def _bearing_delta(a: float, b: float) -> float:
  return abs((a - b + 540.0) % 360.0 - 180.0)


# ===========================================================================
# Token helpers
# ===========================================================================

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


def _read_destination(params: Params) -> Coordinate | None:
  d = params.get("NkaoudNavDestination")
  if not isinstance(d, dict):
    return None
  if "latitude" not in d or "longitude" not in d:
    return None
  return Coordinate(float(d["latitude"]), float(d["longitude"]))


def _read_token(params: Params) -> str:
  return (params.get("NkaoudNavMapboxToken") or "").strip()


def _location_from_llk(llk) -> tuple[Coordinate | None, float | None, float]:
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


# ===========================================================================
# Route fetcher (unchanged from v1 — it works well)
# ===========================================================================

class RouteFetcher:
  def __init__(self) -> None:
    self._thread: threading.Thread | None = None
    self._result: RouteData | None = None
    self._error: str | None = None
    self._request_id: int = 0
    self._lock = threading.Lock()

  def in_flight(self) -> bool:
    with self._lock:
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


class ShareFetcher:
  def __init__(self) -> None:
    self._thread: threading.Thread | None = None
    self._result: dict | None = None
    self._error: str | None = None
    self._request_id: int = 0
    self._lock = threading.Lock()

  def in_flight(self) -> bool:
    with self._lock:
      return self._thread is not None and self._thread.is_alive()

  def submit(self, url: str) -> int:
    with self._lock:
      self._request_id += 1
      rid = self._request_id
      self._result = None
      self._error = None
    self._thread = threading.Thread(
      target=self._run, args=(rid, url),
      name="nkaoud_navd_share_fetch", daemon=True,
    )
    self._thread.start()
    return rid

  def _run(self, rid: int, url: str) -> None:
    try:
      result = fetch_share_destination(url)
      with self._lock:
        if rid == self._request_id:
          self._result = result
    except ShareFetchError as e:
      with self._lock:
        if rid == self._request_id:
          self._error = str(e)

  def take_result(self) -> tuple[dict | None, str | None]:
    with self._lock:
      r, e = self._result, self._error
      self._result = None
      self._error = None
      return r, e


# ===========================================================================
# Main navigation daemon
# ===========================================================================

class NkaoudNavd:
  def __init__(self) -> None:
    self.params = Params()
    self.sm = messaging.SubMaster(['liveLocationKalman', 'modelV2', 'carState', 'visualVehicleDetectorStateSP'])
    self.pm = messaging.PubMaster(['nkaoudNavigationSP', 'navRoute', 'navInstruction'])
    self.rk = Ratekeeper(5.0)

    # Route state
    self.fetcher = RouteFetcher()
    self.share_fetcher = ShareFetcher()
    self.route: RouteData | None = None
    self.destination: Coordinate | None = None
    self.step_idx: int = 0
    self.arrived: bool = False
    self.rerouting: bool = False
    self.last_route_fetch_t: float = 0.0
    self._prev_step_idx: int = 0

    # Position / speed
    self.last_pos: Coordinate | None = None
    self.last_bearing: float | None = None
    self.last_v_ego: float = 0.0
    self.last_distance_along: float = 0.0

    # Lane position (from LanePositionEstimator)
    self.lane_position_est = LanePositionEstimator()
    self.lane_current: int = 0
    self.lane_total: int = 0
    self.lane_conf: str = "unknown"

    # Driver blinker state (for conflict check + highway suppression)
    self.left_blinker: bool = False
    self.right_blinker: bool = False

    # Blind spot state (from carState — used to block keepLeft/keepRight)
    self.left_blindspot: bool = False
    self.right_blindspot: bool = False

    # Side-clear gate — counts consecutive ticks each side has been clear by
    # BOTH signals together: blindspot false + VVD probability below threshold.
    # keepLeft/keepRight is suppressed until the side has been clear for
    # SIDE_CLEAR_TICKS_REQUIRED ticks (4 s).
    self._left_clear_ticks: int = 0
    self._right_clear_ticks: int = 0

    # Reroute counters
    self.bearing_misalign_counter: int = 0
    self.cross_track_m: float = 0.0
    self.cross_track_counter: int = 0   # +1 on violation, -1 on clear (hysteresis)

    # Highway default suppression
    # Suppressed when driver manually blinkers on the highway outside a nav zone.
    # Re-armed when any TURN or POSITION nav desire fires.
    self._highway_suppressed: bool = False

    # Share trigger
    self._last_share_trigger: str | None = None
    self._share_next_retry_t: float = 0.0
    self._share_attempts: int = 0

    # Cached params (refreshed every PARAMS_REFRESH_INTERVAL ticks)
    self._tick: int = 0
    self._steer_enabled: bool = False
    self._speed_enabled: bool = False
    self._highway_default_enabled: bool = False
    self._turn_tolerance: float = 1.0        # NkaoudNavTurnTolerance / 100
    self._max_lat_accel: float = MAX_LAT_ACCEL_MS2
    self._lane_change_cooldown_s: float = 2.0
    self._keep_cooldown_until: float = 0.0
    self._last_lane_current_observed: int = 0
    self._refresh_params()

  # -------------------------------------------------------------------------
  # Param refresh
  # -------------------------------------------------------------------------

  def _refresh_params(self) -> None:
    self._steer_enabled = self.params.get_bool("NkaoudNavControlSteer")
    self._speed_enabled = self.params.get_bool("NkaoudNavControlSpeed")
    self._highway_default_enabled = self.params.get_bool("NkaoudNavHighwayDefault")

    raw_tol = self.params.get("NkaoudNavTurnTolerance")
    tol_idx = int(raw_tol) if raw_tol and raw_tol.isdigit() else 1
    tol_pct = _TOLERANCE_IDX_TO_PCT[max(0, min(tol_idx, len(_TOLERANCE_IDX_TO_PCT) - 1))]
    self._turn_tolerance = tol_pct / 100.0

    raw_accel = self.params.get("NkaoudNavMaxLatAccel")
    accel_idx = int(raw_accel) if raw_accel and raw_accel.isdigit() else 1
    accel_int100 = _ACCEL_IDX_TO_INT100[max(0, min(accel_idx, len(_ACCEL_IDX_TO_INT100) - 1))]
    self._max_lat_accel = accel_int100 / 100.0

    raw_cooldown = self.params.get("NkaoudNavLaneChangeCooldown")
    cooldown_idx = int(raw_cooldown) if raw_cooldown and raw_cooldown.isdigit() else 2
    self._lane_change_cooldown_s = _LANE_CHANGE_COOLDOWN_IDX_TO_S[
      max(0, min(cooldown_idx, len(_LANE_CHANGE_COOLDOWN_IDX_TO_S) - 1))
    ]

    raw_pref = self.params.get("NkaoudNavHighwayLanePref")
    self._highway_lane_pref = int(raw_pref) if raw_pref and raw_pref.isdigit() else _LANE_PREF_CENTER

  # -------------------------------------------------------------------------
  # Main loop
  # -------------------------------------------------------------------------

  def step(self) -> None:
    self._tick += 1
    self.sm.update(0)

    if self._tick % PARAMS_REFRESH_INTERVAL == 0:
      _maybe_import_token_file(self.params)
      self._refresh_params()

    # Update position and speed from Kalman filter
    pos, bearing, v_ego = _location_from_llk(self.sm['liveLocationKalman'])
    if pos is not None:
      self.last_pos = pos
    if bearing is not None:
      self.last_bearing = bearing
    self.last_v_ego = v_ego

    # Update lane position from model
    if self.sm.updated['modelV2']:
      mv2 = self.sm['modelV2']
      self.lane_current, self.lane_total, self.lane_conf = self.lane_position_est.update(
        mv2, filter_mode=FILTER_MODE_WIDTH,
      )
      self._update_lane_change_cooldown()

    # Read blinker + blind spot state from carState
    cs = self.sm['carState']
    self.left_blinker = cs.leftBlinker
    self.right_blinker = cs.rightBlinker
    self.left_blindspot = cs.leftBlindspot
    self.right_blindspot = cs.rightBlindspot

    # Compute VVD clear state from per-zone probability. Each Zone in
    # classifier.zones has a name ("left"/"right"), probability (Float32, 0-1),
    # and hasProbability flag. A zone with probability >= VVD_CONF_THRESHOLD
    # (0.60) is treated as blocked.
    left_vvd_clear = False
    right_vvd_clear = False
    if self.sm.updated['visualVehicleDetectorStateSP'] and self.sm.valid['visualVehicleDetectorStateSP']:
      vvd = self.sm['visualVehicleDetectorStateSP']
      for zone in vvd.classifier.zones:
        if not zone.hasProbability:
          continue
        prob = float(zone.probability)
        if zone.name == "left":
          left_vvd_clear = prob < VVD_CONF_THRESHOLD
        elif zone.name == "right":
          right_vvd_clear = prob < VVD_CONF_THRESHOLD

    # Require blindspot and VVD to stay clear simultaneously for the full
    # window. Any violation resets the side's counter immediately.
    if (not self.left_blindspot) and left_vvd_clear:
      self._left_clear_ticks = min(self._left_clear_ticks + 1, SIDE_CLEAR_TICKS_REQUIRED)
    else:
      self._left_clear_ticks = 0

    if (not self.right_blindspot) and right_vvd_clear:
      self._right_clear_ticks = min(self._right_clear_ticks + 1, SIDE_CLEAR_TICKS_REQUIRED)
    else:
      self._right_clear_ticks = 0

    # Handle destination changes
    new_dest = _read_destination(self.params)
    if not self._same_destination(new_dest):
      self.destination = new_dest
      self.route = None
      self.step_idx = 0
      self.arrived = False
      self.bearing_misalign_counter = 0
      self.cross_track_counter = 0
      self._highway_suppressed = False
      self._keep_cooldown_until = 0.0
      self._try_fetch_initial()

    # Route fetch lifecycle
    self._drain_fetcher()
    self._handle_share_trigger()

    # Progress and reroute
    if self.route is not None and self.last_pos is not None:
      self._update_progress()
      self._check_reroute()

    # Update highway suppression from blinker
    self._update_highway_suppression()

    self._publish()

  def run(self) -> None:
    while True:
      self.step()
      self.rk.keep_time()

  # -------------------------------------------------------------------------
  # Destination / fetch helpers
  # -------------------------------------------------------------------------

  def _same_destination(self, d: Coordinate | None) -> bool:
    if d is None and self.destination is None:
      return True
    if d is None or self.destination is None:
      return False
    return (abs(d.latitude - self.destination.latitude) < 1e-7
            and abs(d.longitude - self.destination.longitude) < 1e-7)

  def _try_fetch_initial(self) -> None:
    if self.destination is None or self.last_pos is None:
      return
    token = _read_token(self.params)
    if not token:
      cloudlog.warning("nkaoud_navd: cannot fetch route — no Mapbox token")
      return
    self.fetcher.submit(self.last_pos, self.destination, token, self.last_bearing)
    self.last_route_fetch_t = time.monotonic()

  def _drain_fetcher(self) -> None:
    if self.fetcher.in_flight():
      return
    result, error = self.fetcher.take_result()
    if error:
      cloudlog.error(f"nkaoud_navd: route fetch failed: {error}")
      self.rerouting = False
    elif result is not None:
      self.route = result
      self.step_idx = 0
      self.last_distance_along = 0.0
      self.bearing_misalign_counter = 0
      self.cross_track_counter = 0
      self.rerouting = False
      cloudlog.info(f"nkaoud_navd: new route loaded, "
                    f"{result.distance_total/1000:.1f} km, "
                    f"{len(result.steps)} steps")

  def _handle_share_trigger(self) -> None:
    trigger = (self.params.get("NkaoudNavShareTrigger") or "").strip()
    if self._last_share_trigger is None:
      self._last_share_trigger = trigger
      return

    new_trigger = trigger != self._last_share_trigger
    if new_trigger:
      self._last_share_trigger = trigger
      self._share_attempts = 0
      self._share_next_retry_t = 0.0
      if not trigger:
        return

    if self.share_fetcher.in_flight():
      return
    result, error = self.share_fetcher.take_result()

    if not trigger:
      return

    if result is not None:
      self.params.put("NkaoudNavDestination", result)
      self._share_attempts = SHARE_FETCH_MAX_ATTEMPTS
      return

    if error is not None:
      self._share_attempts += 1
      self._share_next_retry_t = time.monotonic() + SHARE_FETCH_RETRY_S
      cloudlog.error(f"nkaoud_navd: share fetch failed (attempt {self._share_attempts}): {error}")

    if (self._share_attempts < SHARE_FETCH_MAX_ATTEMPTS
        and time.monotonic() >= self._share_next_retry_t
        and not self.share_fetcher.in_flight()):
      self.share_fetcher.submit(trigger)

  # -------------------------------------------------------------------------
  # Progress tracking
  # -------------------------------------------------------------------------

  def _update_progress(self) -> None:
    if self.route is None or self.last_pos is None:
      return

    self.last_distance_along = distance_along_geometry(self.route.geometry, self.last_pos)

    # Arrival check
    dist_remaining = max(0.0, self.route.distance_total - self.last_distance_along)
    if dist_remaining <= ARRIVAL_DISTANCE_M:
      self.arrived = True
      return

    # Advance step_idx to the step that contains the current position
    cumulative = self.route.cumulative_step_distance
    new_idx = 0
    for i, c in enumerate(cumulative):
      if c <= self.last_distance_along:
        new_idx = i
    new_idx = min(new_idx, len(self.route.steps) - 1)
    self._prev_step_idx = self.step_idx
    self.step_idx = new_idx

  # -------------------------------------------------------------------------
  # Reroute detection
  # -------------------------------------------------------------------------

  def _check_reroute(self) -> None:
    if self.route is None or self.last_pos is None:
      return
    if self.fetcher.in_flight():
      return
    if self.arrived:
      return
    if time.monotonic() - self.last_route_fetch_t < MIN_REROUTE_INTERVAL_S:
      return

    # -- Bearing misalignment --
    route_brg = route_bearing_at(self.route.geometry, self.last_pos)
    if (route_brg is not None and self.last_bearing is not None
        and self.last_v_ego >= BEARING_MISALIGN_MIN_SPEED_MS):
      delta = _bearing_delta(self.last_bearing, route_brg)
      if delta > BEARING_MISALIGN_THRESHOLD_DEG:
        self.bearing_misalign_counter += 1
      else:
        self.bearing_misalign_counter = 0

      if self.bearing_misalign_counter >= BEARING_MISALIGN_COUNTER_MIN:
        cloudlog.info(f"nkaoud_navd: rerouting — bearing delta {delta:.1f}°")
        self._trigger_reroute()
        return

    # -- Cross-track distance (with hysteresis counter) --
    _, cross_track, _ = closest_segment_index(self.route.geometry, self.last_pos)
    self.cross_track_m = cross_track

    if cross_track > CROSS_TRACK_THRESHOLD_M:
      self.cross_track_counter = min(self.cross_track_counter + 1, CROSS_TRACK_COUNTER_ON + 5)
    else:
      # Decrement toward 0 when back on route (hysteresis)
      self.cross_track_counter = max(self.cross_track_counter - 1, 0)

    if self.cross_track_counter >= CROSS_TRACK_COUNTER_ON:
      cloudlog.info(f"nkaoud_navd: rerouting — cross-track {cross_track:.1f} m")
      self._trigger_reroute()

  def _trigger_reroute(self) -> None:
    if self.destination is None or self.last_pos is None:
      return
    token = _read_token(self.params)
    if not token:
      return
    self.rerouting = True
    self.bearing_misalign_counter = 0
    self.cross_track_counter = 0
    self.fetcher.submit(self.last_pos, self.destination, token, self.last_bearing)
    self.last_route_fetch_t = time.monotonic()

  # -------------------------------------------------------------------------
  # Highway suppression
  # -------------------------------------------------------------------------

  def _update_highway_suppression(self) -> None:
    """Suppress highway default when driver manually blinkers on a highway
    outside a navigation zone. The flag stays set until a TURN or POSITION
    desire fires (re-armed inside _lateral_desire via _arm_highway)."""
    if not (self.left_blinker or self.right_blinker):
      return
    cur_step = self._current_step()
    if _road_class(cur_step) != "highway":
      return
    dist = self._distance_to_maneuver()
    _, lane_keep_m = ZONE_THRESHOLDS["highway"]
    # Only suppress when the driver blinkers OUTSIDE the nav zone
    if dist > lane_keep_m or self.route is None:
      self._highway_suppressed = True

  def _arm_highway(self) -> None:
    """Re-arm highway default after a real nav command fires."""
    self._highway_suppressed = False

  def _update_lane_change_cooldown(self) -> None:
    """Start cooldown when the observed lane index changes."""
    current = self.lane_current
    prev = self._last_lane_current_observed
    self._last_lane_current_observed = current
    if self._lane_change_cooldown_s <= 0.0:
      return
    if prev <= 0 or current <= 0 or current == prev:
      return
    self._keep_cooldown_until = max(
      self._keep_cooldown_until, time.monotonic() + self._lane_change_cooldown_s,
    )

  def _keep_cooldown_active(self) -> bool:
    return time.monotonic() < self._keep_cooldown_until

  # -------------------------------------------------------------------------
  # Core lateral desire — single priority chain
  # -------------------------------------------------------------------------

  def _lateral_desire(self):
    """Returns the desire to send this tick. Priority: TURN > POSITION > HIGHWAY_DEFAULT > NONE."""
    if not self._steer_enabled or self.route is None or self.arrived:
      return NavDesire.none

    cur_step = self._current_step()
    next_step = self._upcoming_step()
    dist = self._distance_to_maneuver()

    # No upcoming maneuver — only highway default applies
    if next_step is None:
      return self._highway_default(cur_step)

    modifier = next_step.maneuver_modifier
    road_class = _road_class(cur_step)
    turn_cue_m, lane_keep_m = ZONE_THRESHOLDS[road_class]

    # ---- ZONE 1: TURN CUE ------------------------------------------------
    if dist <= turn_cue_m and modifier in (SHARP_TURN_LEFT | SHARP_TURN_RIGHT):
      side = "left" if modifier in SHARP_TURN_LEFT else "right"

      if self._driver_conflicting(side):
        # Driver is steering opposite — respect their intent
        return NavDesire.none

      if self._on_correct_side_strict(side):
        # Already in the outermost lane — command the turn
        desire = NavDesire.turnLeft if side == "left" else NavDesire.turnRight
        self._arm_highway()
        return desire

      # Not in the correct lane yet (or lane confidence too low to confirm).
      # Fall through to POSITION — keep nudging toward the correct side.

    # ---- ZONE 2: POSITION ------------------------------------------------
    if dist <= lane_keep_m:
      side = _modifier_to_side(modifier)
      if not side:
        return NavDesire.none
      if self._driver_conflicting(side):
        return NavDesire.none
      if self._side_gate_blocking(side):
        return NavDesire.none
      if self._keep_cooldown_active():
        return NavDesire.none
      if self._needs_to_move(side):
        desire = NavDesire.keepLeft if side == "left" else NavDesire.keepRight
        self._arm_highway()
        return desire
      return NavDesire.none  # already on the correct side

    # ---- ZONE 3: HIGHWAY DEFAULT -----------------------------------------
    return self._highway_default(cur_step)

  def _on_correct_side_strict(self, side: str) -> bool:
    """True when lane position confirms the car is in the outermost lane for a turn.
    Returns True (fire anyway) when confidence is unknown — callers with a
    distance fallback handle the half-distance override separately."""
    if self.lane_total <= 1:
      return True  # single-lane road: always "correct"
    if self.lane_current <= 0:
      return False  # no valid reading
    if self.lane_conf == "unknown":
      return False  # unknown: don't fire without evidence
    if side == "left":
      return self.lane_current == 1
    if side == "right":
      return self.lane_current == self.lane_total
    return False

  def _needs_to_move(self, side: str) -> bool:
    """True when lane position says we need to change toward `side`.
    Returns False when confidence is unknown — send nothing rather than guess."""
    if self.lane_conf == "unknown" or self.lane_total <= 1 or self.lane_current <= 0:
      return False
    if side == "left":
      return self.lane_current > 1
    if side == "right":
      return self.lane_current < self.lane_total
    return False

  def _driver_conflicting(self, nav_side: str) -> bool:
    """True when the driver is signalling the opposite direction."""
    one_blinker = self.left_blinker != self.right_blinker
    if not one_blinker:
      return False
    if nav_side == "left" and self.right_blinker:
      return True
    if nav_side == "right" and self.left_blinker:
      return True
    return False

  def _side_gate_blocking(self, nav_side: str) -> bool:
    """True when the side has not been clear for the full 4 s window.

    A side is only considered clear on a tick when BOTH signals agree:
      - blindspot on that side is false
      - VVD probability on that side is below VVD_CONF_THRESHOLD

    The counter increments only while both stay clear simultaneously and resets
    immediately on any violation. If VVD is missing, counters stay at 0 and the
    gate remains closed — fail-safe default."""
    if nav_side == "left":
      return self._left_clear_ticks < SIDE_CLEAR_TICKS_REQUIRED
    if nav_side == "right":
      return self._right_clear_ticks < SIDE_CLEAR_TICKS_REQUIRED
    return False

  def _highway_default(self, cur_step: Step | None):
    """When cruising on a motorway with no imminent maneuver, bias toward
    the center lane. Uses keepLeft/keepRight (conservative lane change).
    Suppressed by driver blinker; re-armed by the next nav command."""
    if not self._highway_default_enabled:
      return NavDesire.none
    if self._highway_suppressed:
      return NavDesire.none
    if cur_step is None or _road_class(cur_step) != "highway":
      return NavDesire.none
    if self.last_v_ego < HIGHWAY_DEFAULT_MIN_SPEED_MS:
      return NavDesire.none
    if self.lane_conf in ("unknown", "low") or self.lane_total <= 1 or self.lane_current <= 0:
      return NavDesire.none

    # Target lane based on user preference
    if self._highway_lane_pref == _LANE_PREF_RIGHTMOST:
      target = self.lane_total
    elif self._highway_lane_pref == _LANE_PREF_LEFTMOST:
      target = 1
    else:  # CENTER (default) — ceil(N/2), ties bias left
      target = math.ceil(self.lane_total / 2)

    if self.lane_current == target:
      return NavDesire.none

    side = "left" if self.lane_current > target else "right"
    if self._driver_conflicting(side):
      return NavDesire.none
    if self._side_gate_blocking(side):
      return NavDesire.none
    if self._keep_cooldown_active():
      return NavDesire.none

    return NavDesire.keepLeft if side == "left" else NavDesire.keepRight

  # -------------------------------------------------------------------------
  # Longitudinal — geometry-based turn speed cap
  # -------------------------------------------------------------------------

  def _longitudinal_cap(self) -> float:
    """Returns a speed cap in m/s, or 0.0 if no constraint applies.
    The longitudinal planner should treat 0.0 / negative as inactive."""
    if not self._speed_enabled or self.route is None or self.arrived:
      return 0.0

    next_step = self._upcoming_step()
    if next_step is None or not next_step.geometry:
      return 0.0

    curvature = _max_curvature(next_step.geometry)
    if curvature < MIN_CURVATURE:
      return 0.0   # essentially straight — no slowdown needed

    a_lat = self._max_lat_accel * self._turn_tolerance
    v_target = math.sqrt(a_lat / curvature)

    cur_step = self._current_step()
    road_class = _road_class(cur_step)
    turn_cue_m, _ = ZONE_THRESHOLDS[road_class]
    slow_start_m = turn_cue_m * SLOW_START_FACTOR

    dist = self._distance_to_maneuver()

    if dist > slow_start_m:
      return 0.0   # not in slowdown window yet

    if dist <= turn_cue_m:
      # Full constraint — clamp to v_target
      return float(v_target)

    # Linear ramp between slow_start_m (no constraint) and turn_cue_m (full)
    blend = (slow_start_m - dist) / (slow_start_m - turn_cue_m)
    v_cruise_approx = self.last_v_ego  # use current speed as reference
    v_cap = v_cruise_approx + (v_target - v_cruise_approx) * blend

    # Only constrain, never raise speed
    if v_cap >= self.last_v_ego:
      return 0.0

    return float(v_cap)

  # -------------------------------------------------------------------------
  # Step helpers
  # -------------------------------------------------------------------------

  def _current_step(self) -> Step | None:
    if self.route is None or not self.route.steps:
      return None
    return self.route.steps[min(self.step_idx, len(self.route.steps) - 1)]

  def _upcoming_step(self) -> Step | None:
    """The step whose maneuver we are approaching.
    Banners are on _current_step; modifier/type/geometry are on _upcoming_step."""
    if self.route is None:
      return None
    nxt = self.step_idx + 1
    if nxt >= len(self.route.steps):
      return None
    return self.route.steps[nxt]

  def _distance_to_maneuver(self) -> float:
    """Distance remaining (m) to the end of the current step (= the upcoming maneuver)."""
    if self.route is None or not self.route.cumulative_step_distance:
      return 0.0
    idx = min(self.step_idx, len(self.route.cumulative_step_distance) - 1)
    step_start = self.route.cumulative_step_distance[idx]
    step = self.route.steps[idx]
    step_end = step_start + step.distance
    return max(0.0, step_end - self.last_distance_along)

  # -------------------------------------------------------------------------
  # Publishing
  # -------------------------------------------------------------------------

  def _publish(self) -> None:
    active = self.route is not None and not self.arrived and self.destination is not None

    # Compute desires and longitudinal cap
    desire = self._lateral_desire() if active else NavDesire.none
    speed_cap = self._longitudinal_cap() if active else 0.0

    # -- nkaoudNavigationSP --
    nav_msg = messaging.new_message('nkaoudNavigationSP')
    nav = nav_msg.nkaoudNavigationSP
    nav.active = active
    nav.onRoute = active and not self.rerouting
    nav.rerouting = self.rerouting
    nav.arrived = self.arrived
    nav.navDesire = desire
    nav.maneuverTargetSpeed = float(speed_cap)
    if self.route is not None:
      nav.distanceRemaining = max(0.0, self.route.distance_total - self.last_distance_along)
      nav.distanceToManeuver = self._distance_to_maneuver()
    nav.laneCurrentIndex = self.lane_current
    nav.laneTotalCount = self.lane_total
    nav.laneConfidence = self.lane_conf
    nav.highwaySuppressed = self._highway_suppressed
    self.pm.send('nkaoudNavigationSP', nav_msg)

    # -- navRoute (polyline) — only re-publish on new route --
    if self.route is not None and self.sm.frame % 25 == 0:  # ~5 s
      route_msg = messaging.new_message('navRoute')
      route_msg.navRoute.coordinates = [
        {'latitude': c.latitude, 'longitude': c.longitude}
        for c in self.route.geometry
      ]
      self.pm.send('navRoute', route_msg)

    # -- navInstruction (current maneuver) --
    inst_msg = messaging.new_message('navInstruction')
    inst = inst_msg.navInstruction
    if active:
      cur_step = self._current_step()
      next_step = self._upcoming_step()
      dist = self._distance_to_maneuver()
      inst.maneuverDistance = dist
      if next_step is not None:
        inst.maneuverType = next_step.maneuver_type
        inst.maneuverModifier = next_step.maneuver_modifier
        inst.roadName = next_step.name
      if self.route is not None:
        dist_rem = max(0.0, self.route.distance_total - self.last_distance_along)
        inst.distanceRemaining = dist_rem
        inst.timeRemaining = dist_rem / max(self.last_v_ego, 1.0)
      # Banner selection: pick the most recently activated banner.
      # distanceAlongGeometry is distance remaining to maneuver, so any
      # banner where dist < b.distance_along_geometry is currently active.
      # We take the last such banner (smallest distanceAlongGeometry still > dist).
      if cur_step is not None and cur_step.banners:
        active_banner = cur_step.banners[0]
        for b in cur_step.banners:
          if dist < b.distance_along_geometry:
            active_banner = b
        inst.showFull = dist < active_banner.distance_along_geometry
        inst.primaryText = active_banner.primary_text
        inst.secondaryText = active_banner.secondary_text
    self.pm.send('navInstruction', inst_msg)


def main() -> None:
  NkaoudNavd().run()


if __name__ == "__main__":
  main()
