"""
StarPilot-style Mapbox navigation provider for nkaoud_nav.

This is a target-native adaptation of StarPilot's navigation route engine and
navigation turn-speed calculation. It does not publish messages, own a
destination parameter, or issue any vehicle commands. The existing nkaoud_navd
remains the only publisher; the source-equivalent lateral policy runs in
DesireHelper at model cadence, as it does in StarPilot.

The implementation is based on the MIT-licensed StarPilot route engine and
navigation policy at commit 101c354f3e3e64ae273056a216133b74d7343727.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import requests

from openpilot.common.constants import CV
from openpilot.sunnypilot.nkaoud_nav.geometry import Coordinate, EARTH_MEAN_RADIUS


ROUTE_FETCH_RETRY_S = 5.0
REROUTE_TRIGGER_SECONDS = 2.0
ARRIVAL_CLEAR_SECONDS = 5.0

OFF_ROUTE_SPEED_BREAKPOINTS = (0.0, 5.0, 10.0, 20.0, 40.0)
OFF_ROUTE_DISTANCE_BREAKPOINTS = (40.0, 50.0, 60.0, 80.0, 100.0)

NAV_TURN_COMFORT_DECEL = 1.25
NAV_TURN_DISTANCE_BUFFER = 8.0
NAV_TURN_MIN_TARGET_DELTA = 0.25

LANE_DIRECTIONS = frozenset((
  "none",
  "left",
  "right",
  "straight",
  "slightLeft",
  "slightRight",
))
LEFT_DIRECTIONS = frozenset(("slightLeft", "left", "sharpLeft"))
RIGHT_DIRECTIONS = frozenset(("slightRight", "right", "sharpRight"))


def _interpolate(value: float, breakpoints: tuple[float, ...], values: tuple[float, ...]) -> float:
  if value <= breakpoints[0]:
    return values[0]
  if value >= breakpoints[-1]:
    return values[-1]
  for i in range(1, len(breakpoints)):
    if value <= breakpoints[i]:
      ratio = (value - breakpoints[i - 1]) / (breakpoints[i] - breakpoints[i - 1])
      return values[i - 1] + ratio * (values[i] - values[i - 1])
  return values[-1]


def _field_valid(data: dict[str, Any], field: str) -> bool:
  return field in data and data[field] is not None


def string_to_direction(direction: object) -> str:
  normalized = str(direction or "").lower()
  for direction_name in ("left", "right", "straight"):
    if direction_name not in normalized:
      continue
    if "slight" in normalized and direction_name in ("left", "right"):
      return f"slight{direction_name.capitalize()}"
    if "sharp" in normalized and direction_name in ("left", "right"):
      return f"sharp{direction_name.capitalize()}"
    return direction_name
  if "uturn" in normalized or "u-turn" in normalized:
    return "uturn"
  return "none"


def normalize_lane_direction(direction: object) -> str:
  normalized = string_to_direction(direction)
  return normalized if normalized in LANE_DIRECTIONS else "none"


def _maxspeed_to_ms(maxspeed: dict[str, Any]) -> float:
  try:
    speed = float(maxspeed["speed"])
    unit = str(maxspeed["unit"])
  except (KeyError, TypeError, ValueError):
    return 0.0
  if unit == "km/h":
    return speed * CV.KPH_TO_MS
  if unit == "mph":
    return speed * CV.MPH_TO_MS
  return 0.0


def _project_onto_segment(a: Coordinate, b: Coordinate, point: Coordinate) -> tuple[float, float]:
  lat_scale = EARTH_MEAN_RADIUS * math.pi / 180.0
  reference_latitude = math.radians((a.latitude + b.latitude + point.latitude) / 3.0)
  lon_scale = lat_scale * math.cos(reference_latitude)

  ab_x = (b.longitude - a.longitude) * lon_scale
  ab_y = (b.latitude - a.latitude) * lat_scale
  ap_x = (point.longitude - a.longitude) * lon_scale
  ap_y = (point.latitude - a.latitude) * lat_scale
  segment_length_sq = ab_x * ab_x + ab_y * ab_y
  if segment_length_sq <= 1e-6:
    return 0.0, math.hypot(ap_x, ap_y)

  t = max(0.0, min(1.0, (ap_x * ab_x + ap_y * ab_y) / segment_length_sq))
  return t, math.hypot(ap_x - ab_x * t, ap_y - ab_y * t)


def _bearing_between(a: Coordinate, b: Coordinate) -> float:
  latitude_a = math.radians(a.latitude)
  latitude_b = math.radians(b.latitude)
  longitude_delta = math.radians(b.longitude - a.longitude)
  bearing = math.atan2(
    math.sin(longitude_delta) * math.cos(latitude_b),
    math.cos(latitude_a) * math.sin(latitude_b) -
    math.sin(latitude_a) * math.cos(latitude_b) * math.cos(longitude_delta),
  )
  return (math.degrees(bearing) + 360.0) % 360.0


@dataclass(frozen=True)
class StarPilotRouteStep:
  banner_instructions: list[dict[str, Any]]
  distance: float
  duration: float
  maneuver: str
  location: Coordinate
  cumulative_distance: float
  maxspeed_ms: float
  modifier: str
  instruction: str


@dataclass(frozen=True)
class StarPilotRouteProgress:
  closest_index: int
  closest_segment_index: int
  distance_from_route: float
  current_step: StarPilotRouteStep
  next_step: StarPilotRouteStep | None
  current_step_index: int
  distance_to_end_of_step: float
  distance_remaining: float
  time_remaining: float
  current_speed_limit_ms: float
  all_maneuvers: list[dict[str, Any]]


@dataclass(frozen=True)
class StarPilotRoute:
  route_id: str
  geometry: list[Coordinate]
  geometry_cumulative_distances: list[float]
  bearings: list[float]
  steps: list[StarPilotRouteStep]
  total_distance: float
  total_duration: float

  @classmethod
  def from_mapbox_response(cls, response: dict[str, Any]) -> StarPilotRoute | None:
    routes = response.get("routes") or []
    route_data = routes[0] if routes else None
    legs = route_data.get("legs") if isinstance(route_data, dict) else None
    leg = legs[0] if legs else None
    if response.get("code") != "Ok" or not isinstance(route_data, dict) or not isinstance(leg, dict):
      return None

    raw_geometry = (route_data.get("geometry") or {}).get("coordinates") or []
    raw_steps = leg.get("steps") or []
    if not raw_geometry or not raw_steps:
      return None

    geometry = [Coordinate(float(point[1]), float(point[0])) for point in raw_geometry]
    cumulative_distances = [0.0]
    for index in range(1, len(geometry)):
      cumulative_distances.append(cumulative_distances[-1] + geometry[index - 1].distance_to(geometry[index]))

    maxspeeds = [
      _maxspeed_to_ms(item)
      for item in (leg.get("annotation") or {}).get("maxspeed", [])
      if isinstance(item, dict) and _field_valid(item, "speed") and _field_valid(item, "unit")
    ]

    steps: list[StarPilotRouteStep] = []
    for raw_step in raw_steps:
      maneuver = raw_step.get("maneuver") or {}
      location_raw = maneuver.get("location") or ()
      if len(location_raw) < 2:
        return None
      location = Coordinate(float(location_raw[1]), float(location_raw[0]))
      closest_index = min(range(len(geometry)), key=lambda index: location.distance_to(geometry[index]))
      maxspeed_ms = maxspeeds[min(closest_index, len(maxspeeds) - 1)] if maxspeeds else 0.0
      steps.append(StarPilotRouteStep(
        banner_instructions=list(raw_step.get("bannerInstructions") or []),
        distance=float(raw_step.get("distance") or 0.0),
        duration=float(raw_step.get("duration") or 0.0),
        maneuver=str(maneuver.get("type") or ""),
        location=location,
        cumulative_distance=cumulative_distances[closest_index],
        maxspeed_ms=maxspeed_ms,
        modifier=string_to_direction(maneuver.get("modifier")),
        instruction=str(maneuver.get("instruction") or ""),
      ))

    bearings = [_bearing_between(geometry[index], geometry[index + 1]) for index in range(len(geometry) - 1)]
    return cls(
      route_id=str(response.get("uuid") or route_data.get("uuid") or route_data.get("weight_name") or "starpilot-route"),
      geometry=geometry,
      geometry_cumulative_distances=cumulative_distances,
      bearings=bearings,
      steps=steps,
      total_distance=float(route_data.get("distance") or 0.0),
      total_duration=float(route_data.get("duration") or 0.0),
    )

  def route_bearing_misaligned(self, closest_segment_index: int, current_bearing: float | None, v_ego: float) -> bool:
    if current_bearing is None or v_ego < 2.5 or closest_segment_index < 0 or closest_segment_index >= len(self.bearings):
      return False
    route_bearing = self.bearings[closest_segment_index]
    current_bearing = (current_bearing + 360.0) % 360.0
    difference = abs(current_bearing - route_bearing)
    return min(difference, 360.0 - difference) > 75.0

  def get_progress(self, position: Coordinate) -> StarPilotRouteProgress | None:
    if not self.geometry or not self.steps:
      return None

    if len(self.geometry) == 1:
      closest_index = 0
      closest_segment_index = 0
      distance_from_route = position.distance_to(self.geometry[0])
      closest_cumulative = 0.0
    else:
      best_segment_index = 0
      best_distance = float("inf")
      best_t = 0.0
      for index in range(len(self.geometry) - 1):
        t, distance = _project_onto_segment(self.geometry[index], self.geometry[index + 1], position)
        if distance < best_distance:
          best_distance = distance
          best_segment_index = index
          best_t = t

      closest_segment_index = best_segment_index
      start_distance = self.geometry_cumulative_distances[closest_segment_index]
      end_distance = self.geometry_cumulative_distances[closest_segment_index + 1]
      closest_cumulative = start_distance + (end_distance - start_distance) * best_t
      distance_from_route = best_distance
      closest_index = min(closest_segment_index + int(best_t >= 0.5), len(self.geometry) - 1)

    current_step_index = max(
      (index for index, step in enumerate(self.steps) if step.cumulative_distance <= closest_cumulative + 1e-3),
      default=-1,
    )
    current_index = max(current_step_index, 0)
    current_step = self.steps[current_index]
    next_index = current_step_index + 1
    next_step = self.steps[next_index] if 0 <= next_index < len(self.steps) else None

    distance_to_end_of_step = max(0.0, current_step.distance - (closest_cumulative - current_step.cumulative_distance))
    distance_remaining = max(0.0, self.total_distance - closest_cumulative)
    remaining_current_duration = current_step.duration * min(distance_to_end_of_step / max(current_step.distance, 1.0), 1.0)
    later_duration = sum(step.duration for step in self.steps[next_index:])

    all_maneuvers: list[dict[str, Any]] = []
    start_index = max(current_step_index, 0)
    for index in range(start_index, min(start_index + 3, len(self.steps))):
      step = self.steps[index]
      distance = distance_to_end_of_step if index == start_index else max(0.0, step.cumulative_distance - closest_cumulative)
      all_maneuvers.append({"distance": distance, "type": step.maneuver, "modifier": step.modifier})

    return StarPilotRouteProgress(
      closest_index=closest_index,
      closest_segment_index=closest_segment_index,
      distance_from_route=distance_from_route,
      current_step=current_step,
      next_step=next_step,
      current_step_index=current_index,
      distance_to_end_of_step=distance_to_end_of_step,
      distance_remaining=distance_remaining,
      time_remaining=max(0.0, remaining_current_duration + later_duration),
      current_speed_limit_ms=current_step.maxspeed_ms,
      all_maneuvers=all_maneuvers,
    )

  def off_route_distance_exceeded(self, progress: StarPilotRouteProgress, v_ego: float) -> bool:
    threshold = _interpolate(v_ego, OFF_ROUTE_SPEED_BREAKPOINTS, OFF_ROUTE_DISTANCE_BREAKPOINTS)
    return progress.distance_from_route > threshold

  def arrived(self, progress: StarPilotRouteProgress, v_ego: float) -> bool:
    if v_ego >= 2.0 or not progress.all_maneuvers:
      return False
    current = progress.all_maneuvers[0]
    destination_step = current["type"] == "arrive" or progress.current_step.maneuver == "arrive"
    destination_step |= progress.current_step.instruction.startswith("Your destination")
    if not destination_step and progress.next_step is not None:
      destination_step = progress.next_step.maneuver == "arrive"
      destination_step &= progress.distance_to_end_of_step <= max(15.0, v_ego * 8.0)
    return destination_step and progress.distance_remaining <= 40.0

  def build_instruction_payload(self, progress: StarPilotRouteProgress) -> dict[str, Any]:
    parsed = parse_banner_instructions(progress.current_step.banner_instructions, progress.distance_to_end_of_step) or {}
    lanes = []
    for lane in parsed.get("lanes") or []:
      directions = [direction for direction in lane.get("directions") or [] if direction in LANE_DIRECTIONS]
      active_direction = lane.get("activeDirection", "none")
      lanes.append({
        "directions": directions or ["none"],
        "active": bool(lane.get("active", False)),
        "activeDirection": active_direction if active_direction in LANE_DIRECTIONS else "none",
      })

    return {
      "maneuverPrimaryText": str(parsed.get("maneuverPrimaryText") or progress.current_step.instruction),
      "maneuverSecondaryText": str(parsed.get("maneuverSecondaryText") or ""),
      "maneuverDistance": progress.distance_to_end_of_step,
      "maneuverType": str(parsed.get("maneuverType") or progress.current_step.maneuver),
      "maneuverModifier": str(parsed.get("maneuverModifier") or progress.current_step.modifier),
      "distanceRemaining": progress.distance_remaining,
      "timeRemaining": progress.time_remaining,
      "timeRemainingTypical": progress.time_remaining,
      "lanes": lanes,
      "showFull": bool(parsed.get("showFull", True)),
      "speedLimit": progress.current_speed_limit_ms,
      "allManeuvers": progress.all_maneuvers,
    }


def parse_banner_instructions(banners: list[dict[str, Any]], distance_to_maneuver: float) -> dict[str, Any] | None:
  if not banners:
    return None
  current_banner = banners[0]
  for banner in banners:
    if distance_to_maneuver < float(banner.get("distanceAlongGeometry") or 0.0):
      current_banner = banner

  instruction: dict[str, Any] = {
    "showFull": distance_to_maneuver < float(current_banner.get("distanceAlongGeometry") or 0.0),
  }
  primary = current_banner.get("primary") or {}
  if _field_valid(primary, "text"):
    instruction["maneuverPrimaryText"] = primary["text"]
  if _field_valid(primary, "type"):
    instruction["maneuverType"] = primary["type"]
  if _field_valid(primary, "modifier"):
    instruction["maneuverModifier"] = string_to_direction(primary["modifier"])

  secondary = current_banner.get("secondary") or {}
  if _field_valid(secondary, "text"):
    instruction["maneuverSecondaryText"] = secondary["text"]

  lanes = []
  for component in ((current_banner.get("sub") or {}).get("components") or []):
    if component.get("type") != "lane":
      continue
    lane = {
      "active": bool(component.get("active", False)),
      "directions": [normalize_lane_direction(direction) for direction in component.get("directions") or []],
    }
    if _field_valid(component, "active_direction"):
      lane["activeDirection"] = normalize_lane_direction(component["active_direction"])
    lanes.append(lane)
  instruction["lanes"] = lanes
  return instruction


def _nav_instruction_state(payload: dict[str, Any]) -> dict[str, Any]:
  lanes = payload.get("lanes") or []
  active_direction = ""
  active_index = -1
  for index, lane in enumerate(lanes):
    if not isinstance(lane, dict) or not lane.get("active", False):
      continue
    candidate = str(lane.get("activeDirection") or "")
    directions = lane.get("directions") or []
    if (not candidate or candidate == "none") and len(directions) == 1:
      candidate = str(directions[0] or "")
    if candidate and candidate != "none":
      active_direction = candidate
      active_index = index
      break

  active_side = "left" if active_direction in LEFT_DIRECTIONS else "right" if active_direction in RIGHT_DIRECTIONS else ""
  same_side_count = 0
  active_at_road_edge = False
  has_shared_same_side_lane = False
  if active_side:
    same_side_directions = LEFT_DIRECTIONS if active_side == "left" else RIGHT_DIRECTIONS
    active_at_road_edge = active_index == 0 if active_side == "left" else active_index == len(lanes) - 1
    for lane in lanes:
      if not isinstance(lane, dict):
        continue
      directions = {str(direction) for direction in lane.get("directions") or [] if direction}
      if directions & same_side_directions:
        same_side_count += 1
        has_shared_same_side_lane |= bool(directions - same_side_directions)

  all_maneuvers = payload.get("allManeuvers") or []
  next_maneuver = all_maneuvers[1] if len(all_maneuvers) > 1 and isinstance(all_maneuvers[1], dict) else {}
  return {
    "valid": bool(payload),
    "maneuverModifier": str(payload.get("maneuverModifier") or ""),
    "maneuverType": str(payload.get("maneuverType") or ""),
    "maneuverDistance": float(payload.get("maneuverDistance") or 0.0),
    "nextManeuverType": str(next_maneuver.get("type") or ""),
    "nextManeuverModifier": str(next_maneuver.get("modifier") or ""),
    "nextManeuverDistance": float(next_maneuver.get("distance") or 0.0),
    "laneCount": len(lanes),
    "activeLaneDirection": active_direction,
    "activeLaneIndex": active_index,
    "activeLaneAtRoadEdge": active_at_road_edge,
    "hasSharedSameSideLane": has_shared_same_side_lane,
    "sameSideLaneCount": same_side_count,
  }


def _maneuver_target_speed(maneuver_type: object, maneuver_modifier: object) -> float | None:
  maneuver_type = str(maneuver_type or "").strip().lower()
  modifier = string_to_direction(maneuver_modifier)
  if not maneuver_type and modifier == "none":
    return None
  if modifier == "uturn" or "uturn" in maneuver_type or "u-turn" in maneuver_type:
    return 5.0 * CV.MPH_TO_MS
  if "roundabout" in maneuver_type or "rotary" in maneuver_type:
    return 12.0 * CV.MPH_TO_MS
  if maneuver_type == "turn":
    return {
      "sharpLeft": 10.0 * CV.MPH_TO_MS,
      "sharpRight": 10.0 * CV.MPH_TO_MS,
      "left": 14.0 * CV.MPH_TO_MS,
      "right": 14.0 * CV.MPH_TO_MS,
    }.get(modifier)
  return None


def _target_for_distance(target_speed: float, maneuver_distance: object) -> float:
  try:
    remaining_distance = max(float(maneuver_distance) - NAV_TURN_DISTANCE_BUFFER, 0.0)
  except (TypeError, ValueError):
    return 0.0
  return math.sqrt(max(target_speed * target_speed + 2.0 * NAV_TURN_COMFORT_DECEL * remaining_distance, 0.0))


def starpilot_turn_speed_target(payload: dict[str, Any], v_cruise: float, min_steer_speed: float = 0.0) -> float:
  """Calculate StarPilot's route-turn cap; zero means no cap.

  This function only yields a lower cruise target.  The target longitudinal
  planner still arbitrates it with SCC and speed-limit sources.
  """
  state = _nav_instruction_state(payload)
  candidates = (
    (state["maneuverType"], state["maneuverModifier"], state["maneuverDistance"]),
    (state["nextManeuverType"], state["nextManeuverModifier"], state["nextManeuverDistance"]),
  )
  for maneuver_type, maneuver_modifier, maneuver_distance in candidates:
    target_speed = _maneuver_target_speed(maneuver_type, maneuver_modifier)
    if target_speed is None:
      continue
    target_speed = max(target_speed, max(float(min_steer_speed), 0.0))
    target = max(target_speed, _target_for_distance(target_speed, maneuver_distance))
    if target + NAV_TURN_MIN_TARGET_DELTA < v_cruise:
      return target
  return 0.0


class StarPilotRouteFetchError(RuntimeError):
  pass


@dataclass(frozen=True)
class StarPilotRouteRequest:
  key: tuple[int, float, float]
  origin: Coordinate
  destination: Coordinate
  token: str
  bearing: float | None


class StarPilotRouteEngine:
  DIRECTIONS_URL = "https://api.mapbox.com/directions/v5/mapbox/driving-traffic"

  def __init__(self, session: Any = requests) -> None:
    self._session = session

  def fetch_route(self, request: StarPilotRouteRequest) -> StarPilotRoute:
    coordinates = f"{request.origin.longitude},{request.origin.latitude};{request.destination.longitude},{request.destination.latitude}"
    params = {
      "access_token": request.token,
      "geometries": "geojson",
      "steps": "true",
      "overview": "full",
      "annotations": "maxspeed",
      "alternatives": "false",
      "banner_instructions": "true",
    }
    if request.bearing is not None:
      params["bearings"] = f"{int((request.bearing + 360.0) % 360.0)},90;"
    try:
      response = self._session.get(f"{self.DIRECTIONS_URL}/{coordinates}", params=params, timeout=5)
    except requests.RequestException as error:
      raise StarPilotRouteFetchError(f"network error: {error}") from error
    if response.status_code != 200:
      raise StarPilotRouteFetchError(f"Mapbox HTTP {response.status_code}")
    try:
      response_data = response.json()
      if not isinstance(response_data, dict):
        raise TypeError("response was not an object")
      route = StarPilotRoute.from_mapbox_response(response_data)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
      raise StarPilotRouteFetchError(f"invalid Mapbox response: {error}") from error
    if route is None:
      raise StarPilotRouteFetchError("Mapbox returned no usable route")
    return route


def fetch_starpilot_route(request: StarPilotRouteRequest) -> tuple[tuple[int, float, float], StarPilotRoute]:
  return request.key, StarPilotRouteEngine().fetch_route(request)


@dataclass(frozen=True)
class StarPilotNavigationState:
  valid: bool
  active: bool
  on_route: bool
  rerouting: bool
  arrived: bool
  route: StarPilotRoute | None
  progress: StarPilotRouteProgress | None
  instruction: dict[str, Any]
  instruction_state: dict[str, Any]
  maneuver_target_speed: float


class StarPilotNavigationProvider:
  """State machine for the selectable StarPilot route provider.

  It deliberately never modifies the target destination. The caller owns the
  target's normal arrival lifecycle.
  """
  def __init__(self) -> None:
    self.destination: Coordinate | None = None
    self._destination_key: tuple[float, float] | None = None
    self._completed_destination_key: tuple[float, float] | None = None
    self._generation = 0
    self.route_generation = 0
    self.route: StarPilotRoute | None = None
    self.reroute_requested = False
    self.arrived = False
    self.last_fetch_at = -ROUTE_FETCH_RETRY_S
    self.last_error = ""
    self._off_route_started_at: float | None = None
    self._bearing_misaligned_started_at: float | None = None
    self._arrival_started_at: float | None = None

  @staticmethod
  def _key(destination: Coordinate | None) -> tuple[float, float] | None:
    if destination is None:
      return None
    return round(destination.latitude, 7), round(destination.longitude, 7)

  def _reset_route_state(self) -> None:
    self.route = None
    self.reroute_requested = False
    self.arrived = False
    self._off_route_started_at = None
    self._bearing_misaligned_started_at = None
    self._arrival_started_at = None

  def set_destination(self, destination: Coordinate | None) -> bool:
    key = self._key(destination)
    if key == self._destination_key:
      return False
    self.destination = destination
    self._destination_key = key
    self._completed_destination_key = None
    self._generation += 1
    self.last_fetch_at = -ROUTE_FETCH_RETRY_S
    self._reset_route_state()
    return True

  def clear(self) -> None:
    self.destination = None
    self._destination_key = None
    self._completed_destination_key = None
    self._generation += 1
    self.last_fetch_at = -ROUTE_FETCH_RETRY_S
    self._reset_route_state()

  def _request_key(self) -> tuple[int, float, float] | None:
    if self._destination_key is None:
      return None
    return self._generation, *self._destination_key

  def next_fetch_request(self, position: Coordinate | None, bearing: float | None, token: str, now: float,
                         fetch_in_flight: bool) -> StarPilotRouteRequest | None:
    if fetch_in_flight or position is None or self.destination is None or not token:
      return None
    if self._destination_key == self._completed_destination_key:
      return None
    if self.route is not None and not self.reroute_requested:
      return None
    if now - self.last_fetch_at < ROUTE_FETCH_RETRY_S:
      return None
    key = self._request_key()
    if key is None:
      return None
    self.last_fetch_at = now
    return StarPilotRouteRequest(key, position, self.destination, token, bearing)

  def accept_fetch(self, request_key: tuple[int, float, float], route: StarPilotRoute) -> bool:
    if request_key != self._request_key():
      return False
    self.route = route
    self.route_generation += 1
    self.reroute_requested = False
    self.arrived = False
    self.last_error = ""
    self._off_route_started_at = None
    self._bearing_misaligned_started_at = None
    self._arrival_started_at = None
    return True

  def reject_fetch(self, request_key: tuple[int, float, float] | None, error: str) -> None:
    if request_key == self._request_key():
      self.last_error = error

  @staticmethod
  def _bump_timer(started_at: float | None, condition: bool, now: float) -> float | None:
    if not condition:
      return None
    return now if started_at is None else started_at

  def update(self, position: Coordinate | None, bearing: float | None, v_ego: float, v_cruise: float, now: float,
             min_steer_speed: float) -> StarPilotNavigationState:
    location_valid = position is not None
    if self.route is None or not location_valid:
      if not location_valid:
        self._off_route_started_at = None
        self._bearing_misaligned_started_at = None
        self._arrival_started_at = None
      return StarPilotNavigationState(
        valid=False,
        active=False,
        on_route=False,
        rerouting=self.reroute_requested,
        arrived=self.arrived,
        route=self.route,
        progress=None,
        instruction={},
        instruction_state={},
        maneuver_target_speed=0.0,
      )

    progress = self.route.get_progress(position)
    if progress is None:
      return StarPilotNavigationState(
        valid=False,
        active=False,
        on_route=False,
        rerouting=self.reroute_requested,
        arrived=self.arrived,
        route=self.route,
        progress=None,
        instruction={},
        instruction_state={},
        maneuver_target_speed=0.0,
      )

    off_route = self.route.off_route_distance_exceeded(progress, v_ego)
    misaligned = self.route.route_bearing_misaligned(progress.closest_segment_index, bearing, v_ego)
    arrived = self.route.arrived(progress, v_ego)
    self._off_route_started_at = self._bump_timer(self._off_route_started_at, off_route and not arrived, now)
    self._bearing_misaligned_started_at = self._bump_timer(self._bearing_misaligned_started_at, misaligned and not arrived, now)
    self._arrival_started_at = self._bump_timer(self._arrival_started_at, arrived, now)

    # StarPilot skips rerouting once the final route step is active. This
    # avoids a fetch loop near a destination or after intentionally stopping.
    can_reroute = progress.current_step_index < len(self.route.steps) - 1
    if can_reroute:
      if self._off_route_started_at is not None and now - self._off_route_started_at >= REROUTE_TRIGGER_SECONDS:
        self.reroute_requested = True
      if self._bearing_misaligned_started_at is not None and now - self._bearing_misaligned_started_at >= REROUTE_TRIGGER_SECONDS:
        self.reroute_requested = True
    else:
      self._off_route_started_at = None
      self._bearing_misaligned_started_at = None
    if self._arrival_started_at is not None and now - self._arrival_started_at >= ARRIVAL_CLEAR_SECONDS:
      self._completed_destination_key = self._destination_key
      self.route = None
      self.reroute_requested = False
      self.arrived = True
      return StarPilotNavigationState(
        valid=False,
        active=False,
        on_route=False,
        rerouting=False,
        arrived=True,
        route=None,
        progress=None,
        instruction={},
        instruction_state={},
        maneuver_target_speed=0.0,
      )

    # The old route may point down an entirely different road. While a
    # StarPilot reroute request is pending, keep the geometry visible but make
    # all control-facing output invalid and neutral.
    if self.reroute_requested:
      return StarPilotNavigationState(
        valid=False,
        active=False,
        on_route=False,
        rerouting=True,
        arrived=False,
        route=self.route,
        progress=progress,
        instruction={},
        instruction_state={},
        maneuver_target_speed=0.0,
      )

    instruction = self.route.build_instruction_payload(progress)
    return StarPilotNavigationState(
      valid=True,
      active=True,
      on_route=not self.reroute_requested,
      rerouting=self.reroute_requested,
      arrived=False,
      route=self.route,
      progress=progress,
      instruction=instruction,
      instruction_state=_nav_instruction_state(instruction),
      maneuver_target_speed=starpilot_turn_speed_target(instruction, v_cruise, min_steer_speed),
    )
