"""
Mapbox Directions client for nkaoud_nav.

Fetches a single driving-traffic route given (origin, destination, token, bearing)
and returns a RouteData dataclass that navd consumes.

Banner-instruction parsing is a fresh implementation (not reusing
sunnypilot.navd.helpers) so the experimental layer stays self-contained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import requests

from openpilot.sunnypilot.nkaoud_nav.geometry import Coordinate


MAPBOX_HOST = "https://api.mapbox.com"

DIRECTIONS = ("left", "right", "straight", "uturn")
MODIFIABLE_DIRECTIONS = ("left", "right")


@dataclass
class LaneOption:
  """One lane entry from Mapbox banner.sub.components (type == "lane").

  `active` is true if this lane leads to the upcoming maneuver. `directions`
  is the list of arrows shown on the lane sign (e.g. ["straight", "right"]).
  """
  active: bool = False
  directions: tuple[str, ...] = ()
  active_direction: str = ""


@dataclass
class Banner:
  primary_text: str = ""
  secondary_text: str = ""
  maneuver_type: str = ""
  maneuver_modifier: str = ""
  distance_along_geometry: float = 0.0
  lanes: tuple[LaneOption, ...] = ()


@dataclass
class Step:
  geometry: list[Coordinate]
  distance: float           # m
  duration: float           # s
  maneuver_type: str
  maneuver_modifier: str
  banners: list[Banner]
  name: str = ""            # street name
  # Road classes from intersections[0].classes ("motorway", "primary", "tunnel"...).
  # Empty tuple if the step has no intersections (rare).
  road_classes: tuple[str, ...] = ()
  # Mapbox maneuver bearings (compass degrees, clockwise from north): the
  # heading just before and after the maneuver point. None when the step has
  # no maneuver bearings (e.g. depart/arrive). The signed difference is the
  # turn angle. Present on turn maneuvers, which is all the turn assist needs.
  maneuver_bearing_before: float | None = None
  maneuver_bearing_after: float | None = None


@dataclass
class RouteData:
  route_id: str
  geometry: list[Coordinate]                  # full route polyline
  steps: list[Step]
  distance_total: float
  duration_total: float
  cumulative_step_distance: list[float] = field(default_factory=list)


class RouteFetchError(RuntimeError):
  pass


def _direction_token(direction: str) -> str:
  for d in DIRECTIONS:
    if d in direction:
      if "slight" in direction and d in MODIFIABLE_DIRECTIONS:
        return "slight" + d.capitalize()
      return d
  return "none"


def _parse_banners(raw_banners: list[dict[str, Any]] | None) -> list[Banner]:
  out: list[Banner] = []
  if not raw_banners:
    return out
  for b in raw_banners:
    banner = Banner()
    banner.distance_along_geometry = float(b.get("distanceAlongGeometry") or 0.0)
    primary = b.get("primary") or {}
    banner.primary_text = primary.get("text") or ""
    banner.maneuver_type = primary.get("type") or ""
    banner.maneuver_modifier = primary.get("modifier") or ""
    secondary = b.get("secondary") or {}
    banner.secondary_text = secondary.get("text") or ""

    # Lane guidance lives under sub.components -- each entry of type "lane".
    sub = b.get("sub") or {}
    lanes: list[LaneOption] = []
    for comp in (sub.get("components") or []):
      if comp.get("type") != "lane":
        continue
      lanes.append(LaneOption(
        active=bool(comp.get("active") or False),
        directions=tuple(comp.get("directions") or ()),
        active_direction=comp.get("active_direction") or "",
      ))
    banner.lanes = tuple(lanes)
    out.append(banner)
  return out


def _parse_step(raw: dict[str, Any]) -> Step:
  raw_coords = (raw.get("geometry") or {}).get("coordinates") or []
  geom = [Coordinate.from_mapbox_tuple(tuple(c)) for c in raw_coords]
  maneuver = raw.get("maneuver") or {}
  intersections = raw.get("intersections") or []
  road_classes: tuple[str, ...] = ()
  if intersections:
    raw_classes = intersections[0].get("classes") or []
    road_classes = tuple(str(c) for c in raw_classes)
  bearing_before = maneuver.get("bearing_before")
  bearing_after = maneuver.get("bearing_after")
  return Step(
    geometry=geom,
    distance=float(raw.get("distance") or 0.0),
    duration=float(raw.get("duration") or 0.0),
    maneuver_type=maneuver.get("type") or "",
    maneuver_modifier=maneuver.get("modifier") or "",
    banners=_parse_banners(raw.get("bannerInstructions")),
    name=raw.get("name") or "",
    road_classes=road_classes,
    maneuver_bearing_before=float(bearing_before) if bearing_before is not None else None,
    maneuver_bearing_after=float(bearing_after) if bearing_after is not None else None,
  )


def fetch_route(origin: Coordinate, destination: Coordinate, token: str,
                bearing_deg: float | None = None, timeout: float = 10.0) -> RouteData:
  if not token:
    raise RouteFetchError("missing mapbox token")

  coords_str = f"{origin.longitude},{origin.latitude};{destination.longitude},{destination.latitude}"
  url = f"{MAPBOX_HOST}/directions/v5/mapbox/driving-traffic/{coords_str}"
  params: dict[str, str] = {
    "access_token": token,
    "annotations": "maxspeed",
    "geometries": "geojson",
    "overview": "full",
    "steps": "true",
    "banner_instructions": "true",
    "alternatives": "false",
    "language": "en",
  }
  if bearing_deg is not None:
    # 90deg uncertainty cone on origin, nothing on destination
    params["bearings"] = f"{int((bearing_deg + 360) % 360)},90;"

  try:
    resp = requests.get(url, params=params, timeout=timeout)
  except requests.RequestException as e:
    raise RouteFetchError(f"network error: {e}") from e
  if resp.status_code != 200:
    raise RouteFetchError(f"mapbox http {resp.status_code}: {resp.text[:200]}")

  body = resp.json()
  routes = body.get("routes") or []
  if not routes:
    raise RouteFetchError("mapbox returned no routes")

  route = routes[0]
  legs = route.get("legs") or []
  raw_steps = [s for leg in legs for s in (leg.get("steps") or [])]
  steps = [_parse_step(s) for s in raw_steps]

  raw_geom = (route.get("geometry") or {}).get("coordinates") or []
  geometry = [Coordinate.from_mapbox_tuple(tuple(c)) for c in raw_geom]

  cumulative: list[float] = []
  acc = 0.0
  for s in steps:
    cumulative.append(acc)
    acc += s.distance

  return RouteData(
    route_id=body.get("uuid") or route.get("weight_name") or "route",
    geometry=geometry,
    steps=steps,
    distance_total=float(route.get("distance") or 0.0),
    duration_total=float(route.get("duration") or 0.0),
    cumulative_step_distance=cumulative,
  )
