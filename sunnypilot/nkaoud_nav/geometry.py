"""
Coordinate math used by nkaoud_navd.

Standalone (no dependency on sunnypilot/navd/helpers.py per the
'no existing sunnypilot navigation code' rule for this fork).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


EARTH_MEAN_RADIUS = 6371007.2


@dataclass
class Coordinate:
  latitude: float
  longitude: float
  annotations: dict[str, float] = field(default_factory=dict)

  @classmethod
  def from_mapbox_tuple(cls, t: tuple[float, float]) -> "Coordinate":
    # Mapbox geojson is [lon, lat]
    return cls(t[1], t[0])

  def distance_to(self, other: "Coordinate") -> float:
    # Haversine
    dlat = math.radians(other.latitude - self.latitude)
    dlon = math.radians(other.longitude - self.longitude)
    h = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(self.latitude))
         * math.cos(math.radians(other.latitude))
         * math.sin(dlon / 2) ** 2)
    return 2 * EARTH_MEAN_RADIUS * math.asin(math.sqrt(h))


def bearing_between(a: Coordinate, b: Coordinate) -> float:
  """Initial bearing (degrees, 0..360) from a to b."""
  lat1 = math.radians(a.latitude)
  lat2 = math.radians(b.latitude)
  dlon = math.radians(b.longitude - a.longitude)
  y = math.sin(dlon) * math.cos(lat2)
  x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
  return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _segment_projection(a: Coordinate, b: Coordinate, p: Coordinate) -> tuple[float, float]:
  """Distance from p to segment a-b, plus the [0..1] parameter along a-b of the closest point."""
  ax, ay = a.longitude, a.latitude
  bx, by = b.longitude, b.latitude
  px, py = p.longitude, p.latitude
  dx, dy = bx - ax, by - ay
  seg_len_sq = dx * dx + dy * dy
  if seg_len_sq < 1e-12:
    return a.distance_to(p), 0.0
  t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
  proj = Coordinate(ay + t * dy, ax + t * dx)
  return proj.distance_to(p), t


def closest_segment_index(geometry: list[Coordinate], pos: Coordinate) -> tuple[int, float, float]:
  """Index of closest segment, the perpendicular distance (m), and t along that segment."""
  if len(geometry) < 2:
    return 0, geometry[0].distance_to(pos) if geometry else 0.0, 0.0
  best_idx, best_d, best_t = 0, float("inf"), 0.0
  for i in range(len(geometry) - 1):
    d, t = _segment_projection(geometry[i], geometry[i + 1], pos)
    if d < best_d:
      best_idx, best_d, best_t = i, d, t
  return best_idx, best_d, best_t


def distance_along_geometry(geometry: list[Coordinate], pos: Coordinate) -> float:
  """Distance from start of geometry up to the point on geometry closest to pos."""
  if len(geometry) < 2:
    return 0.0
  idx, _, t = closest_segment_index(geometry, pos)
  total = 0.0
  for i in range(idx):
    total += geometry[i].distance_to(geometry[i + 1])
  total += geometry[idx].distance_to(geometry[idx + 1]) * t
  return total


def total_geometry_length(geometry: list[Coordinate]) -> float:
  if len(geometry) < 2:
    return 0.0
  return sum(geometry[i].distance_to(geometry[i + 1]) for i in range(len(geometry) - 1))


def route_bearing_at(geometry: list[Coordinate], pos: Coordinate, lookahead_m: float = 50.0) -> float | None:
  """Bearing of the route at the closest point to pos, measured ~lookahead_m ahead."""
  if len(geometry) < 2:
    return None
  idx, _, t = closest_segment_index(geometry, pos)
  start = geometry[idx]
  # walk forward lookahead_m
  remaining = lookahead_m
  cur = idx
  start_dist_into_segment = geometry[cur].distance_to(geometry[cur + 1]) * t
  remaining -= max(0.0, geometry[cur].distance_to(geometry[cur + 1]) - start_dist_into_segment)
  end = geometry[cur + 1]
  while remaining > 0.0 and cur + 2 < len(geometry):
    cur += 1
    seg_len = geometry[cur].distance_to(geometry[cur + 1])
    if seg_len >= remaining:
      end = geometry[cur + 1]
      break
    remaining -= seg_len
    end = geometry[cur + 1]
  if start.distance_to(end) < 1e-3:
    return None
  return bearing_between(start, end)
