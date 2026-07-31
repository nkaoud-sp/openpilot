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


def abs_bearing_diff(a: float, b: float) -> float:
  """Smallest absolute angle (0..180) between two compass bearings."""
  d = abs((a - b) % 360.0)
  return min(d, 360.0 - d)


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


def closest_segment_in_window(
  geometry: list[Coordinate],
  pos: Coordinate,
  start_idx: int,
  window_back_m: float = 40.0,
  window_fwd_m: float = 600.0,
  bearing: float | None = None,
  max_bearing_diff_deg: float = 75.0,
) -> tuple[int, float, float]:
  """Route-aware projection of pos onto geometry.

  Unlike closest_segment_index (which scans the whole polyline and returns the
  globally nearest segment), this restricts the search to a window of the route
  around start_idx and, when a heading is supplied, rejects segments whose
  direction disagrees with it. That stops the projection from snapping onto a
  parallel road, the opposing carriageway of a divided highway, or a later/
  earlier pass of a route that loops back near itself -- all of which corrupt
  distance-to-maneuver and cross-track.

  Pass window_fwd_m=inf (and window_back_m=0, start_idx=0) to scan the whole
  route while still heading-gating -- used to re-acquire after losing the local
  anchor.

  Returns (segment_index, perpendicular_distance_m, t_along_segment), matching
  closest_segment_index's shape. Falls back to the heading-agnostic nearest
  in-window segment if every candidate disagrees with `bearing`.
  """
  n = len(geometry)
  if n < 2:
    return 0, geometry[0].distance_to(pos) if geometry else 0.0, 0.0

  start_idx = max(0, min(start_idx, n - 2))

  # Grow a segment window backward/forward from start_idx by arc length.
  lo = start_idx
  acc = 0.0
  while lo > 0 and acc < window_back_m:
    lo -= 1
    acc += geometry[lo].distance_to(geometry[lo + 1])
  hi = start_idx
  acc = 0.0
  while hi < n - 2 and acc < window_fwd_m:
    acc += geometry[hi].distance_to(geometry[hi + 1])
    hi += 1

  best = (start_idx, float("inf"), 0.0)       # nearest heading-agreeing segment
  best_any = (start_idx, float("inf"), 0.0)   # nearest regardless of heading
  for i in range(lo, hi + 1):
    d, t = _segment_projection(geometry[i], geometry[i + 1], pos)
    if d < best_any[1]:
      best_any = (i, d, t)
    if bearing is not None:
      if geometry[i].distance_to(geometry[i + 1]) < 1e-6:
        continue
      seg_bearing = bearing_between(geometry[i], geometry[i + 1])
      if abs_bearing_diff(bearing, seg_bearing) > max_bearing_diff_deg:
        continue
    if d < best[1]:
      best = (i, d, t)

  return best if best[1] != float("inf") else best_any


def distance_along_from(geometry: list[Coordinate], idx: int, t: float) -> float:
  """Arc length from the start of geometry to the point identified by
  (segment idx, fraction t) -- i.e. the output of a projection helper."""
  if len(geometry) < 2:
    return 0.0
  idx = max(0, min(idx, len(geometry) - 2))
  total = 0.0
  for i in range(idx):
    total += geometry[i].distance_to(geometry[i + 1])
  total += geometry[idx].distance_to(geometry[idx + 1]) * max(0.0, min(1.0, t))
  return total


def distance_along_geometry(geometry: list[Coordinate], pos: Coordinate) -> float:
  """Distance from start of geometry up to the point on geometry closest to pos."""
  if len(geometry) < 2:
    return 0.0
  idx, _, t = closest_segment_index(geometry, pos)
  return distance_along_from(geometry, idx, t)


def total_geometry_length(geometry: list[Coordinate]) -> float:
  if len(geometry) < 2:
    return 0.0
  return sum(geometry[i].distance_to(geometry[i + 1]) for i in range(len(geometry) - 1))


def route_bearing_from(geometry: list[Coordinate], idx: int, t: float, lookahead_m: float = 50.0) -> float | None:
  """Bearing of the route from an already-matched point (segment idx, fraction
  t), measured ~lookahead_m ahead. Lets callers reuse a route-aware match rather
  than re-projecting globally."""
  n = len(geometry)
  if n < 2:
    return None
  idx = max(0, min(idx, n - 2))
  start = geometry[idx]
  # walk forward lookahead_m
  remaining = lookahead_m
  cur = idx
  start_dist_into_segment = geometry[cur].distance_to(geometry[cur + 1]) * t
  remaining -= max(0.0, geometry[cur].distance_to(geometry[cur + 1]) - start_dist_into_segment)
  end = geometry[cur + 1]
  while remaining > 0.0 and cur + 2 < n:
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


def route_bearing_at(geometry: list[Coordinate], pos: Coordinate, lookahead_m: float = 50.0) -> float | None:
  """Bearing of the route at the closest point to pos, measured ~lookahead_m ahead."""
  if len(geometry) < 2:
    return None
  idx, _, t = closest_segment_index(geometry, pos)
  return route_bearing_from(geometry, idx, t, lookahead_m)
