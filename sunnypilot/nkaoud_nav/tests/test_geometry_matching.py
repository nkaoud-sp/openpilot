"""
Tests for the route-aware, heading-gated map-matching in geometry.py.

These pin the behaviour that distinguishes closest_segment_in_window from the
plain global-nearest closest_segment_index: it must not snap the vehicle onto a
parallel road / opposing carriageway or a later pass of a route that loops back
near itself.
"""
from openpilot.sunnypilot.nkaoud_nav.geometry import (
  Coordinate as C,
  abs_bearing_diff,
  closest_segment_in_window,
  closest_segment_index,
  distance_along_from,
  distance_along_geometry,
  route_bearing_from,
)


def _straight_east(n: int = 6, step: float = 0.002) -> list[C]:
  # Eastbound polyline along the equator; each segment ~step deg lon (~222 m).
  return [C(0.0, i * step) for i in range(n)]


# An out-and-back: east along lat 0, a short northward connector, then west
# along lat 0.0001 (~11 m north). The outbound and return legs run within a
# lane's width of each other, exactly the geometry that fools global-nearest.
OUT_AND_BACK = [
  C(0.0, 0.000), C(0.0, 0.002), C(0.0, 0.004), C(0.0, 0.006), C(0.0, 0.008), C(0.0, 0.010),
  C(0.0001, 0.010), C(0.0001, 0.008), C(0.0001, 0.006), C(0.0001, 0.004), C(0.0001, 0.002), C(0.0001, 0.000),
]
OUTBOUND_SEGS = range(5)        # eastbound segments
RETURN_SEGS = range(6, 11)      # westbound segments


def test_abs_bearing_diff_wraps():
  assert abs_bearing_diff(10.0, 350.0) == 20.0
  assert abs_bearing_diff(90.0, 270.0) == 180.0
  assert abs_bearing_diff(0.0, 0.0) == 0.0


def test_global_nearest_mis_snaps_to_return_leg():
  # Vehicle nearer the return leg but actually driving the outbound leg east.
  pos = C(0.00007, 0.005)
  idx, _, _ = closest_segment_index(OUT_AND_BACK, pos)
  # Documents the failure mode the window fix exists to prevent.
  assert idx in RETURN_SEGS


def test_window_localizes_to_outbound_without_heading():
  pos = C(0.00007, 0.005)
  # Seeded on the outbound leg; the default forward window can't reach the
  # return leg, so the return segments are excluded by index alone.
  idx, perp, _ = closest_segment_in_window(OUT_AND_BACK, pos, start_idx=2, bearing=None)
  assert idx in OUTBOUND_SEGS
  assert perp < 10.0


def test_heading_gate_rejects_opposing_carriageway():
  pos = C(0.00007, 0.005)
  # Even scanning the whole route (re-acquire mode), an eastbound heading must
  # reject the westbound return leg and land on the outbound leg.
  idx, _, _ = closest_segment_in_window(
    OUT_AND_BACK, pos, start_idx=0,
    window_back_m=0.0, window_fwd_m=float("inf"),
    bearing=90.0, max_bearing_diff_deg=75.0,
  )
  assert idx in OUTBOUND_SEGS


def test_heading_gate_falls_back_when_nothing_agrees():
  # Westbound route, but we claim an eastbound heading: no segment agrees, so it
  # must still return the nearest segment rather than an empty/degenerate match.
  geom = [C(0.0001, 0.010 - i * 0.002) for i in range(6)]  # all westbound
  pos = C(0.0, 0.005)
  idx, perp, t = closest_segment_in_window(geom, pos, start_idx=0, bearing=90.0)
  assert 0 <= idx < len(geom) - 1
  assert perp != float("inf")
  assert 0.0 <= t <= 1.0


def test_window_matches_global_when_unrestricted():
  # With no heading gate and an unbounded window, the windowed match reduces to
  # plain global-nearest.
  geom = _straight_east()
  pos = C(0.00003, 0.005)
  g_idx, g_perp, g_t = closest_segment_index(geom, pos)
  w_idx, w_perp, w_t = closest_segment_in_window(
    geom, pos, start_idx=0, window_back_m=1e12, window_fwd_m=float("inf"), bearing=None,
  )
  assert w_idx == g_idx
  assert w_perp == g_perp
  assert w_t == g_t


def test_distance_along_from_matches_projection():
  geom = _straight_east()
  pos = C(0.00003, 0.005)
  idx, _, t = closest_segment_index(geom, pos)
  assert distance_along_from(geom, idx, t) == distance_along_geometry(geom, pos)


def test_distance_along_from_is_monotonic_along_route():
  geom = _straight_east()
  near_start = distance_along_from(geom, 0, 0.25)
  near_end = distance_along_from(geom, 3, 0.75)
  assert near_end > near_start


def test_route_bearing_from_reads_forward_heading():
  geom = _straight_east()
  # From the middle of the eastbound route, the forward bearing is ~90 deg.
  b = route_bearing_from(geom, 1, 0.5, lookahead_m=50.0)
  assert b is not None
  assert abs_bearing_diff(b, 90.0) < 5.0
