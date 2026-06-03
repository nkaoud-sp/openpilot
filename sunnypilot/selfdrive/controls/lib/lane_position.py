"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Lane position estimation from modelV2 road edges.

Adapted from the frogpilot fork (nkaoud-fp/openpilot, branch
fp-new-nkaoud-g4.7-nav): divide the gap between left/right road edges by a
standard lane width to estimate total lanes, then place the car within them
by measuring the distance from y=0 to the left road edge. Uses the median
|y| over a near-field window (5-30 m ahead) which is robust to single-frame
outliers and avoids the curve bias picked up by averaging the full 0-192 m
prediction on bends. Hysteresis on the float->int rounding step keeps the
reported lane count and current lane from flickering at the boundaries.
"""
from __future__ import annotations

import numpy as np

LANE_POSITION_MAX_LANES = 6
ROAD_EDGE_STD_HIGH = 1.0
ROAD_EDGE_STD_LOW = 2.5
STANDARD_LANE_WIDTH = 3.7  # metres
EDGE_RESIDUAL_TIGHT = 0.6  # leftover metres after dividing total width by lane width
NEAR_FIELD_X_LO = 5.0      # metres ahead — start of the trusted window
NEAR_FIELD_X_HI = 30.0     # metres ahead — end of the trusted window
EDGE_HYSTERESIS = 0.25     # lane-widths past the half-integer boundary before flipping


def _median_abs_y_near(line, x_lo: float = NEAR_FIELD_X_LO, x_hi: float = NEAR_FIELD_X_HI):
  xs = np.asarray(line.x)
  ys = np.asarray(line.y)
  if xs.size == 0 or ys.size == 0:
    return None
  mask = (xs >= x_lo) & (xs <= x_hi)
  ys_window = ys[mask]
  if ys_window.size == 0:
    ys_window = ys
  return float(np.median(np.abs(ys_window)))


def _hysteresis_round(value: float, last: int) -> int:
  if last <= 0:
    return int(round(value))
  if value >= last + 0.5 + EDGE_HYSTERESIS or value <= last - 0.5 - EDGE_HYSTERESIS:
    return int(round(value))
  return last


class LanePositionEstimator:
  """Stateful road-edge-based lane position estimator.

  Returns (current_lane, total_lanes, confidence) per call:
    current_lane: 1-indexed lane the car is in (0 if unknown)
    total_lanes:  total lanes detected (0 if unknown)
    confidence:   'high' | 'medium' | 'low' | 'unknown'
  """

  def __init__(self):
    self._last_total = 0
    self._last_current = 0

  def update(self, modelV2, max_lanes: int = LANE_POSITION_MAX_LANES):
    road_edges = list(modelV2.roadEdges)
    road_edge_stds = list(modelV2.roadEdgeStds)
    if len(road_edges) < 2 or len(road_edge_stds) < 2:
      return 0, 0, "unknown"

    dist_left = _median_abs_y_near(road_edges[0])
    dist_right = _median_abs_y_near(road_edges[1])
    if dist_left is None or dist_right is None:
      return 0, 0, "unknown"

    total_width = dist_left + dist_right
    if total_width < 1.5:
      self._last_total = 0
      self._last_current = 0
      return 0, 0, "low"

    raw_total = total_width / STANDARD_LANE_WIDTH
    total_lanes = _hysteresis_round(raw_total, self._last_total)
    total_lanes = min(max(total_lanes, 1), max_lanes)

    raw_current = dist_left / STANDARD_LANE_WIDTH + 0.5
    current_lane = _hysteresis_round(raw_current, self._last_current)
    current_lane = max(min(current_lane, total_lanes), 1)

    self._last_total = total_lanes
    self._last_current = current_lane

    residual = abs(total_width - total_lanes * STANDARD_LANE_WIDTH)
    worst_std = max(road_edge_stds[0], road_edge_stds[1])
    edges_strong = worst_std < ROAD_EDGE_STD_HIGH
    edges_ok = worst_std < ROAD_EDGE_STD_LOW
    fits_lane_width = residual < EDGE_RESIDUAL_TIGHT

    if edges_strong and fits_lane_width:
      confidence = "high"
    elif edges_ok or fits_lane_width:
      confidence = "medium"
    else:
      confidence = "low"

    return current_lane, total_lanes, confidence
