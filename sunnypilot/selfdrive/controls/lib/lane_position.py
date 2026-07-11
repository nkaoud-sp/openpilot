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

Optional edge-lane blocking filter (ported from codex/lane-edge-filter-test):
demotes "fake" outer lanes (shoulders, gore stripes, off-ramps) so the
reported current/total counts only include usable lanes. Two heuristics
vote per side and a 3-frame debounce gates the block. The mode is chosen
externally and passed into update():
  0 = none (off, base behaviour)
  1 = width vote only
  2 = separation vote only
  3 = both (AND -- conservative)
  4 = both (OR  -- aggressive)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LANE_POSITION_MAX_LANES = 6
ROAD_EDGE_STD_HIGH = 1.0
ROAD_EDGE_STD_LOW = 2.5
STANDARD_LANE_WIDTH = 3.7  # metres
EDGE_RESIDUAL_TIGHT = 0.6  # leftover metres after dividing total width by lane width
NEAR_FIELD_X_LO = 5.0      # metres ahead — start of the trusted window
NEAR_FIELD_X_HI = 30.0     # metres ahead — end of the trusted window
EDGE_HYSTERESIS = 0.25     # lane-widths past the half-integer boundary before flipping

# Edge-lane blocking filter tunables
LANE_LINE_PROB_STRONG = 0.6
LANE_LINE_PROB_OUTER_WEAK_MAX = 0.50
LANE_LINE_PROB_OUTER_WEAK_GAP = 0.30
EDGE_LANE_NARROW_RATIO = 0.95 # 0.8
EDGE_BLOCK_COUNTER_ON = 3
EDGE_BLOCK_COUNTER_OFF = 1
EDGE_BLOCK_COUNTER_MAX = 6

# Filter modes
FILTER_MODE_NONE = 0
FILTER_MODE_WIDTH = 1
FILTER_MODE_SEPARATION = 2
FILTER_MODE_BOTH_AND = 3
FILTER_MODE_BOTH_OR = 4


@dataclass
class LanePositionDebug:
  raw_current_lane: int = 0
  raw_total_lanes: int = 0
  usable_current_lane: int = 0
  usable_total_lanes: int = 0
  confidence: str = "unknown"
  blocked_left: bool = False
  blocked_right: bool = False
  width_vote_left: bool = False
  width_vote_right: bool = False
  separation_vote_left: bool = False
  separation_vote_right: bool = False
  left_block_counter: int = 0
  right_block_counter: int = 0


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


def _median_y_near(line, x_lo: float = NEAR_FIELD_X_LO, x_hi: float = NEAR_FIELD_X_HI):
  xs = np.asarray(line.x)
  ys = np.asarray(line.y)
  if xs.size == 0 or ys.size == 0:
    return None
  mask = (xs >= x_lo) & (xs <= x_hi)
  ys_window = ys[mask]
  if ys_window.size == 0:
    ys_window = ys
  return float(np.median(ys_window))


def _hysteresis_round(value: float, last: int) -> int:
  if last <= 0:
    return int(round(value))
  if value >= last + 0.5 + EDGE_HYSTERESIS or value <= last - 0.5 - EDGE_HYSTERESIS:
    return int(round(value))
  return last


class LanePositionEstimator:
  """Stateful road-edge-based lane position estimator.

  Returns (current_lane, total_lanes, confidence) per call:
    current_lane: 1-indexed usable lane the car is in (0 if unknown)
    total_lanes:  usable lanes detected (0 if unknown)
    confidence:   'high' | 'medium' | 'low' | 'unknown'
  """

  def __init__(self):
    self._last_total = 0
    self._last_current = 0
    self._left_block_counter = 0
    self._right_block_counter = 0
    self._left_blocked = False
    self._right_blocked = False
    self._debug = LanePositionDebug()

  @property
  def debug(self) -> LanePositionDebug:
    return self._debug

  @staticmethod
  def _lane_width_between(left_y, right_y):
    if left_y is None or right_y is None or right_y <= left_y:
      return None
    return right_y - left_y

  @staticmethod
  def _update_block_counter(counter: int, vote: bool) -> int:
    if vote:
      return min(EDGE_BLOCK_COUNTER_MAX, counter + 1)
    return max(0, counter - 1)

  @staticmethod
  def _latch_block(was_blocked: bool, counter: int) -> bool:
    # Asymmetric thresholds: engage at >= ON, release only at <= OFF.
    # Anything between OFF+1 and ON-1 keeps the previous state, so a vote
    # that twitches True/False each frame around counter=ON cannot chatter.
    if was_blocked:
      return counter > EDGE_BLOCK_COUNTER_OFF
    return counter >= EDGE_BLOCK_COUNTER_ON

  @staticmethod
  def _outer_line_weak_relative_to_inner(inner_prob: float, outer_prob: float) -> bool:
    return (
      inner_prob >= LANE_LINE_PROB_STRONG
      and outer_prob <= LANE_LINE_PROB_OUTER_WEAK_MAX
      and (inner_prob - outer_prob) >= LANE_LINE_PROB_OUTER_WEAK_GAP
    )

  def update(self, modelV2, max_lanes: int = LANE_POSITION_MAX_LANES, filter_mode: int = FILTER_MODE_NONE):
    road_edges = list(modelV2.roadEdges)
    road_edge_stds = list(modelV2.roadEdgeStds)
    if len(road_edges) < 2 or len(road_edge_stds) < 2:
      self._debug = LanePositionDebug()
      return 0, 0, "unknown"

    dist_left = _median_abs_y_near(road_edges[0])
    dist_right = _median_abs_y_near(road_edges[1])
    if dist_left is None or dist_right is None:
      self._debug = LanePositionDebug()
      return 0, 0, "unknown"

    total_width = dist_left + dist_right
    if total_width < 1.5:
      self._last_total = 0
      self._last_current = 0
      self._left_block_counter = 0
      self._right_block_counter = 0
      self._left_blocked = False
      self._right_blocked = False
      self._debug = LanePositionDebug(confidence="low")
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

    if filter_mode == FILTER_MODE_NONE:
      self._left_block_counter = 0
      self._right_block_counter = 0
      self._left_blocked = False
      self._right_blocked = False
      self._debug = LanePositionDebug(
        raw_current_lane=current_lane,
        raw_total_lanes=total_lanes,
        usable_current_lane=current_lane,
        usable_total_lanes=total_lanes,
        confidence=confidence,
      )
      return current_lane, total_lanes, confidence

    # Filter on: read lane-line geometry and probabilities for the inner pair
    lane_lines = list(modelV2.laneLines)
    lane_line_probs = list(modelV2.laneLineProbs)[:4]
    lane_line_probs += [0.0] * (4 - len(lane_line_probs))

    line_ys = [_median_y_near(line) for line in lane_lines[:4]]
    line_ys += [None] * (4 - len(line_ys))
    left_outer_y, left_inner_y, right_inner_y, right_outer_y = line_ys[:4]

    ego_lane_width = self._lane_width_between(left_inner_y, right_inner_y)
    left_edge_lane_width = self._lane_width_between(left_outer_y, left_inner_y)
    right_edge_lane_width = self._lane_width_between(right_inner_y, right_outer_y)

    width_vote_left = False
    width_vote_right = False
    if ego_lane_width is not None and ego_lane_width > 0.1:
      if left_edge_lane_width is not None:
        width_vote_left = (left_edge_lane_width / ego_lane_width) < EDGE_LANE_NARROW_RATIO
      if right_edge_lane_width is not None:
        width_vote_right = (right_edge_lane_width / ego_lane_width) < EDGE_LANE_NARROW_RATIO

    left_inner_prob = float(lane_line_probs[1])
    right_inner_prob = float(lane_line_probs[2])
    left_outer_prob = float(lane_line_probs[0])
    right_outer_prob = float(lane_line_probs[3])

    separation_vote_left = (
      current_lane > 1
      and self._outer_line_weak_relative_to_inner(left_inner_prob, left_outer_prob)
    )
    separation_vote_right = (
      current_lane < total_lanes
      and self._outer_line_weak_relative_to_inner(right_inner_prob, right_outer_prob)
    )

    if filter_mode == FILTER_MODE_WIDTH:
      left_candidate = width_vote_left
      right_candidate = width_vote_right
    elif filter_mode == FILTER_MODE_SEPARATION:
      left_candidate = separation_vote_left
      right_candidate = separation_vote_right
    elif filter_mode == FILTER_MODE_BOTH_OR:
      left_candidate = width_vote_left or separation_vote_left
      right_candidate = width_vote_right or separation_vote_right
    else:  # FILTER_MODE_BOTH_AND (default when on)
      left_candidate = width_vote_left and separation_vote_left
      right_candidate = width_vote_right and separation_vote_right

    self._left_block_counter = self._update_block_counter(self._left_block_counter, left_candidate)
    self._right_block_counter = self._update_block_counter(self._right_block_counter, right_candidate)

    self._left_blocked = self._latch_block(self._left_blocked, self._left_block_counter)
    self._right_blocked = self._latch_block(self._right_blocked, self._right_block_counter)
    blocked_left = self._left_blocked
    blocked_right = self._right_blocked

    usable_total = max(1, total_lanes - int(blocked_left) - int(blocked_right))
    usable_current = current_lane - int(blocked_left)
    usable_current = max(1, min(usable_current, usable_total))

    self._debug = LanePositionDebug(
      raw_current_lane=current_lane,
      raw_total_lanes=total_lanes,
      usable_current_lane=usable_current,
      usable_total_lanes=usable_total,
      confidence=confidence,
      blocked_left=blocked_left,
      blocked_right=blocked_right,
      width_vote_left=width_vote_left,
      width_vote_right=width_vote_right,
      separation_vote_left=separation_vote_left,
      separation_vote_right=separation_vote_right,
      left_block_counter=self._left_block_counter,
      right_block_counter=self._right_block_counter,
    )

    return usable_current, usable_total, confidence
