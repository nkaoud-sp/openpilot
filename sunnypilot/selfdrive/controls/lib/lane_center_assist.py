"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Lane Center Assist.

Different driving models hug the lane differently (some sit left, some sit
right). This adds a small, capped curvature bias on top of the model's
desiredCurvature that nudges the car back toward the geometric centre of the
ego lane. It is deliberately weak: a trim on persistent hugging, not a
controller that fights the model.

How it works:
  * Read the two inner lane lines (modelV2.laneLines[1] left, [2] right) over a
    near-field window and take the median y of each (robust to single-frame
    outliers and to curve bias picked up by averaging the full prediction).
  * The lane centre relative to the car is the midpoint of those two lines.
    In openpilot's model frame y is positive to the right and the car is at
    y=0 (laneLines[1] left line sits at negative y, laneLines[2] right line at
    positive y — see ldw.py), so a positive midpoint means the lane centre is
    to our right (the model is hugging left) and we should steer right to
    re-centre.
  * Convert that lateral offset into a target lateral acceleration (capped),
    then into a curvature bias (a_lat / v^2) so the *feel* is consistent across
    speeds. The bias is slew-limited for smooth engage/disengage and is only
    applied in Mode "On"; clip_curvature downstream still bounds the total.

Self-contained by design: it shares no code with lane_position.py so either
feature can be removed independently.
"""
from __future__ import annotations

import numpy as np

from cereal import log
from openpilot.common.params import Params
from openpilot.common.constants import CV

LaneChangeState = log.LaneChangeState

# Modes (LaneCenterAssistMode)
MODE_OFF = 0
MODE_READOUT = 1
MODE_ON = 2

# Method (LaneCenterAssistMethod): this controls-side bias only runs for method 0
METHOD_CURVATURE_BIAS = 0
METHOD_CAMERA_OFFSET = 1

# Strength -> proportional gain from lateral offset (m) to target lat accel (m/s^2 per m)
STRENGTH_GAIN = (0.4, 0.7, 1.0)

# Confidence gate -> minimum probability required on both inner lane lines
CONFIDENCE_MIN_PROB = (0.2, 0.4, 0.6)  # 0=loose, 1=normal, 2=strict

# Near-field window used to measure the inner lane lines
NEAR_FIELD_X_LO = 5.0    # metres ahead
NEAR_FIELD_X_HI = 25.0   # metres ahead

# Sanity bounds
LANE_WIDTH_MIN = 2.6     # metres — below this the inner lines are not a real lane
LANE_WIDTH_MAX = 4.6     # metres — above this the detection is unreliable
OFFSET_MAX_VALID = 1.2   # metres — a larger midpoint offset is treated as a bad detection
OFFSET_DEADZONE = 0.05   # metres — ignore tiny offsets so we don't micro-steer when centred

V_EGO_FLOOR = 5.0        # m/s — floor used in a_lat / v^2 so the bias can't blow up
BIAS_SLEW = 3.0e-5       # 1/m per control frame (100 Hz) — ~1.5 s to reach a full nudge

PARAM_READ_PERIOD_FRAMES = 50  # refresh params at ~2 Hz (controlsd runs at 100 Hz)

# Positive desiredCurvature steers right (see latcontrol_angle.py); a positive
# lane-centre midpoint means the centre is to our right (we're hugging left), so a
# positive bias steers us toward it. Field-flippable if a real drive disagrees.
CENTER_SIGN = 1.0


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


class LaneCenterAssist:
  def __init__(self):
    self.params = Params()
    self._frame = 0

    # tunables (refreshed from params)
    self.mode = MODE_OFF
    self.method = METHOD_CURVATURE_BIAS
    self.strength = 1
    self.max_accel = 0.35        # m/s^2, hard cap on the added lateral accel
    self.min_speed_ms = 40 * CV.KPH_TO_MS
    self.confidence = 0

    # state
    self._bias = 0.0             # slew-limited curvature bias currently applied
    self.offset = 0.0            # measured lateral offset from lane centre (m)
    self.lat_accel_cmd = 0.0     # commanded lateral accel before curvature conversion
    self.active = False
    self.reason = "off"

    self._read_params()

  def _read_params(self) -> None:
    self.mode = self.params.get("LaneCenterAssistMode", return_default=True)
    self.method = self.params.get("LaneCenterAssistMethod", return_default=True)
    self.strength = int(np.clip(self.params.get("LaneCenterAssistStrength", return_default=True), 0, len(STRENGTH_GAIN) - 1))
    self.max_accel = self.params.get("LaneCenterAssistMaxAccel", return_default=True) / 100.0
    self.min_speed_ms = self.params.get("LaneCenterAssistMinKph", return_default=True) * CV.KPH_TO_MS
    self.confidence = int(np.clip(self.params.get("LaneCenterAssistConfidence", return_default=True), 0, len(CONFIDENCE_MIN_PROB) - 1))

  def _compute_offset(self, model_v2) -> tuple[bool, float, str]:
    lane_lines = model_v2.laneLines
    probs = model_v2.laneLineProbs
    if len(lane_lines) < 4 or len(probs) < 4:
      return False, 0.0, "no lines"

    min_prob = CONFIDENCE_MIN_PROB[self.confidence]
    if float(probs[1]) < min_prob or float(probs[2]) < min_prob:
      return False, 0.0, "low conf"

    left_y = _median_y_near(lane_lines[1])
    right_y = _median_y_near(lane_lines[2])
    if left_y is None or right_y is None:
      return False, 0.0, "no points"

    width = right_y - left_y  # right is +y, left is -y, so width is positive
    if not (LANE_WIDTH_MIN <= width <= LANE_WIDTH_MAX):
      return False, 0.0, "bad width"

    center = (left_y + right_y) / 2.0
    if abs(center) > OFFSET_MAX_VALID:
      return False, 0.0, "outlier"

    return True, center, "ok"

  def _slew_to(self, target: float) -> float:
    self._bias = float(np.clip(target, self._bias - BIAS_SLEW, self._bias + BIAS_SLEW))
    return self._bias

  def update(self, model_v2, v_ego: float, lat_active: bool) -> float:
    """Return a curvature bias (1/m) to add to the desired curvature."""
    self._frame += 1
    if self._frame % PARAM_READ_PERIOD_FRAMES == 0:
      self._read_params()

    self.offset = 0.0
    self.lat_accel_cmd = 0.0
    self.active = False

    # Off / Readout never touch steering. Readout is surfaced by the on-road
    # element, which computes independently, so here we just ramp out.
    if self.mode != MODE_ON:
      self.reason = "off" if self.mode == MODE_OFF else "readout"
      return self._slew_to(0.0)

    # The camera-offset method corrects in modeld instead; don't also bias here.
    if self.method != METHOD_CURVATURE_BIAS:
      self.reason = "camera offset"
      return self._slew_to(0.0)

    if not lat_active:
      self.reason = "not active"
      return self._slew_to(0.0)
    if v_ego < self.min_speed_ms:
      self.reason = "speed"
      return self._slew_to(0.0)
    if model_v2.meta.laneChangeState != LaneChangeState.off:
      self.reason = "lane change"
      return self._slew_to(0.0)

    ok, offset, reason = self._compute_offset(model_v2)
    if not ok:
      self.reason = reason
      return self._slew_to(0.0)

    self.offset = offset
    if abs(offset) < OFFSET_DEADZONE:
      self.reason = "centered"
      return self._slew_to(0.0)

    lat_accel = CENTER_SIGN * STRENGTH_GAIN[self.strength] * offset
    lat_accel = float(np.clip(lat_accel, -self.max_accel, self.max_accel))
    self.lat_accel_cmd = lat_accel

    v = max(v_ego, V_EGO_FLOOR)
    target_bias = lat_accel / (v * v)

    self.reason = "active"
    self.active = True
    return self._slew_to(target_bias)
