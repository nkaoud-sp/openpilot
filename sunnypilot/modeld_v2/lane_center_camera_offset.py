"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Lane Center Assist — Camera Offset method (experimental).

The alternative to the controls-side curvature bias: instead of correcting the
steering setpoint downstream, drive the model's *input* by dynamically shifting
the virtual camera (the same shear as the static "Adjust Camera Offset"
setting). The model then re-plans around the shifted centre itself, so its
path/laneLines — and the green on-road path — reflect the correction with no
downstream tug. The cost: the loop runs through the neural net, and the shift
moves the model's whole worldview (leads, adjacent lanes), so it is kept small,
slew-limited and opt-in.

Sign: our lane-centre offset is (left+right)/2 in the model frame where +y is
right, so a left-hug reads positive. A positive CameraOffset moves the model's
centre LEFT, so to correct a left-hug we command a NEGATIVE camera offset
(shift the perceived centre right → the model plans right).

Self-contained: duplicates the small lane-line offset computation rather than
sharing it, so the curvature-bias method and this one can be removed
independently.

Caveats this is deliberately conservative about:
  * The measurement comes from the model's (already shifted) lane lines, so the
    control couples back into what it measures — hence small gain + slew.
  * Proportional only, so a persistent hug leaves a small residual. That is an
    accepted trade for not adding integral lag/instability in an experiment.
"""
from __future__ import annotations

import numpy as np

from openpilot.common.params import Params
from openpilot.common.constants import CV

# LaneCenterAssistMethod
METHOD_CURVATURE_BIAS = 0
METHOD_CAMERA_OFFSET = 1

# LaneCenterAssistMode (shared with the curvature method)
MODE_OFF = 0
MODE_READOUT = 1
MODE_ON = 2

# Strength -> camera-offset gain (metres of shift per metre of lane offset). Kept
# well under 1 for stability, since this closes a loop through the model.
STRENGTH_CAM_GAIN = (0.15, 0.25, 0.40)

CONFIDENCE_MIN_PROB = (0.2, 0.4, 0.6)  # 0=loose, 1=normal, 2=strict

NEAR_FIELD_X_LO = 5.0
NEAR_FIELD_X_HI = 25.0
LANE_WIDTH_MIN = 2.6
LANE_WIDTH_MAX = 4.6
OFFSET_MAX_VALID = 1.2
OFFSET_DEADZONE = 0.05

CAM_OFFSET_HARD_MAX = 0.35   # never exceed the range the static setting allows
CAM_OFFSET_SLEW = 0.005      # metres per model frame (~20 Hz) => ~cap in ~1 s

CONTROL_DT = 0.05            # model frame period (s), ~20 Hz
# Derivative damping: opposes the rate of change of the offset to curb the
# oscillation a proportional-only loop through the model tends to develop. Acts
# on a lightly filtered offset so it doesn't chase measurement noise.
DERIV_FILTER_ALPHA = 0.30    # EMA weight on the newest offset for the rate estimate

PARAM_READ_PERIOD_FRAMES = 20  # ~1 Hz at the model rate


def _median_y_near(line):
  xs = np.asarray(line.x)
  ys = np.asarray(line.y)
  if xs.size == 0 or ys.size == 0:
    return None
  mask = (xs >= NEAR_FIELD_X_LO) & (xs <= NEAR_FIELD_X_HI)
  ys_window = ys[mask]
  if ys_window.size == 0:
    ys_window = ys
  return float(np.median(ys_window))


class LaneCenterCameraOffset:
  def __init__(self):
    self.params = Params()
    self._frame = 0

    self.method = METHOD_CURVATURE_BIAS
    self.mode = MODE_OFF
    self.strength = 1
    self.confidence = 0
    self.max_delta = 0.10        # metres, from LaneCenterAssistCamMaxM
    self.damping = 0.0           # seconds, derivative gain (LaneCenterAssistCamDamping); opt-in
    self.gain_override = 0.0     # m/m, direct gain (LaneCenterAssistCamGain); 0 => use Strength
    self.min_speed_ms = 40 * CV.KPH_TO_MS

    self.offset = 0.0            # last measured lane-centre offset (m)
    self.offset_delta = 0.0      # current dynamic camera-offset delta (m)
    self._offset_filt = 0.0      # EMA of the offset, for the derivative term
    self._prev_offset_filt = 0.0

    self.read_params()

  @property
  def active_method(self) -> bool:
    return self.method == METHOD_CAMERA_OFFSET and self.mode == MODE_ON

  def read_params(self) -> None:
    self.method = self.params.get("LaneCenterAssistMethod", return_default=True)
    self.mode = self.params.get("LaneCenterAssistMode", return_default=True)
    self.strength = int(np.clip(self.params.get("LaneCenterAssistStrength", return_default=True), 0, len(STRENGTH_CAM_GAIN) - 1))
    self.confidence = int(np.clip(self.params.get("LaneCenterAssistConfidence", return_default=True), 0, len(CONFIDENCE_MIN_PROB) - 1))
    self.max_delta = min(CAM_OFFSET_HARD_MAX, self.params.get("LaneCenterAssistCamMaxM", return_default=True) / 100.0)
    self.damping = self.params.get("LaneCenterAssistCamDamping", return_default=True) / 100.0
    self.gain_override = self.params.get("LaneCenterAssistCamGain", return_default=True) / 100.0
    self.min_speed_ms = self.params.get("LaneCenterAssistMinKph", return_default=True) * CV.KPH_TO_MS

  def _gain(self) -> float:
    # Direct override when set (>0), otherwise the Strength-based gain.
    return self.gain_override if self.gain_override > 0.0 else STRENGTH_CAM_GAIN[self.strength]

  def _compute_offset(self, model_v2):
    lane_lines = model_v2.laneLines
    probs = model_v2.laneLineProbs
    if len(lane_lines) < 4 or len(probs) < 4:
      return None
    min_prob = CONFIDENCE_MIN_PROB[self.confidence]
    if float(probs[1]) < min_prob or float(probs[2]) < min_prob:
      return None
    left_y = _median_y_near(lane_lines[1])
    right_y = _median_y_near(lane_lines[2])
    if left_y is None or right_y is None:
      return None
    width = right_y - left_y
    if not (LANE_WIDTH_MIN <= width <= LANE_WIDTH_MAX):
      return None
    center = (left_y + right_y) / 2.0
    if abs(center) > OFFSET_MAX_VALID:
      return None
    return center

  def _slew_to(self, target: float) -> float:
    self.offset_delta = float(np.clip(target, self.offset_delta - CAM_OFFSET_SLEW, self.offset_delta + CAM_OFFSET_SLEW))
    return self.offset_delta

  def update(self, model_v2, v_ego: float, lat_active: bool) -> float:
    """Refresh the dynamic camera-offset delta from the latest model output.

    Returns the delta (metres) to add to the static CameraOffset. When the
    camera method is not the selected/active one it ramps out to 0.
    """
    self._frame += 1
    if self._frame % PARAM_READ_PERIOD_FRAMES == 0:
      self.read_params()

    self.offset = 0.0

    offset = None if not self.active_method else self._compute_offset(model_v2)
    if offset is None or not lat_active or v_ego < self.min_speed_ms or abs(offset) < OFFSET_DEADZONE:
      # Not correcting this frame: hold the derivative filter flat so it doesn't
      # register a spike when the assist re-engages.
      self._offset_filt = 0.0
      self._prev_offset_filt = 0.0
      return self._slew_to(0.0)

    self.offset = offset
    # Derivative damping on a lightly filtered offset.
    self._offset_filt += DERIV_FILTER_ALPHA * (offset - self._offset_filt)
    offset_rate = (self._offset_filt - self._prev_offset_filt) / CONTROL_DT
    self._prev_offset_filt = self._offset_filt

    # +offset (left-hug) -> negative camera offset (shift perceived centre right).
    # The damping term subtracts the trend so we stop pushing before overshoot.
    command = self._gain() * offset + self.damping * offset_rate
    target = float(np.clip(-command, -self.max_delta, self.max_delta))
    return self._slew_to(target)
