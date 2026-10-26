import logging
import time

import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.ldw import CAMERA_OFFSET
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan

try:
  from openpilot.common.swaglog import cloudlog
except ModuleNotFoundError:
  cloudlog = logging.getLogger(__name__)


DT_MDL = 1.0 / ModelConstants.MODEL_RUN_FREQ
LANE_POLICY_ENABLED_PARAM = "LanePolicyEnabled"
LANE_LOCK_ARM_LINE_PROB = 0.92
LANE_LOCK_RETAIN_LINE_PROB = 0.70
LANE_LOCK_MIN_LANE_WIDTH = 2.6
LANE_LOCK_MAX_LANE_WIDTH = 5.2
LANE_LOCK_MIN_WIDTH_EDGE = 0.10
LANE_LOCK_MAX_WIDTH_CHANGE = 0.75
LANE_LOCK_ARM_TIME = 0.75
LANE_LOCK_WIDTH_TIME = 9.95
LANE_LOCK_ONE_LINE_HOLD_TIME = 1.00
LANE_LOCK_FIT_START = 8.0
LANE_LOCK_FIT_END = 55.0
LANE_LOCK_E2E_BLEND_START = 10.0
LANE_LOCK_E2E_BLEND_END = 45.0
LANE_LOCK_E2E_FALLBACK_END = 30.0
LANE_LOCK_PATH_WEIGHT_SPREAD = 0.25
LANE_LOCK_MIN_LOOKAHEAD = 25.0
LANE_LOCK_MAX_LOOKAHEAD = 45.0
LANE_LOCK_HEADING_GAIN = 0.55
LANE_LOCK_HEADING_CURVATURE_FADE = 0.00060
LANE_LOCK_CURVE_REVERSAL_HOLD_TIME = 0.30
LANE_LOCK_MAX_CENTER_CORRECTION = 0.00045
LANE_LOCK_TURN_CURVATURE = 0.00015
LANE_LOCK_TURN_RELEASE_TIME = 0.35
LANE_LOCK_CORRECTION_ENGAGE_STEP = 0.00008
LANE_LOCK_CORRECTION_RELEASE_STEP = 0.00020
LANE_LOCK_CORRECTION_DEADBAND = 0.000012
LANE_LOCK_MAX_LANE_CHANGE_PROB = 0.10
LANE_LOCK_LOG_INTERVAL = 1.0
LANE_LOCK_LEAD_MIN_PROB = 0.60
LANE_LOCK_LEAD_MIN_DISTANCE = 8.0
LANE_LOCK_LEAD_CLOSE_MIN_DISTANCE = 2.0
LANE_LOCK_LEAD_MAX_DISTANCE = 60.0
LANE_LOCK_LEAD_MAX_LATERAL = 2.0
LANE_LOCK_LEAD_MAX_LATERAL_RATIO = 0.08
LANE_LOCK_LEAD_MAX_CORRECTION = 0.00020

LANE_POLICY_INTENSITY_SCALE = (0.6, 1.0, 1.4)
LANE_POLICY_INTENSITY_NORMAL = 1
TURN_CENTERING_CURVATURE = LANE_LOCK_TURN_CURVATURE

LANE_POLICY_MODE_INACTIVE = 'inactive'
LANE_POLICY_MODE_TWO_LINE = 'twoLine'
LANE_POLICY_MODE_ONE_LINE = 'oneLine'
LANE_POLICY_MODE_LEAD = 'lead'

_lane_lock_weight = 0.0
_lane_lock_lane_curvature = 0.0
_lane_lock_has_lane_curvature = False
_lane_lock_full_active = False
_lane_lock_ready = False
_lane_lock_arm_time = 0.0
_lane_lock_width = 3.7
_lane_lock_width_valid = False
_lane_lock_line_loss_time = 0.0
_lane_lock_center_correction = 0.0
_lane_lock_has_center_correction = False
_lane_lock_one_line_hold = False
_lane_lock_error_logged = False
_lane_lock_last_turn_sign = 0
_lane_lock_turn_release_time = 0.0
_lane_lock_last_curve_target_sign = 0
_lane_lock_curve_reversal_hold_time = 0.0
_lane_lock_last_logged_mode = None
_lane_lock_last_log_time = 0.0
_lane_policy_mode = LANE_POLICY_MODE_INACTIVE
_lane_policy_correction = 0.0


def get_intensity_scale(intensity: int) -> float:
  if 0 <= intensity < len(LANE_POLICY_INTENSITY_SCALE):
    return LANE_POLICY_INTENSITY_SCALE[intensity]
  return LANE_POLICY_INTENSITY_SCALE[LANE_POLICY_INTENSITY_NORMAL]


def get_lane_policy_status() -> tuple[str, float]:
  return _lane_policy_mode, _lane_policy_correction


def reset_lane_lock() -> None:
  global _lane_lock_weight, _lane_lock_lane_curvature
  global _lane_lock_has_lane_curvature, _lane_lock_full_active
  global _lane_lock_ready, _lane_lock_arm_time
  global _lane_lock_width, _lane_lock_width_valid, _lane_lock_line_loss_time
  global _lane_lock_center_correction, _lane_lock_has_center_correction
  global _lane_lock_one_line_hold, _lane_lock_last_turn_sign, _lane_lock_turn_release_time
  global _lane_lock_last_curve_target_sign, _lane_lock_curve_reversal_hold_time
  global _lane_policy_mode, _lane_policy_correction
  _lane_lock_weight = 0.0
  _lane_lock_lane_curvature = 0.0
  _lane_lock_has_lane_curvature = False
  _lane_lock_full_active = False
  _lane_lock_ready = False
  _lane_lock_arm_time = 0.0
  _lane_lock_width = 3.7
  _lane_lock_width_valid = False
  _lane_lock_line_loss_time = 0.0
  _lane_lock_center_correction = 0.0
  _lane_lock_has_center_correction = False
  _lane_lock_one_line_hold = False
  _lane_lock_last_turn_sign = 0
  _lane_lock_turn_release_time = 0.0
  _lane_lock_last_curve_target_sign = 0
  _lane_lock_curve_reversal_hold_time = 0.0
  _lane_policy_mode = LANE_POLICY_MODE_INACTIVE
  _lane_policy_correction = 0.0


def log_lane_lock_mode(mode: str) -> None:
  global _lane_lock_last_logged_mode, _lane_lock_last_log_time
  now = time.monotonic()
  if mode != _lane_lock_last_logged_mode and now - _lane_lock_last_log_time >= LANE_LOCK_LOG_INTERVAL:
    cloudlog.info(f"lane-policy: {mode}")
    _lane_lock_last_logged_mode = mode
    _lane_lock_last_log_time = now


def ramp_out_correction(e2e_curvature: float) -> float:
  global _lane_lock_center_correction, _lane_policy_correction
  _lane_lock_center_correction -= float(np.clip(_lane_lock_center_correction,
                                                -LANE_LOCK_CORRECTION_RELEASE_STEP,
                                                LANE_LOCK_CORRECTION_RELEASE_STEP))
  if abs(_lane_lock_center_correction) < LANE_LOCK_CORRECTION_DEADBAND:
    _lane_lock_center_correction = 0.0
  _lane_policy_correction = float(_lane_lock_center_correction)
  return float(e2e_curvature + _lane_lock_center_correction)


def release_lane_lock(e2e_curvature: float, mode: str | None = None) -> float:
  global _lane_lock_center_correction, _lane_lock_has_center_correction
  correction = _lane_lock_center_correction
  reset_lane_lock()
  _lane_lock_center_correction = correction
  _lane_lock_has_center_correction = True
  if mode is not None:
    log_lane_lock_mode(mode)
  return ramp_out_correction(e2e_curvature)


def get_inner_lane_line_probs(model_output: dict[str, np.ndarray]) -> tuple[float, float]:
  lane_line_probs = np.asarray(model_output['lane_lines_prob'])
  if lane_line_probs.shape != (1, 8):
    raise ValueError(f"expected lane_lines_prob shape (1, 8), got {lane_line_probs.shape}")
  return float(lane_line_probs[0, 3]), float(lane_line_probs[0, 5])


def get_lane_width_measurement(left_y: np.ndarray, right_y: np.ndarray,
                               fit: np.ndarray) -> tuple[float, bool]:
  lane_width = right_y - left_y
  width_p10, width_median, width_p90 = np.percentile(lane_width[fit], (10.0, 50.0, 90.0))
  width_change = float(width_p90 - width_p10)
  width_edge_distance = min(width_p10 - LANE_LOCK_MIN_LANE_WIDTH,
                            LANE_LOCK_MAX_LANE_WIDTH - width_p90)
  valid = (LANE_LOCK_MIN_LANE_WIDTH <= width_median <= LANE_LOCK_MAX_LANE_WIDTH and
           width_edge_distance >= LANE_LOCK_MIN_WIDTH_EDGE and
           width_change <= LANE_LOCK_MAX_WIDTH_CHANGE)
  return float(width_median), bool(valid)


def get_lead_center_correction(model_output: dict[str, np.ndarray], v_ego: float,
                               close_lead_enabled: bool = False) -> float | None:
  lead_prob = np.asarray(model_output['lead_prob'])
  leads = np.asarray(model_output['lead'])
  if (lead_prob.ndim != 2 or leads.ndim != 4 or leads.shape[-1] < 2 or
      lead_prob.shape[1] < 1 or leads.shape[1] < 1):
    raise ValueError(f"unexpected lead shapes: lead_prob={lead_prob.shape}, lead={leads.shape}")

  lead_prob_now = float(lead_prob[0, 0])
  lead_x = float(leads[0, 0, 0, 0])
  lead_y = float(leads[0, 0, 0, 1])
  min_distance = LANE_LOCK_LEAD_CLOSE_MIN_DISTANCE if close_lead_enabled else LANE_LOCK_LEAD_MIN_DISTANCE
  if (not np.isfinite(lead_prob_now) or not np.isfinite(lead_x) or not np.isfinite(lead_y) or
      lead_prob_now < LANE_LOCK_LEAD_MIN_PROB or
      lead_x < min_distance or lead_x > LANE_LOCK_LEAD_MAX_DISTANCE or
      abs(lead_y) > min(LANE_LOCK_LEAD_MAX_LATERAL, LANE_LOCK_LEAD_MAX_LATERAL_RATIO * lead_x)):
    return None

  lookahead = float(np.clip(min(lead_x, 2.0 * v_ego), LANE_LOCK_MIN_LOOKAHEAD, LANE_LOCK_MAX_LOOKAHEAD))
  correction = 2.0 * (lead_y + CAMERA_OFFSET) / (lookahead * lookahead)
  return float(np.clip(correction, -LANE_LOCK_LEAD_MAX_CORRECTION, LANE_LOCK_LEAD_MAX_CORRECTION))


def get_e2e_blend_gain(x: np.ndarray | float) -> np.ndarray | float:
  progress = np.clip((np.asarray(x) - LANE_LOCK_E2E_BLEND_START) /
                     (LANE_LOCK_E2E_BLEND_END - LANE_LOCK_E2E_BLEND_START), 0.0, 1.0)
  return progress ** 3 * (10.0 + progress * (-15.0 + 6.0 * progress))


def get_e2e_blended_center_correction(model_output: dict[str, np.ndarray], center_y: np.ndarray,
                                      x: np.ndarray, v_ego: float) -> float:
  position = np.asarray(model_output['plan'][0, :, Plan.POSITION], dtype=np.float64)
  if position.ndim != 2 or position.shape[1] < 2:
    raise ValueError("E2E position path has an invalid shape")

  plan_x, plan_y = position[:, 0], position[:, 1]
  valid_plan = np.isfinite(plan_x) & np.isfinite(plan_y)
  plan_x, plan_y = plan_x[valid_plan], plan_y[valid_plan]
  if plan_x.shape[0] < 4 or np.any(np.diff(plan_x) <= 0.0):
    raise ValueError("E2E position path is not a finite increasing path")

  fit = (x >= LANE_LOCK_FIT_START) & (x <= LANE_LOCK_E2E_FALLBACK_END)
  if np.count_nonzero(fit) < 3:
    raise ValueError("lane horizon has too few E2E blend samples")
  if plan_x[0] > x[fit][0] or plan_x[-1] < x[fit][-1]:
    raise ValueError("E2E position path does not cover the blend horizon")

  fit = (x >= LANE_LOCK_FIT_START) & (x <= LANE_LOCK_FIT_END) & (x <= plan_x[-1])
  fit_x = x[fit]
  e2e_y = np.interp(fit_x, plan_x, plan_y)
  center_vehicle_y = center_y[fit] + CAMERA_OFFSET
  if not np.all(np.isfinite(center_vehicle_y - e2e_y)):
    raise ValueError("lane/E2E blend path is not finite")

  heading, offset = np.polyfit(fit_x, center_vehicle_y - e2e_y, 1)
  aligned_e2e = e2e_y + offset + heading * fit_x
  gain = get_e2e_blend_gain(fit_x)
  target = aligned_e2e + gain * (center_vehicle_y - aligned_e2e)
  lookahead = float(np.clip(2.0 * v_ego, LANE_LOCK_MIN_LOOKAHEAD, LANE_LOCK_MAX_LOOKAHEAD))
  field = gain * (target - e2e_y - (1.0 - LANE_LOCK_HEADING_GAIN) * heading * fit_x)
  weights = np.exp(-0.5 * ((fit_x - lookahead) / (LANE_LOCK_PATH_WEIGHT_SPREAD * lookahead)) ** 2)
  weights *= np.gradient(fit_x) * fit_x ** 2
  if np.dot(weights, gain) <= 0.0:
    raise ValueError("E2E blend weights are degenerate")

  error = float(get_e2e_blend_gain(lookahead) * np.dot(weights, field) / np.dot(weights, gain))
  correction = 2.0 * error / (lookahead * lookahead)
  return float(np.clip(correction, -LANE_LOCK_MAX_CENTER_CORRECTION, LANE_LOCK_MAX_CENTER_CORRECTION))


def apply_lane_lock(model_output: dict[str, np.ndarray], e2e_curvature: float, v_ego: float,
                    blinkers_active: bool = False, lane_policy_enabled: bool = False,
                    one_line_fallback_enabled: bool = True,
                    lead_fallback_enabled: bool = False,
                    close_lead_enabled: bool = False,
                    e2e_blend_enabled: bool = False,
                    two_line_enabled: bool = True,
                    intensity: int = LANE_POLICY_INTENSITY_NORMAL,
                    turn_centering_enabled: bool = False,
                    turn_centering_strength: int = LANE_POLICY_INTENSITY_NORMAL,
                    constants: type = ModelConstants) -> float:
  """Add a bounded lane-centering correction to the model curvature."""
  global _lane_lock_weight, _lane_lock_lane_curvature
  global _lane_lock_has_lane_curvature, _lane_lock_full_active
  global _lane_lock_ready, _lane_lock_arm_time
  global _lane_lock_width, _lane_lock_width_valid, _lane_lock_line_loss_time
  global _lane_lock_center_correction, _lane_lock_has_center_correction
  global _lane_lock_one_line_hold, _lane_lock_error_logged
  global _lane_lock_last_turn_sign, _lane_lock_turn_release_time
  global _lane_lock_last_curve_target_sign, _lane_lock_curve_reversal_hold_time
  global _lane_policy_mode, _lane_policy_correction

  if not lane_policy_enabled:
    return release_lane_lock(e2e_curvature)

  if blinkers_active:
    return release_lane_lock(e2e_curvature, "stock-e2e fallback: blinker")

  try:
    left_y = model_output['lane_lines'][0, 1, :, 0].astype(np.float64)
    right_y = model_output['lane_lines'][0, 2, :, 0].astype(np.float64)
    left_prob, right_prob = get_inner_lane_line_probs(model_output)
    desire_state = model_output['desire_state'][0]
    lane_change_prob = float(desire_state[log.Desire.laneChangeLeft] +
                             desire_state[log.Desire.laneChangeRight])
    if lane_change_prob > LANE_LOCK_MAX_LANE_CHANGE_PROB:
      return release_lane_lock(e2e_curvature, "stock-e2e fallback: lane-change intent")

    x = np.asarray(constants.X_IDXS, dtype=np.float64)
    fit = (x >= LANE_LOCK_FIT_START) & (x <= LANE_LOCK_FIT_END)
    if x.shape != left_y.shape or np.count_nonzero(fit) < 3:
      raise ValueError(f"lane-line horizon {left_y.shape} does not match {constants.__name__}.X_IDXS {x.shape}")

    valid_left = (np.isfinite(left_prob) and left_prob >= LANE_LOCK_RETAIN_LINE_PROB and
                  np.all(np.isfinite(left_y[fit])))
    valid_right = (np.isfinite(right_prob) and right_prob >= LANE_LOCK_RETAIN_LINE_PROB and
                   np.all(np.isfinite(right_y[fit])))
    turn_centering_active = (turn_centering_enabled and
                             abs(e2e_curvature) >= TURN_CENTERING_CURVATURE)
    two_line_geometry = False
    two_line_confidence = min(left_prob, right_prob)
    center_y: np.ndarray | None = None

    if (two_line_enabled or turn_centering_active) and valid_left and valid_right:
      measured_width, two_line_geometry = get_lane_width_measurement(left_y, right_y, fit)
      if two_line_geometry:
        center_y = 0.5 * (left_y + right_y)
        _lane_lock_line_loss_time = 0.0
        _lane_lock_one_line_hold = False

        if two_line_confidence >= LANE_LOCK_ARM_LINE_PROB:
          if not _lane_lock_width_valid:
            _lane_lock_width = measured_width
            _lane_lock_width_valid = True
          else:
            alpha = min(DT_MDL / LANE_LOCK_WIDTH_TIME, 1.0)
            _lane_lock_width += alpha * (measured_width - _lane_lock_width)

    if center_y is None and one_line_fallback_enabled and _lane_lock_full_active and _lane_lock_width_valid:
      selected_left = valid_left and (not valid_right or not two_line_geometry or left_prob >= right_prob)
      selected_right = valid_right and not selected_left
      if selected_left or selected_right:
        _lane_lock_line_loss_time += DT_MDL
        if _lane_lock_line_loss_time <= LANE_LOCK_ONE_LINE_HOLD_TIME:
          center_y = (left_y + _lane_lock_width / 2.0 if selected_left else
                      right_y - _lane_lock_width / 2.0)
          _lane_lock_one_line_hold = True

    lead_center_correction = None
    if center_y is None:
      if lead_fallback_enabled:
        lead_center_correction = get_lead_center_correction(model_output, v_ego, close_lead_enabled)
      if lead_center_correction is None:
        return release_lane_lock(e2e_curvature, "stock-e2e fallback: lane geometry or confidence")

    if turn_centering_active and not two_line_geometry:
      return release_lane_lock(e2e_curvature, "stock-e2e fallback: turn lane geometry")

    if center_y is not None and not _lane_lock_full_active:
      if two_line_geometry and two_line_confidence >= LANE_LOCK_ARM_LINE_PROB:
        _lane_lock_arm_time = min(_lane_lock_arm_time + DT_MDL, LANE_LOCK_ARM_TIME)
        _lane_lock_ready = _lane_lock_arm_time >= LANE_LOCK_ARM_TIME
      else:
        _lane_lock_arm_time = 0.0

      if not _lane_lock_ready:
        _lane_lock_has_lane_curvature = False
        _lane_lock_one_line_hold = False
        _lane_policy_mode = LANE_POLICY_MODE_INACTIVE
        log_lane_lock_mode("stock-e2e fallback: arming lane confidence")
        return ramp_out_correction(e2e_curvature)

      _lane_lock_full_active = True
      _lane_lock_weight = 1.0
      _lane_lock_line_loss_time = 0.0

    correction_limit = LANE_LOCK_MAX_CENTER_CORRECTION
    if lead_center_correction is not None:
      center_correction = lead_center_correction
      correction_limit = LANE_LOCK_LEAD_MAX_CORRECTION
      _lane_lock_one_line_hold = False
      _lane_lock_ready = False
      _lane_lock_full_active = False
      _lane_lock_arm_time = 0.0
    else:
      if e2e_blend_enabled and not turn_centering_active and not _lane_lock_one_line_hold:
        center_correction = get_e2e_blended_center_correction(model_output, center_y, x, v_ego)
      else:
        _, heading, offset = np.polyfit(x[fit], center_y[fit], 2)
        offset += CAMERA_OFFSET
        lookahead = float(np.clip(2.0 * v_ego, LANE_LOCK_MIN_LOOKAHEAD, LANE_LOCK_MAX_LOOKAHEAD))
        heading_scale = min(1.0, LANE_LOCK_HEADING_CURVATURE_FADE / max(abs(e2e_curvature), 1e-6))
        center_correction = (2.0 * (LANE_LOCK_HEADING_GAIN * heading_scale * heading * lookahead + offset) /
                             (lookahead * lookahead))
        center_correction = float(np.clip(center_correction,
                                          -LANE_LOCK_MAX_CENTER_CORRECTION,
                                          LANE_LOCK_MAX_CENTER_CORRECTION))

    intensity_scale = get_intensity_scale(intensity)
    if turn_centering_active:
      center_correction *= get_intensity_scale(turn_centering_strength)
    center_correction = float(np.clip(center_correction * intensity_scale,
                                      -correction_limit, correction_limit))
    if abs(center_correction) < LANE_LOCK_CORRECTION_DEADBAND:
      center_correction = 0.0

    target_sign = 1 if center_correction > 0.0 else -1 if center_correction < 0.0 else 0
    if abs(e2e_curvature) >= LANE_LOCK_HEADING_CURVATURE_FADE:
      if (target_sign and _lane_lock_last_curve_target_sign and
          target_sign != _lane_lock_last_curve_target_sign):
        _lane_lock_curve_reversal_hold_time = LANE_LOCK_CURVE_REVERSAL_HOLD_TIME
      if target_sign:
        _lane_lock_last_curve_target_sign = target_sign
    else:
      _lane_lock_last_curve_target_sign = 0

    if abs(e2e_curvature) >= LANE_LOCK_TURN_CURVATURE:
      e2e_turn_sign = 1 if e2e_curvature > 0.0 else -1
      if _lane_lock_last_turn_sign and e2e_turn_sign != _lane_lock_last_turn_sign:
        _lane_lock_turn_release_time = LANE_LOCK_TURN_RELEASE_TIME
      _lane_lock_last_turn_sign = e2e_turn_sign
    if _lane_lock_turn_release_time > 0.0:
      _lane_lock_turn_release_time = max(0.0, _lane_lock_turn_release_time - DT_MDL)
      if center_correction * e2e_curvature < 0.0:
        center_correction = 0.0

    if _lane_lock_curve_reversal_hold_time > 0.0:
      _lane_lock_curve_reversal_hold_time = max(0.0, _lane_lock_curve_reversal_hold_time - DT_MDL)
      center_correction = 0.0

    if not _lane_lock_has_center_correction:
      _lane_lock_center_correction = 0.0
      _lane_lock_has_center_correction = True

    correction_step = (LANE_LOCK_CORRECTION_RELEASE_STEP
                       if (abs(center_correction) < abs(_lane_lock_center_correction) or
                           center_correction * _lane_lock_center_correction < 0.0)
                       else LANE_LOCK_CORRECTION_ENGAGE_STEP * intensity_scale)
    delta = float(np.clip(center_correction - _lane_lock_center_correction,
                          -correction_step, correction_step))
    _lane_lock_center_correction += delta

    _lane_lock_lane_curvature = float(e2e_curvature + _lane_lock_center_correction)
    _lane_lock_has_lane_curvature = True
    _lane_lock_weight = 1.0 if lead_center_correction is None else 0.0
    if lead_center_correction is None:
      _lane_lock_full_active = True
      _lane_lock_ready = True
    _lane_policy_mode = (LANE_POLICY_MODE_LEAD if lead_center_correction is not None else
                         LANE_POLICY_MODE_ONE_LINE if _lane_lock_one_line_hold else
                         LANE_POLICY_MODE_TWO_LINE)
    _lane_policy_correction = float(_lane_lock_center_correction)
    _lane_lock_error_logged = False
    log_lane_lock_mode("lead vehicle fallback" if lead_center_correction is not None else
                       "lane center hold" if _lane_lock_one_line_hold else "full lane center")
    return _lane_lock_lane_curvature

  except (KeyError, IndexError, TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as err:
    if not _lane_lock_error_logged:
      cloudlog.warning(f"lane-policy input error: {type(err).__name__}: {err}")
      _lane_lock_error_logged = True
    return release_lane_lock(e2e_curvature, "stock-e2e fallback: lane-policy input error")
