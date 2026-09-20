import time
import logging

import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.modeld.constants import ModelConstants

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
LANE_LOCK_LEAD_MAX_DISTANCE = 60.0
LANE_LOCK_LEAD_MAX_LATERAL = 2.0
LANE_LOCK_LEAD_MAX_CORRECTION = 0.00020
LANE_POLICY_MODE_INACTIVE = 0
LANE_POLICY_MODE_TWO_LINE = 1
LANE_POLICY_MODE_ONE_LINE = 2
LANE_POLICY_MODE_LEAD = 3

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


def get_lane_policy_status() -> tuple[int, float]:
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


def get_lead_center_correction(model_output: dict[str, np.ndarray], v_ego: float) -> float | None:
  lead_prob = np.asarray(model_output['lead_prob'])
  leads = np.asarray(model_output['lead'])
  if lead_prob.ndim != 2 or leads.ndim != 4 or leads.shape[-1] < 2:
    raise ValueError(f"unexpected lead shapes: lead_prob={lead_prob.shape}, lead={leads.shape}")

  best_idx = int(np.argmax(lead_prob[0]))
  best_prob = float(lead_prob[0, best_idx])
  lead_x = float(leads[0, best_idx, 0, 0])
  lead_y = float(leads[0, best_idx, 0, 1])
  if (not np.isfinite(best_prob) or not np.isfinite(lead_x) or not np.isfinite(lead_y) or
      best_prob < LANE_LOCK_LEAD_MIN_PROB or
      lead_x < LANE_LOCK_LEAD_MIN_DISTANCE or lead_x > LANE_LOCK_LEAD_MAX_DISTANCE or
      abs(lead_y) > LANE_LOCK_LEAD_MAX_LATERAL):
    return None

  lookahead = float(np.clip(min(lead_x, 2.0 * v_ego), LANE_LOCK_MIN_LOOKAHEAD, LANE_LOCK_MAX_LOOKAHEAD))
  correction = 2.0 * lead_y / (lookahead * lookahead)
  return float(np.clip(correction, -LANE_LOCK_LEAD_MAX_CORRECTION, LANE_LOCK_LEAD_MAX_CORRECTION))


def apply_lane_lock(model_output: dict[str, np.ndarray], e2e_curvature: float, v_ego: float,
                    blinkers_active: bool = False, lane_policy_enabled: bool = False,
                    one_line_fallback_enabled: bool = True,
                    lead_fallback_enabled: bool = False) -> float:
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
    reset_lane_lock()
    return float(e2e_curvature)

  if blinkers_active:
    reset_lane_lock()
    log_lane_lock_mode("stock-e2e fallback: blinker")
    return float(e2e_curvature)

  try:
    left_y = model_output['lane_lines'][0, 1, :, 0].astype(np.float64)
    right_y = model_output['lane_lines'][0, 2, :, 0].astype(np.float64)
    left_prob, right_prob = get_inner_lane_line_probs(model_output)
    desire_state = model_output['desire_state'][0]
    lane_change_prob = float(desire_state[log.Desire.laneChangeLeft] +
                             desire_state[log.Desire.laneChangeRight])
    if lane_change_prob > LANE_LOCK_MAX_LANE_CHANGE_PROB:
      reset_lane_lock()
      log_lane_lock_mode("stock-e2e fallback: lane-change intent")
      return float(e2e_curvature)

    x = np.asarray(ModelConstants.X_IDXS, dtype=np.float64)
    fit = (x >= LANE_LOCK_FIT_START) & (x <= LANE_LOCK_FIT_END)
    if x.shape != left_y.shape or np.count_nonzero(fit) < 3:
      raise ValueError("lane-line horizon does not match ModelConstants.X_IDXS")

    valid_left = (np.isfinite(left_prob) and left_prob >= LANE_LOCK_RETAIN_LINE_PROB and
                  np.all(np.isfinite(left_y[fit])))
    valid_right = (np.isfinite(right_prob) and right_prob >= LANE_LOCK_RETAIN_LINE_PROB and
                   np.all(np.isfinite(right_y[fit])))
    two_line_geometry = False
    two_line_confidence = min(left_prob, right_prob)
    center_y: np.ndarray | None = None

    if valid_left and valid_right:
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
        lead_center_correction = get_lead_center_correction(model_output, v_ego)
      if lead_center_correction is None:
        reset_lane_lock()
        log_lane_lock_mode("stock-e2e fallback: lane geometry or confidence")
        return float(e2e_curvature)

    if center_y is not None and not _lane_lock_full_active:
      if two_line_geometry and two_line_confidence >= LANE_LOCK_ARM_LINE_PROB:
        _lane_lock_arm_time = min(_lane_lock_arm_time + DT_MDL, LANE_LOCK_ARM_TIME)
        _lane_lock_ready = _lane_lock_arm_time >= LANE_LOCK_ARM_TIME
      else:
        _lane_lock_arm_time = 0.0

      if not _lane_lock_ready:
        _lane_lock_has_lane_curvature = False
        _lane_lock_one_line_hold = False
        log_lane_lock_mode("stock-e2e fallback: arming lane confidence")
        return float(e2e_curvature)

      _lane_lock_full_active = True
      _lane_lock_weight = 1.0
      _lane_lock_line_loss_time = 0.0

    if lead_center_correction is not None:
      center_correction = lead_center_correction
      _lane_lock_one_line_hold = False
      _lane_lock_ready = False
      _lane_lock_full_active = False
    else:
      _, heading, offset = np.polyfit(x[fit], center_y[fit], 2)
      lookahead = float(np.clip(2.0 * v_ego, LANE_LOCK_MIN_LOOKAHEAD, LANE_LOCK_MAX_LOOKAHEAD))
      heading_scale = min(1.0, LANE_LOCK_HEADING_CURVATURE_FADE / max(abs(e2e_curvature), 1e-6))
      center_correction = (2.0 * (LANE_LOCK_HEADING_GAIN * heading_scale * heading * lookahead + offset) /
                           (lookahead * lookahead))
      center_correction = float(np.clip(center_correction,
                                        -LANE_LOCK_MAX_CENTER_CORRECTION,
                                        LANE_LOCK_MAX_CENTER_CORRECTION))
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
                       else LANE_LOCK_CORRECTION_ENGAGE_STEP)
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
    reset_lane_lock()
    if not _lane_lock_error_logged:
      cloudlog.warning(f"lane-policy input error: {type(err).__name__}: {err}")
      _lane_lock_error_logged = True
    log_lane_lock_mode("stock-e2e fallback: lane-policy input error")
    return float(e2e_curvature)
