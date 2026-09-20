import unittest

import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.modeld import lane_policy as modeld
from openpilot.selfdrive.modeld.constants import ModelConstants


def make_model_output(left_prob: float = 0.99, right_prob: float = 0.99, lane_width: float = 3.6,
                      lane_width_end: float | None = None, lane_center: float = 0.0,
                      lane_heading: float = 0.0, lead_prob: float = 0.0,
                      lead_x: float = 35.0, lead_y: float = 0.0) -> dict[str, np.ndarray]:
  x = np.asarray(ModelConstants.X_IDXS, dtype=np.float64)
  lane_lines = np.zeros((1, 4, len(x), 2), dtype=np.float64)
  target_width = lane_width if lane_width_end is None else lane_width_end
  width_progress = np.clip((x - modeld.LANE_LOCK_FIT_START) /
                           (modeld.LANE_LOCK_FIT_END - modeld.LANE_LOCK_FIT_START), 0.0, 1.0)
  widths = lane_width + (target_width - lane_width) * width_progress
  centerline = lane_center + lane_heading * x
  lane_lines[0, 1, :, 0] = centerline - widths / 2.0
  lane_lines[0, 2, :, 0] = centerline + widths / 2.0
  lane_line_probs = np.zeros((1, 8), dtype=np.float64)
  lane_line_probs[0, 3] = left_prob
  lane_line_probs[0, 5] = right_prob
  plan = np.zeros((1, len(x), ModelConstants.PLAN_WIDTH), dtype=np.float64)
  plan[0, :, 0] = x
  lead = np.zeros((1, ModelConstants.LEAD_MHP_SELECTION, ModelConstants.LEAD_TRAJ_LEN, ModelConstants.LEAD_WIDTH), dtype=np.float64)
  lead[:, :, :, 0] = lead_x
  lead[:, :, :, 1] = lead_y
  lead_probs = np.zeros((1, ModelConstants.LEAD_MHP_SELECTION), dtype=np.float64)
  lead_probs[0, 0] = lead_prob
  return {'lane_lines': lane_lines, 'lane_lines_prob': lane_line_probs,
          'desire_state': np.zeros((1, ModelConstants.DESIRE_LEN), dtype=np.float64), 'plan': plan,
          'lead': lead, 'lead_prob': lead_probs}


class TestLanePolicy(unittest.TestCase):
  def setUp(self):
    modeld.reset_lane_lock()

  def apply_for(self, output: dict[str, np.ndarray], seconds: float, e2e_curvature: float = 0.0,
                blinkers_active: bool = False) -> float:
    result = e2e_curvature
    for _ in range(max(1, int(np.ceil(seconds / modeld.DT_MDL)))):
      result = modeld.apply_lane_lock(output, e2e_curvature, 20.0,
                                      blinkers_active=blinkers_active, lane_policy_enabled=True)
    return result

  def arm_lane_policy(self, output: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    output = make_model_output() if output is None else output
    self.apply_for(output, modeld.LANE_LOCK_ARM_TIME + modeld.DT_MDL)
    self.assertTrue(modeld._lane_lock_ready)
    self.assertTrue(modeld._lane_lock_full_active)
    self.assertTrue(modeld._lane_lock_width_valid)
    return output

  def test_disabled_mode_returns_exact_e2e_target(self):
    self.assertEqual(modeld.apply_lane_lock(make_model_output(), 0.0123, 20.0, lane_policy_enabled=False), 0.0123)

  def test_raw_probability_indices(self):
    output = make_model_output(0.97, 0.96)
    output['lane_lines_prob'][0, 1] = 0.01
    self.assertEqual(modeld.get_inner_lane_line_probs(output), (0.97, 0.96))

  def test_arms_only_after_clean_two_line_timer(self):
    output = make_model_output()
    self.apply_for(output, modeld.LANE_LOCK_ARM_TIME - modeld.DT_MDL)
    self.assertFalse(modeld._lane_lock_ready)
    self.assertEqual(modeld._lane_lock_weight, 0.0)
    self.apply_for(output, 2.0 * modeld.DT_MDL)
    self.assertTrue(modeld._lane_lock_full_active)
    self.assertEqual(modeld._lane_lock_weight, 1.0)

  def test_full_center_correction_keeps_e2e_curve_feedforward(self):
    self.arm_lane_policy()
    output = make_model_output(lane_center=0.45)
    e2e = 0.0010
    curvature = modeld.apply_lane_lock(output, e2e, 20.0, lane_policy_enabled=True)
    self.assertTrue(modeld._lane_lock_full_active)
    self.assertGreater(curvature, e2e)
    self.assertLessEqual(curvature - e2e, modeld.LANE_LOCK_MAX_CENTER_CORRECTION)

  def test_no_plan_gate_for_clean_lanes(self):
    output = make_model_output(lane_center=0.35)
    output['plan'][:] = np.nan
    self.arm_lane_policy(output)
    curvature = modeld.apply_lane_lock(output, 0.0, 20.0, lane_policy_enabled=True)
    self.assertGreater(curvature, 0.0)

  def test_below_exit_confidence_releases_to_exact_e2e(self):
    self.arm_lane_policy()
    low_confidence = make_model_output(left_prob=0.65, right_prob=0.65)
    self.assertEqual(modeld.apply_lane_lock(low_confidence, -0.0012, 20.0, lane_policy_enabled=True), -0.0012)
    self.assertFalse(modeld._lane_lock_full_active)

  def test_one_line_hold_uses_learned_width(self):
    self.arm_lane_policy()
    one_line = make_model_output(left_prob=0.99, right_prob=0.10, lane_center=0.35)
    curvature = self.apply_for(one_line, 0.50)
    self.assertTrue(modeld._lane_lock_full_active)
    self.assertTrue(modeld._lane_lock_one_line_hold)
    self.assertGreater(curvature, 0.0)

  def test_one_line_fallback_toggle_disables_hold(self):
    self.arm_lane_policy()
    one_line = make_model_output(left_prob=0.99, right_prob=0.10, lane_center=0.35)
    self.assertEqual(modeld.apply_lane_lock(one_line, -0.0012, 20.0, lane_policy_enabled=True,
                                            one_line_fallback_enabled=False), -0.0012)
    self.assertFalse(modeld._lane_lock_full_active)

  def test_lead_fallback_after_unclear_lanes(self):
    bad_lanes = make_model_output(left_prob=0.10, right_prob=0.10, lead_prob=0.90, lead_y=0.6)
    curvature = modeld.apply_lane_lock(bad_lanes, 0.0010, 20.0, lane_policy_enabled=True,
                                       one_line_fallback_enabled=False, lead_fallback_enabled=True)
    self.assertGreater(curvature, 0.0010)
    self.assertLessEqual(curvature - 0.0010, modeld.LANE_LOCK_LEAD_MAX_CORRECTION)
    self.assertFalse(modeld._lane_lock_full_active)

  def test_lead_fallback_toggle_off_returns_e2e(self):
    bad_lanes = make_model_output(left_prob=0.10, right_prob=0.10, lead_prob=0.90, lead_y=0.6)
    self.assertEqual(modeld.apply_lane_lock(bad_lanes, 0.0010, 20.0, lane_policy_enabled=True,
                                            one_line_fallback_enabled=False,
                                            lead_fallback_enabled=False), 0.0010)

  def test_one_line_fallback_takes_priority_over_lead(self):
    self.arm_lane_policy()
    one_line_with_lead = make_model_output(left_prob=0.99, right_prob=0.10, lane_center=0.35,
                                           lead_prob=0.90, lead_y=-0.6)
    curvature = self.apply_for(one_line_with_lead, 0.50)
    self.assertTrue(modeld._lane_lock_one_line_hold)
    self.assertGreater(curvature, 0.0)

  def test_blinker_releases_lane_lock(self):
    output = self.arm_lane_policy()
    self.assertEqual(modeld.apply_lane_lock(output, 0.0123, 20.0, blinkers_active=True, lane_policy_enabled=True), 0.0123)
    self.assertFalse(modeld._lane_lock_full_active)

  def test_lane_change_intent_releases_lane_lock(self):
    self.arm_lane_policy()
    output = make_model_output()
    output['desire_state'][0, log.Desire.laneChangeLeft] = 0.2
    self.assertEqual(modeld.apply_lane_lock(output, 0.0123, 20.0, lane_policy_enabled=True), 0.0123)
    self.assertFalse(modeld._lane_lock_full_active)

  def test_robust_geometry_accepts_normal_taper_and_rejects_extreme_taper(self):
    self.arm_lane_policy(make_model_output(lane_width=3.6, lane_width_end=4.1))
    self.assertTrue(modeld._lane_lock_full_active)

    modeld.reset_lane_lock()
    output = make_model_output(lane_width=3.6, lane_width_end=5.0)
    self.apply_for(output, modeld.LANE_LOCK_ARM_TIME + modeld.DT_MDL)
    self.assertFalse(modeld._lane_lock_full_active)


if __name__ == "__main__":
  unittest.main()
