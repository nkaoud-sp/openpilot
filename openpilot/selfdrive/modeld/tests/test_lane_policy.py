import unittest

import numpy as np

from openpilot.selfdrive.modeld import lane_policy
from openpilot.selfdrive.modeld.constants import ModelConstants


def make_model_output(left_prob=0.99, right_prob=0.99, lane_center=0.0,
                      lead_prob=0.0, lead_x=35.0, lead_y=0.0):
  x = np.asarray(ModelConstants.X_IDXS, dtype=np.float64)
  lane_lines = np.zeros((1, 4, len(x), 2), dtype=np.float64)
  lane_lines[0, 1, :, 0] = lane_center - 1.8
  lane_lines[0, 2, :, 0] = lane_center + 1.8
  lane_line_probs = np.zeros((1, 8), dtype=np.float64)
  lane_line_probs[0, 3] = left_prob
  lane_line_probs[0, 5] = right_prob
  plan = np.zeros((1, len(x), ModelConstants.PLAN_WIDTH), dtype=np.float64)
  plan[0, :, 0] = x
  lead = np.zeros((1, ModelConstants.LEAD_MHP_SELECTION,
                   ModelConstants.LEAD_TRAJ_LEN, ModelConstants.LEAD_WIDTH), dtype=np.float64)
  lead[:, :, :, 0] = lead_x
  lead[:, :, :, 1] = lead_y
  lead_probs = np.zeros((1, ModelConstants.LEAD_MHP_SELECTION), dtype=np.float64)
  lead_probs[0, 0] = lead_prob
  return {
    'lane_lines': lane_lines,
    'lane_lines_prob': lane_line_probs,
    'desire_state': np.zeros((1, ModelConstants.DESIRE_LEN), dtype=np.float64),
    'plan': plan,
    'lead': lead,
    'lead_prob': lead_probs,
  }


class TestLanePolicy(unittest.TestCase):
  def setUp(self):
    lane_policy.reset_lane_lock()

  def apply_for(self, output, seconds, **kwargs):
    result = 0.0
    for _ in range(max(1, int(np.ceil(seconds / lane_policy.DT_MDL)))):
      result = lane_policy.apply_lane_lock(output, 0.0, 20.0, lane_policy_enabled=True, **kwargs)
    return result

  def arm(self, output=None):
    output = make_model_output() if output is None else output
    self.apply_for(output, lane_policy.LANE_LOCK_ARM_TIME + lane_policy.DT_MDL)
    self.assertTrue(lane_policy._lane_lock_full_active)
    return output

  def test_disabled_returns_model_curvature(self):
    self.assertEqual(lane_policy.apply_lane_lock(make_model_output(), 0.0123, 20.0), 0.0123)

  def test_clean_lanes_arm_before_correcting(self):
    output = make_model_output(lane_center=0.35)
    self.apply_for(output, lane_policy.LANE_LOCK_ARM_TIME - lane_policy.DT_MDL)
    self.assertFalse(lane_policy._lane_lock_full_active)
    correction = self.apply_for(output, 2.0 * lane_policy.DT_MDL)
    self.assertTrue(lane_policy._lane_lock_full_active)
    self.assertGreater(correction, 0.0)
    self.assertLessEqual(correction, lane_policy.LANE_LOCK_MAX_CENTER_CORRECTION)

  def test_one_line_hold_uses_learned_width(self):
    self.arm()
    one_line = make_model_output(left_prob=0.99, right_prob=0.10, lane_center=0.35)
    self.assertGreater(self.apply_for(one_line, 0.5), 0.0)
    self.assertEqual(lane_policy.get_lane_policy_status()[0], lane_policy.LANE_POLICY_MODE_ONE_LINE)

  def test_lead_fallback_is_lower_authority(self):
    output = make_model_output(left_prob=0.1, right_prob=0.1, lead_prob=0.9, lead_y=0.6)
    correction = lane_policy.apply_lane_lock(output, 0.0, 20.0, lane_policy_enabled=True,
                                              one_line_fallback_enabled=False, lead_fallback_enabled=True)
    self.assertGreater(correction, 0.0)
    self.assertLessEqual(correction, lane_policy.LANE_LOCK_LEAD_MAX_CORRECTION)
    self.assertEqual(lane_policy.get_lane_policy_status()[0], lane_policy.LANE_POLICY_MODE_LEAD)

  def test_close_lead_requires_option(self):
    output = make_model_output(left_prob=0.1, right_prob=0.1, lead_prob=0.9, lead_x=4.0, lead_y=0.2)
    disabled = lane_policy.apply_lane_lock(output, 0.001, 20.0, lane_policy_enabled=True,
                                           lead_fallback_enabled=True)
    lane_policy.reset_lane_lock()
    enabled = lane_policy.apply_lane_lock(output, 0.001, 20.0, lane_policy_enabled=True,
                                          lead_fallback_enabled=True, close_lead_enabled=True)
    self.assertEqual(disabled, 0.001)
    self.assertGreater(enabled, 0.001)

  def test_intensity_scales_without_raising_cap(self):
    values = []
    for intensity in range(3):
      lane_policy.reset_lane_lock()
      self.arm()
      values.append(self.apply_for(make_model_output(lane_center=0.1), 1.0, intensity=intensity))
    self.assertLess(values[0], values[1])
    self.assertLess(values[1], values[2])
    self.assertLessEqual(values[2], lane_policy.LANE_LOCK_MAX_CENTER_CORRECTION)

  def test_turn_centering_uses_lane_midpoint_over_e2e_blend(self):
    output = make_model_output(lane_center=0.35)
    self.apply_for(output, lane_policy.LANE_LOCK_ARM_TIME + lane_policy.DT_MDL)
    centered = lane_policy.apply_lane_lock(output, 0.001, 20.0,
                                            lane_policy_enabled=True,
                                            e2e_blend_enabled=True,
                                            two_line_enabled=False,
                                            turn_centering_enabled=True)
    self.assertGreater(centered, 0.001)
    self.assertLessEqual(centered, 0.001 + lane_policy.LANE_LOCK_MAX_CENTER_CORRECTION)

  def test_turn_centering_strength_scales_firmness(self):
    values = []
    for strength in range(3):
      lane_policy.reset_lane_lock()
      output = make_model_output(lane_center=0.1)
      self.apply_for(output, lane_policy.LANE_LOCK_ARM_TIME + lane_policy.DT_MDL)
      result = 0.0
      for _ in range(100):
        result = lane_policy.apply_lane_lock(output, 0.001, 20.0, lane_policy_enabled=True,
                                             turn_centering_enabled=True,
                                             turn_centering_strength=strength)
      self.assertEqual(lane_policy.get_lane_policy_status()[0], lane_policy.LANE_POLICY_MODE_TWO_LINE)
      values.append(result - 0.001)
    self.assertLess(values[0], values[1])
    self.assertLess(values[1], values[2])
    self.assertLessEqual(values[2], lane_policy.LANE_LOCK_MAX_CENTER_CORRECTION)

  def sloped_output(self, slope, offset=0.1):
    x = np.asarray(ModelConstants.X_IDXS, dtype=np.float64)
    output = make_model_output()
    center = offset + slope * x
    output['lane_lines'][0, 1, :, 0] = center - 1.8
    output['lane_lines'][0, 2, :, 0] = center + 1.8
    return output

  def test_turn_centering_preserves_heading_authority_mid_turn(self):
    # A lane whose midpoint carries a heading error during a sharp turn: turn
    # centering keeps full heading authority while the general faded path does not,
    # so turn centering produces the larger corrective curvature.
    curvature = 0.0024  # well past the heading-fade knee
    results = {}
    for turn in (True, False):
      lane_policy.reset_lane_lock()
      output = self.sloped_output(slope=0.004)
      self.apply_for(output, lane_policy.LANE_LOCK_ARM_TIME + lane_policy.DT_MDL)
      result = 0.0
      for _ in range(200):
        result = lane_policy.apply_lane_lock(output, curvature, 20.0, lane_policy_enabled=True,
                                             turn_centering_enabled=turn)
      results[turn] = result - curvature
    self.assertGreater(results[True], results[False])
    self.assertLessEqual(results[True], lane_policy.LANE_LOCK_MAX_CENTER_CORRECTION)

  def test_turn_centering_releases_without_two_line_geometry(self):
    output = make_model_output(left_prob=0.99, right_prob=0.10, lane_center=0.35)
    result = lane_policy.apply_lane_lock(output, 0.001, 20.0,
                                         lane_policy_enabled=True,
                                         turn_centering_enabled=True)
    self.assertEqual(result, 0.001)

  def test_blinker_ramps_correction_out(self):
    output = self.arm(make_model_output(lane_center=0.45))
    self.apply_for(output, 0.5)
    previous = lane_policy._lane_lock_center_correction
    released = lane_policy.apply_lane_lock(output, 0.0, 20.0, blinkers_active=True, lane_policy_enabled=True)
    self.assertGreaterEqual(released, 0.0)
    self.assertLess(released, previous)
    self.assertFalse(lane_policy._lane_lock_full_active)


if __name__ == "__main__":
  unittest.main()
