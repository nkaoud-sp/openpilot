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


def make_curve_output(radius, lane_center, plan_radius=None, v_ego=12.0, right_prob=0.99):
  """A left turn as the model reports it: lateral y (right-positive) at fixed X_IDXS, with the
  e2e plan driving an arc from the car (by default parallel to the lane)."""
  x = np.asarray(ModelConstants.X_IDXS, dtype=np.float64)
  output = make_model_output(right_prob=right_prob)
  curve_center_y = lane_center - radius
  for idx, line_radius in ((1, radius - 1.8), (2, radius + 1.8)):
    xx = np.minimum(x, 0.999 * line_radius)
    output['lane_lines'][0, idx, :, 0] = curve_center_y + np.sqrt(line_radius * line_radius - xx * xx)
  plan_radius = radius if plan_radius is None else plan_radius
  arc = v_ego * np.asarray(ModelConstants.T_IDXS, dtype=np.float64)
  output['plan'][0, :, 0] = plan_radius * np.sin(arc / plan_radius)
  output['plan'][0, :, 1] = -plan_radius * (1.0 - np.cos(arc / plan_radius))
  return output


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
      self.assertEqual(lane_policy.get_lane_policy_status()[0], lane_policy.LANE_POLICY_MODE_TURN_CENTERING)
      values.append(result - 0.001)
    self.assertLess(values[0], values[1])
    self.assertLess(values[1], values[2])
    self.assertLessEqual(values[2], lane_policy.LANE_LOCK_MAX_CENTER_CORRECTION)

  def turn_for(self, output, curvature, frames, **kwargs):
    for _ in range(frames):
      result = lane_policy.apply_lane_lock(output, curvature, 12.0, lane_policy_enabled=True,
                                           turn_centering_enabled=True, **kwargs)
    return result - curvature

  def arm_on_straight(self, lane_center):
    for _ in range(20):
      lane_policy.apply_lane_lock(make_model_output(lane_center=lane_center), 0.0, 12.0,
                                  lane_policy_enabled=True, turn_centering_enabled=True)

  def test_turn_centering_stays_engaged_through_tight_curve(self):
    # A constant-width lane on a 70 m left turn looks ~2 m wider far ahead when sampled
    # at fixed distances; that used to fail the width check and drop turn centering.
    self.arm_on_straight(0.4)
    output = make_curve_output(70.0, 0.4)
    for _ in range(60):
      lane_policy.apply_lane_lock(output, -1.0 / 70.0, 12.0, lane_policy_enabled=True, turn_centering_enabled=True)
      self.assertEqual(lane_policy.get_lane_policy_status()[0], lane_policy.LANE_POLICY_MODE_TURN_CENTERING)

  def test_turn_centering_has_no_curve_bias(self):
    straight = []
    for lane_center in (0.0, 0.2, -0.2):
      lane_policy.reset_lane_lock()
      self.arm_on_straight(lane_center)
      straight.append(self.turn_for(make_model_output(lane_center=lane_center), 0.0, 60))
    for radius in (250.0, 100.0, 50.0):
      for lane_center, straight_correction in zip((0.0, 0.2, -0.2), straight, strict=True):
        lane_policy.reset_lane_lock()
        self.arm_on_straight(lane_center)
        correction = self.turn_for(make_curve_output(radius, lane_center), -1.0 / radius, 60)
        # same answer as the straight road: the turn's own curvature must not leak in
        self.assertAlmostEqual(correction, straight_correction, delta=0.6e-4, msg=f"R={radius} c0={lane_center}")

  def test_turn_centering_pulls_inside_hugging_plan_back_out(self):
    self.arm_on_straight(0.0)
    hugging = make_curve_output(100.0, 0.0, plan_radius=90.0)
    # positive steers right, toward the outside of this left turn and the lane midpoint
    self.assertGreater(self.turn_for(hugging, -1.0 / 90.0, 60), 2.0e-4)

  def test_turn_centering_rides_out_brief_geometry_glitch(self):
    self.arm_on_straight(0.4)
    output = make_curve_output(100.0, 0.4)
    self.turn_for(output, -0.01, 40)
    glitch = make_curve_output(100.0, 0.4, right_prob=0.1)
    self.turn_for(glitch, -0.01, 2)
    self.assertTrue(lane_policy._lane_lock_full_active)
    self.turn_for(output, -0.01, 1)
    self.assertEqual(lane_policy.get_lane_policy_status()[0], lane_policy.LANE_POLICY_MODE_TURN_CENTERING)

    frames = int(np.ceil(lane_policy.TURN_CENTERING_GEOMETRY_GRACE_TIME / lane_policy.DT_MDL)) + 2
    self.turn_for(glitch, -0.01, frames)
    self.assertFalse(lane_policy._lane_lock_full_active)

  def test_turn_centering_reports_distinct_mode(self):
    output = make_model_output(lane_center=0.35)
    self.apply_for(output, lane_policy.LANE_LOCK_ARM_TIME + lane_policy.DT_MDL)
    # Straight: ordinary two-line centering.
    self.apply_for(output, lane_policy.DT_MDL, turn_centering_enabled=True)
    self.assertEqual(lane_policy.get_lane_policy_status()[0], lane_policy.LANE_POLICY_MODE_TWO_LINE)
    # Mid-turn with turn centering on: distinct mode for the visual indicator.
    lane_policy.apply_lane_lock(output, 0.001, 20.0, lane_policy_enabled=True,
                                turn_centering_enabled=True)
    self.assertEqual(lane_policy.get_lane_policy_status()[0], lane_policy.LANE_POLICY_MODE_TURN_CENTERING)

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
