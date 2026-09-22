import unittest

import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.modeld import lane_policy as modeld
from openpilot.selfdrive.modeld.constants import ModelConstants


def make_model_output(left_prob: float = 0.99, right_prob: float = 0.99, lane_width: float = 3.6,
                      lane_width_end: float | None = None, lane_center: float = 0.0,
                      lane_heading: float = 0.0, lead_prob: float = 0.0,
                      lead_x: float = 35.0, lead_y: float = 0.0,
                      plan_y: np.ndarray | None = None) -> dict[str, np.ndarray]:
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
  if plan_y is not None:
    plan[0, :, 1] = plan_y
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

  def test_e2e_blend_toggle_changes_only_two_line_correction(self):
    x = np.asarray(ModelConstants.X_IDXS, dtype=np.float64)
    output = make_model_output(lane_center=0.35, plan_y=np.full_like(x, 0.35 + modeld.CAMERA_OFFSET))
    self.arm_lane_policy(output)
    legacy = modeld.apply_lane_lock(output, 0.0, 20.0, lane_policy_enabled=True,
                                    e2e_blend_enabled=False)

    modeld.reset_lane_lock()
    self.arm_lane_policy(output)
    blended = modeld.apply_lane_lock(output, 0.0, 20.0, lane_policy_enabled=True,
                                     e2e_blend_enabled=True)

    self.assertNotEqual(blended, legacy)
    self.assertEqual(modeld.get_lane_policy_status()[0], modeld.LANE_POLICY_MODE_TWO_LINE)

  def test_e2e_blend_bad_plan_releases_instead_of_using_bad_anchor(self):
    output = make_model_output(lane_center=0.35)
    output['plan'][:] = np.nan
    self.arm_lane_policy(make_model_output(lane_center=-modeld.CAMERA_OFFSET))
    self.assertEqual(modeld.apply_lane_lock(output, 0.0010, 20.0, lane_policy_enabled=True,
                                            e2e_blend_enabled=True), 0.0010)

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

  def engage_max_correction(self) -> dict[str, np.ndarray]:
    self.arm_lane_policy()
    offset = make_model_output(lane_center=0.45)
    self.apply_for(offset, 0.5)
    self.assertAlmostEqual(modeld._lane_lock_center_correction, modeld.LANE_LOCK_MAX_CENTER_CORRECTION)
    return offset

  def test_release_ramps_correction_out_instead_of_stepping(self):
    offset = self.engage_max_correction()

    # e2e is 0.0 here, so the returned curvature is exactly the applied
    # correction. It must walk down at the release rate, not drop in one frame.
    previous = modeld._lane_lock_center_correction
    first = modeld.apply_lane_lock(offset, 0.0, 20.0, blinkers_active=True, lane_policy_enabled=True)
    self.assertGreater(first, 0.0)
    self.assertLessEqual(previous - first, modeld.LANE_LOCK_CORRECTION_RELEASE_STEP + 1e-12)
    self.assertFalse(modeld._lane_lock_full_active)

    previous = first
    for _ in range(5):
      curvature = modeld.apply_lane_lock(offset, 0.0, 20.0, blinkers_active=True, lane_policy_enabled=True)
      self.assertLessEqual(previous - curvature, modeld.LANE_LOCK_CORRECTION_RELEASE_STEP + 1e-12)
      previous = curvature
    self.assertEqual(previous, 0.0)

  def test_disabling_policy_mid_drive_ramps_out(self):
    offset = self.engage_max_correction()
    first = modeld.apply_lane_lock(offset, 0.0, 20.0, lane_policy_enabled=False)
    self.assertGreater(first, 0.0)
    curvature = first
    for _ in range(5):
      curvature = modeld.apply_lane_lock(offset, 0.0, 20.0, lane_policy_enabled=False)
    self.assertEqual(curvature, 0.0)

  def test_release_ramp_converges_to_exact_e2e(self):
    offset = self.engage_max_correction()
    for _ in range(6):
      curvature = modeld.apply_lane_lock(offset, -0.0012, 20.0, blinkers_active=True, lane_policy_enabled=True)
    self.assertEqual(curvature, -0.0012)

  def test_lead_reference_uses_the_present_lead_not_the_likeliest_horizon(self):
    # lead_prob is indexed by time offset, so a confident 4 s lead must not
    # override a t=0 hypothesis the gates would otherwise reject.
    bad_lanes = make_model_output(left_prob=0.10, right_prob=0.10)
    bad_lanes['lead_prob'][0, 0] = 0.10
    bad_lanes['lead_prob'][0, 2] = 0.99
    bad_lanes['lead'][0, 2, 0, 1] = 1.5
    self.assertEqual(modeld.apply_lane_lock(bad_lanes, 0.0010, 20.0, lane_policy_enabled=True,
                                            one_line_fallback_enabled=False,
                                            lead_fallback_enabled=True), 0.0010)

  def test_centers_the_car_not_the_camera(self):
    # The car's centerline sits at y = -CAMERA_OFFSET in the model frame, so a
    # lane midpoint sitting exactly there means the car is already centered and
    # nothing should be commanded.
    from openpilot.selfdrive.controls.lib.ldw import CAMERA_OFFSET
    self.arm_lane_policy(make_model_output(lane_center=-CAMERA_OFFSET))
    curvature = modeld.apply_lane_lock(make_model_output(lane_center=-CAMERA_OFFSET), 0.0, 20.0,
                                       lane_policy_enabled=True)
    self.assertEqual(curvature, 0.0)

    # Centering the camera instead leaves the car CAMERA_OFFSET to the left of
    # the lane, so a midpoint at y=0 must command a correction back to the right.
    modeld.reset_lane_lock()
    self.arm_lane_policy()
    curvature = modeld.apply_lane_lock(make_model_output(), 0.0, 20.0, lane_policy_enabled=True)
    self.assertGreater(curvature, 0.0)

  def test_lead_gate_rejects_adjacent_lane_lead_up_close(self):
    # 1.2 m off the nose at 10 m is ~7 degrees: the next lane over, and inside
    # the old flat 2.0 m gate.
    near = make_model_output(left_prob=0.10, right_prob=0.10, lead_prob=0.90,
                             lead_x=10.0, lead_y=1.2)
    self.assertEqual(modeld.apply_lane_lock(near, 0.0010, 20.0, lane_policy_enabled=True,
                                            one_line_fallback_enabled=False,
                                            lead_fallback_enabled=True), 0.0010)

    # The same lateral offset at 40 m is within the corridor and is accepted.
    modeld.reset_lane_lock()
    far = make_model_output(left_prob=0.10, right_prob=0.10, lead_prob=0.90,
                            lead_x=40.0, lead_y=1.2)
    self.assertGreater(modeld.apply_lane_lock(far, 0.0010, 20.0, lane_policy_enabled=True,
                                              one_line_fallback_enabled=False,
                                              lead_fallback_enabled=True), 0.0010)

  def test_lead_fallback_exit_still_requires_the_arm_timer(self):
    self.arm_lane_policy()
    lead_only = make_model_output(left_prob=0.10, right_prob=0.10, lead_prob=0.90, lead_x=40.0, lead_y=0.6)
    modeld.apply_lane_lock(lead_only, 0.0, 20.0, lane_policy_enabled=True,
                           one_line_fallback_enabled=False, lead_fallback_enabled=True)
    self.assertFalse(modeld._lane_lock_full_active)
    self.assertEqual(modeld._lane_lock_arm_time, 0.0)

    # Good two-line geometry returns: the policy must spend the full arm time
    # re-qualifying rather than snapping back to engaged on the first frame.
    clean = make_model_output()
    modeld.apply_lane_lock(clean, 0.0, 20.0, lane_policy_enabled=True, lead_fallback_enabled=True)
    self.assertFalse(modeld._lane_lock_full_active)
    self.apply_for(clean, modeld.LANE_LOCK_ARM_TIME)
    self.assertTrue(modeld._lane_lock_full_active)

  def test_runner_constants_drive_the_horizon_check(self):
    class ShortHorizonConstants:
      X_IDXS = list(ModelConstants.X_IDXS[:-1])

    output = make_model_output(lane_center=0.45)
    # The bundle's horizon no longer matches the lane-line array, so the policy
    # has to fall back rather than fit against the wrong distances.
    self.assertEqual(modeld.apply_lane_lock(output, 0.0010, 20.0, lane_policy_enabled=True,
                                            constants=ShortHorizonConstants), 0.0010)

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
