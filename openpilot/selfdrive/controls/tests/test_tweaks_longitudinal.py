from types import SimpleNamespace

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_park_assist_offset
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  LAUNCH_DONE, LAUNCH_LAUNCHING, LAUNCH_READY, LongitudinalPlanner,
)


def build_sm(v_ego=0.0, lead_v=0.0, d_rel=5.0, lead_present=True, brake=False, gas=False):
  return {
    'carState': SimpleNamespace(vEgo=v_ego, brakePressed=brake, gasPressed=gas),
    'radarState': SimpleNamespace(leadOne=SimpleNamespace(present=lead_present, vLead=lead_v, dRel=d_rel)),
  }


def build_planner(enabled=True, eagerness=7, state=LAUNCH_READY):
  planner = LongitudinalPlanner.__new__(LongitudinalPlanner)
  planner.launch_assist = enabled
  planner.launch_eagerness = eagerness
  planner.launch_state = state
  return planner


class TestLaunchAssist(OpenpilotTestCase):
  def test_eagerness_threshold(self):
    self.assertFalse(build_planner(eagerness=1).launch_assist_ready(build_sm(lead_v=1.0)))

    planner = build_planner(eagerness=10)
    self.assertTrue(planner.launch_assist_ready(build_sm(lead_v=0.3)))
    self.assertEqual(planner.launch_state, LAUNCH_LAUNCHING)

  def test_driver_input_blocks_launch(self):
    self.assertFalse(build_planner(eagerness=10).launch_assist_ready(build_sm(lead_v=1.0, brake=True)))
    self.assertFalse(build_planner(eagerness=10).launch_assist_ready(build_sm(lead_v=1.0, gas=True)))

  def test_launch_lifecycle(self):
    planner = build_planner(eagerness=10)
    self.assertTrue(planner.launch_assist_ready(build_sm(lead_v=0.3)))
    self.assertFalse(planner.launch_assist_ready(build_sm(v_ego=0.6, lead_v=0.3)))
    self.assertEqual(planner.launch_state, LAUNCH_DONE)

    self.assertFalse(planner.launch_assist_ready(build_sm(lead_v=0.0)))
    self.assertEqual(planner.launch_state, LAUNCH_READY)

  def test_disabled_resets_state(self):
    planner = build_planner(enabled=False, state=LAUNCH_DONE)
    self.assertFalse(planner.launch_assist_ready(build_sm(lead_v=2.0)))
    self.assertEqual(planner.launch_state, LAUNCH_READY)


class TestHaltAssist(OpenpilotTestCase):
  def test_full_offset_at_low_speed(self):
    self.assertAlmostEqual(get_park_assist_offset(0.0, 2.8), 3.2)
    self.assertAlmostEqual(get_park_assist_offset(2.0, 2.8), 3.2)

  def test_offset_fades_with_speed(self):
    self.assertAlmostEqual(get_park_assist_offset(3.0, 2.8), 1.6)
    self.assertAlmostEqual(get_park_assist_offset(4.0, 2.8), 0.0)

  def test_invalid_or_default_gap_has_no_offset(self):
    self.assertAlmostEqual(get_park_assist_offset(0.0, 6.0), 0.0)
    self.assertAlmostEqual(get_park_assist_offset(0.0, 7.0), 0.0)
