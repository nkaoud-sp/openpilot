from types import SimpleNamespace

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_park_assist_offset
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  LAUNCH_DONE, LAUNCH_LAUNCHING, LAUNCH_READY, SPEED_ASSIST_OFF, SPEED_ASSIST_ON, SPEED_ASSIST_READOUT, LongitudinalPlanner,
)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalPlanSource


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


def build_speed_assist_planner(mode=SPEED_ASSIST_ON, lead_mode=0, openpilot_longitudinal=True):
  planner = LongitudinalPlanner.__new__(LongitudinalPlanner)
  planner.CP = SimpleNamespace(openpilotLongitudinalControl=openpilot_longitudinal)
  planner.speed_assist_mode = mode
  planner.speed_assist_strength = 1
  planner.speed_assist_min_kph = 50
  planner.speed_assist_max_kph = 130
  planner.speed_assist_start_gap_kph = 8
  planner.speed_assist_lead_mode = lead_mode
  planner.speed_assist_enabled = False
  planner.speed_assist_readout_only = False
  planner.speed_assist_eligible = False
  planner.speed_assist_active = False
  planner.speed_assist_a_target_original = 0.0
  planner.speed_assist_speed_gap_kph = 0.0
  planner.speed_assist_reason = "off"
  planner.output_should_stop = False
  planner.mpc = SimpleNamespace(source=LongitudinalPlanSource.e2e)
  return planner


def build_speed_assist_sm(*, enabled=True, lead=False, brake=False, gas=False):
  disengage_predictions = SimpleNamespace(brakePressProbs=[0.0], brakeDisengageProbs=[0.0])
  model = SimpleNamespace(
    action=SimpleNamespace(shouldStop=False),
    leadsV3=[],
    meta=SimpleNamespace(disengagePredictions=disengage_predictions),
  )
  return {
    'carState': SimpleNamespace(vEgo=20.0, vCruise=100.0, brakePressed=brake, gasPressed=gas),
    'carControl': SimpleNamespace(enabled=enabled),
    'selfdriveState': SimpleNamespace(),
    'radarState': SimpleNamespace(leadOne=SimpleNamespace(present=lead)),
    'modelV2': model,
  }


class TestSpeedAssist(OpenpilotTestCase):
  def test_off_mode_has_no_boost(self):
    planner = build_speed_assist_planner(mode=SPEED_ASSIST_OFF)
    self.assertEqual(planner._speed_assist_desired_boost(build_speed_assist_sm(), 0.0, 0.0), 0.0)
    self.assertEqual(planner.speed_assist_reason, "off")

  def test_readout_reports_eligibility_without_selecting_on_mode(self):
    planner = build_speed_assist_planner(mode=SPEED_ASSIST_READOUT)
    boost = planner._speed_assist_desired_boost(build_speed_assist_sm(), 0.0, 0.0)
    self.assertGreater(boost, 0.0)
    self.assertTrue(planner.speed_assist_eligible)
    self.assertTrue(planner.speed_assist_readout_only)
    self.assertEqual(planner.speed_assist_reason, "readout")

  def test_lead_and_driver_input_block_boost(self):
    planner = build_speed_assist_planner()
    self.assertEqual(planner._speed_assist_desired_boost(build_speed_assist_sm(lead=True), 0.0, 0.0), 0.0)
    self.assertEqual(planner.speed_assist_reason, "lead")

    self.assertEqual(planner._speed_assist_desired_boost(build_speed_assist_sm(brake=True), 0.0, 0.0), 0.0)
    self.assertEqual(planner.speed_assist_reason, "driver")

  def test_requires_openpilot_longitudinal_and_e2e_agreement(self):
    planner = build_speed_assist_planner(openpilot_longitudinal=False)
    self.assertEqual(planner._speed_assist_desired_boost(build_speed_assist_sm(), 0.0, 0.0), 0.0)
    self.assertEqual(planner.speed_assist_reason, "stock longitudinal")

    planner = build_speed_assist_planner()
    planner.mpc.source = LongitudinalPlanSource.cruise
    self.assertEqual(planner._speed_assist_desired_boost(build_speed_assist_sm(), 0.0, 0.0), 0.0)
    self.assertEqual(planner.speed_assist_reason, "not e2e")

    planner.mpc.source = LongitudinalPlanSource.e2e
    self.assertEqual(planner._speed_assist_desired_boost(build_speed_assist_sm(), 0.0, 0.2), 0.0)
    self.assertEqual(planner.speed_assist_reason, "mpc low")
