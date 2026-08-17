"""Parity tests for the StarPilot-provider navigation overlay.

These exercise the policy directly so generic Sunnypilot lane-change settings
cannot obscure the source-equivalent StarPilot navigation rules.
"""
from types import SimpleNamespace

from cereal import log

from openpilot.common.constants import CV
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper


def _car_state(**overrides):
  values = {
    "vEgo": 5.0,
    "leftBlinker": False,
    "rightBlinker": False,
    "leftBlindspot": False,
    "rightBlindspot": False,
    "steeringPressed": False,
    "steeringTorque": 0.0,
    "standstill": False,
  }
  values.update(overrides)
  return SimpleNamespace(**values)


def _helper(**overrides):
  helper = DesireHelper()
  helper.nav_steer_enabled = True
  helper.nav_starpilot_lane_positioning = True
  helper.nav_starpilot_lane_detection_width = 3.0
  helper.nav_starpilot_min_lane_change_speed = 20.0 * CV.MPH_TO_MS
  helper.starpilot_lane_width_left = 4.0
  helper.starpilot_lane_width_right = 4.0
  for name, value in overrides.items():
    setattr(helper, name, value)
  return helper


def _state(**overrides):
  values = {
    "valid": True,
    "maneuverType": "turn",
    "maneuverModifier": "right",
    "maneuverDistance": 25.0,
    "laneCount": 0,
    "activeLaneDirection": "",
    "activeLaneAtRoadEdge": False,
    "hasSharedSameSideLane": False,
    "sameSideLaneCount": 0,
  }
  values.update(overrides)
  return values


def test_starpilot_turn_desire_uses_source_distance_and_speed_gates():
  helper = _helper()
  car = _car_state(vEgo=5.0)
  assert helper._starpilot_navigation_desire(car, True, _state()) == log.Desire.turnRight

  # At 5 m/s the exact StarPilot turn window is 25 m; a farther turn must not
  # request a model desire.
  assert helper._starpilot_navigation_desire(car, True, _state(maneuverDistance=25.1)) == log.Desire.none
  assert helper._starpilot_navigation_desire(
    _car_state(vEgo=20.0), True, _state(),
  ) == log.Desire.none


def test_starpilot_turn_does_not_require_driver_torque_but_respects_bsm_and_blinker():
  helper = _helper()
  state = _state(maneuverModifier="left", maneuverDistance=10.0)
  assert helper._starpilot_navigation_desire(_car_state(), True, state) == log.Desire.turnLeft
  assert helper._starpilot_navigation_desire(_car_state(rightBlinker=True), True, state) == log.Desire.none
  assert helper._starpilot_navigation_desire(_car_state(leftBlindspot=True), True, state) == log.Desire.none


def test_starpilot_slight_guidance_requires_width_and_matching_driver_nudge_only():
  state = _state(maneuverType="off ramp", maneuverModifier="right", maneuverDistance=50.0,
                 activeLaneDirection="slightRight")
  helper = _helper()
  assert helper._starpilot_navigation_desire(
    _car_state(vEgo=10.0, steeringPressed=True, steeringTorque=-1.0), True, state,
  ) == log.Desire.keepRight
  assert helper._starpilot_navigation_desire(
    _car_state(vEgo=10.0, steeringPressed=True, steeringTorque=1.0), True, state,
  ) == log.Desire.none

  helper.starpilot_lane_width_right = 2.9
  assert helper._starpilot_navigation_desire(
    _car_state(vEgo=10.0, steeringPressed=True, steeringTorque=-1.0), True, state,
  ) == log.Desire.none


def test_starpilot_fork_uses_ambiguous_split_and_edge_lane_suppression():
  helper = _helper()
  car = _car_state(vEgo=22.5, steeringPressed=True, steeringTorque=-1.0)
  common = {
    "maneuverType": "fork",
    "maneuverModifier": "right",
    "activeLaneDirection": "slightRight",
    "sameSideLaneCount": 3,
  }
  # 0.6 * 125 m at this speed: 60 m is allowed, 120 m is not.
  assert helper._starpilot_navigation_desire(car, True, _state(**common, maneuverDistance=60.0)) == log.Desire.keepRight
  assert helper._starpilot_navigation_desire(car, True, _state(**common, maneuverDistance=120.0)) == log.Desire.none
  edge_case = common | {
    "laneCount": 3,
    "sameSideLaneCount": 2,
    "activeLaneAtRoadEdge": True,
    "hasSharedSameSideLane": True,
    "maneuverDistance": 10.0,
  }
  assert helper._starpilot_navigation_desire(
    car, True, _state(**edge_case),
  ) == log.Desire.none


def test_starpilot_lane_width_matches_source_road_edge_rule():
  def line(y):
    return SimpleNamespace(x=[0.0, 10.0], y=[y, y])

  assert DesireHelper._starpilot_lane_width(line(4.0), line(0.0), line(6.0)) == 4.0
  assert DesireHelper._starpilot_lane_width(line(4.0), line(0.0), line(2.0)) == 0.0


def test_malformed_starpilot_instruction_state_fails_neutral():
  helper = _helper()
  malformed = helper._parse_starpilot_instruction_state('{"valid": true, "sameSideLaneCount": "bad"}')
  assert helper._starpilot_navigation_desire(_car_state(), True, malformed) == log.Desire.none
  assert helper._parse_starpilot_instruction_state("not-json") == {}
