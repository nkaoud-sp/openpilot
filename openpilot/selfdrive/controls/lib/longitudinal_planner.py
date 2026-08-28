#!/usr/bin/env python3
import math
import numpy as np

import openpilot.cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import DYNAMIC_T_FOLLOW_MIN, DYNAMIC_T_FOLLOW_MAX, DYNAMIC_T_FOLLOW_CURVE
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import PARK_MODE_ALL_LOW_SPEED, STOP_DISTANCE
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan, should_stop
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
J_CRUISE_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MIN = -1.2
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5

# Lead-departure launch assist: when stopped behind a lead that pulls away,
# defer to the radar-based MPC so the car launches sooner. The MPC still owns
# the safe gap and the assist never overrides brake or gas.
LAUNCH_MAX_EGO_SPEED = 0.5
LAUNCH_MIN_DREL = 2.0
LAUNCH_VLEAD_BP = [1, 10]
LAUNCH_VLEAD_V = [1.5, 0.2]
LAUNCH_READY, LAUNCH_LAUNCHING, LAUNCH_DONE = 0, 1, 2

# Experimental Speed Assist: a small, heavily gated acceleration nudge for
# clean experimental open-road cases where e2e is far below cruise.
SPEED_ASSIST_OFF, SPEED_ASSIST_READOUT, SPEED_ASSIST_ON = 0, 1, 2
SPEED_ASSIST_MAX_BOOSTS = [0.15, 0.25, 0.35]
SPEED_ASSIST_FULL_GAP_KPH = 30.0
SPEED_ASSIST_MODEL_PLAN_GAP_MAX = 0.10
SPEED_ASSIST_GENTLE_ACCEL_MAX = 0.20
SPEED_ASSIST_DECEL_BLOCK = -0.15
SPEED_ASSIST_MODEL_LEAD_PROB_MAX = 0.35
SPEED_ASSIST_BRAKE_PROB_MAX = 0.06
SPEED_ASSIST_RC = 1.5

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py

def get_cruise_accel(e2e, v_cruise, v_ego, a_cruise_prev, angle_steers, CP, dt, accel_coast, allow_throttle):
  max_accel = ACCEL_MAX if e2e else get_max_accel(v_ego)

  if not e2e:
    a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
    a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
    a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))
    max_accel = min(max_accel, a_x_allowed)
    if not allow_throttle:
      clipped_accel_coast = max(accel_coast, ACCEL_MIN)
      coast_limit = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [max_accel, clipped_accel_coast])
      max_accel = min(max_accel, coast_limit)

  target_accel = np.clip(v_cruise - v_ego, A_CRUISE_MIN, max_accel)
  j_cruise = np.interp(v_ego, A_CRUISE_MAX_BP, J_CRUISE_VALS)
  target_accel = float(np.clip(target_accel, a_cruise_prev - j_cruise * dt, a_cruise_prev + j_cruise * dt))

  return target_accel


class LongitudinalPlanner(LongitudinalPlannerSP):
  def __init__(self, CP, CP_SP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    LongitudinalPlannerSP.__init__(self, self.CP, CP_SP, self.mpc)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.a_cruise = init_a
    self.output_a_target = init_a
    self.output_should_stop = False
    self.params = Params()
    self.param_read_frame = 0
    self.launch_assist = False
    self.launch_eagerness = 10
    self.launch_assist_active = False
    self.launch_assist_latched = False
    self.launch_state = LAUNCH_READY
    self.dynamic_follow = False
    self.dynamic_follow_min = DYNAMIC_T_FOLLOW_MIN
    self.dynamic_follow_max = DYNAMIC_T_FOLLOW_MAX
    self.dynamic_follow_curve = DYNAMIC_T_FOLLOW_CURVE
    self.park_assist = False
    self.park_distance = STOP_DISTANCE
    self.speed_assist_mode = SPEED_ASSIST_OFF
    self.speed_assist_strength = 1
    self.speed_assist_min_kph = 50
    self.speed_assist_max_kph = 130
    self.speed_assist_start_gap_kph = 8
    self.speed_assist_lead_mode = 0
    self.speed_assist_boost_filter = FirstOrderFilter(0.0, SPEED_ASSIST_RC, self.dt)
    self.speed_assist_enabled = False
    self.speed_assist_readout_only = False
    self.speed_assist_eligible = False
    self.speed_assist_active = False
    self.speed_assist_a_target_original = 0.0
    self.speed_assist_a_boost = 0.0
    self.speed_assist_speed_gap_kph = 0.0
    self.speed_assist_reason = "off"

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  def read_tweaks_params(self):
    if self.param_read_frame % int(1. / self.dt) == 0:
      self.launch_assist = self.params.get_bool("LaunchAssist")
      self.launch_eagerness = self.params.get("LaunchEagerness", return_default=True)
      self.dynamic_follow = self.params.get_bool("DynamicFollow")
      self.dynamic_follow_min = self.params.get("DynamicFollowMinTime", return_default=True) / 100.0
      self.dynamic_follow_max = self.params.get("DynamicFollowMaxTime", return_default=True) / 100.0
      self.dynamic_follow_curve = self.params.get("DynamicFollowCurve", return_default=True) / 100.0
      self.park_assist = self.params.get_bool("ParkAssist")
      self.park_distance = self.params.get("ParkDistance", return_default=True) / 100.0
      self.speed_assist_mode = self.params.get("ExperimentalSpeedAssistMode", return_default=True)
      self.speed_assist_strength = self.params.get("ExperimentalSpeedAssistStrength", return_default=True)
      self.speed_assist_min_kph = self.params.get("ExperimentalSpeedAssistMinKph", return_default=True)
      self.speed_assist_max_kph = self.params.get("ExperimentalSpeedAssistMaxKph", return_default=True)
      self.speed_assist_start_gap_kph = self.params.get("ExperimentalSpeedAssistStartGapKph", return_default=True)
      self.speed_assist_lead_mode = self.params.get("ExperimentalSpeedAssistLeadMode", return_default=True)
    self.param_read_frame += 1

  @staticmethod
  def _list_value(values, idx: int, default: float = 0.0) -> float:
    return float(values[idx]) if len(values) > idx else default

  def _model_lead_prob(self, model_msg) -> float:
    if len(model_msg.leadsV3) == 0:
      return 0.0
    return max(float(lead.prob) for lead in model_msg.leadsV3)

  def _speed_assist_desired_boost(self, sm, output_a_target: float, output_a_target_e2e: float) -> float:
    self.speed_assist_enabled = self.speed_assist_mode != SPEED_ASSIST_OFF
    self.speed_assist_readout_only = self.speed_assist_mode == SPEED_ASSIST_READOUT
    self.speed_assist_eligible = False
    self.speed_assist_active = False
    self.speed_assist_a_target_original = float(output_a_target)
    self.speed_assist_speed_gap_kph = 0.0

    if not self.speed_assist_enabled:
      self.speed_assist_reason = "off"
      return 0.0

    CS = sm['carState']
    if not sm['selfdriveState'].experimentalMode:
      self.speed_assist_reason = "not experimental"
      return 0.0
    if not sm['carControl'].enabled:
      self.speed_assist_reason = "not enabled"
      return 0.0
    if self.mpc.source != LongitudinalPlanSource.e2e:
      self.speed_assist_reason = "not e2e"
      return 0.0
    if CS.gasPressed or CS.brakePressed:
      self.speed_assist_reason = "driver"
      return 0.0
    if CS.vCruise == V_CRUISE_UNSET or CS.vCruise >= V_CRUISE_UNSET:
      self.speed_assist_reason = "no cruise"
      return 0.0

    v_ego_kph = CS.vEgo * CV.MS_TO_KPH
    min_kph = min(self.speed_assist_min_kph, self.speed_assist_max_kph)
    max_kph = max(self.speed_assist_min_kph, self.speed_assist_max_kph)
    if v_ego_kph < min_kph or v_ego_kph > max_kph:
      self.speed_assist_reason = "speed"
      return 0.0

    self.speed_assist_speed_gap_kph = float(CS.vCruise - v_ego_kph)
    if self.speed_assist_speed_gap_kph < self.speed_assist_start_gap_kph:
      self.speed_assist_reason = "gap"
      return 0.0

    if self.speed_assist_lead_mode == 0 and (sm['radarState'].leadOne.present or self._model_lead_prob(sm['modelV2']) > SPEED_ASSIST_MODEL_LEAD_PROB_MAX):
      self.speed_assist_reason = "lead"
      return 0.0
    if sm['modelV2'].action.shouldStop:
      self.speed_assist_reason = "model stop"
      return 0.0
    if self.output_should_stop:
      self.speed_assist_reason = "plan stop"
      return 0.0

    preds = sm['modelV2'].meta.disengagePredictions
    brake_press = self._list_value(preds.brakePressProbs, 0)
    brake_disengage = self._list_value(preds.brakeDisengageProbs, 0)
    if brake_press > SPEED_ASSIST_BRAKE_PROB_MAX or brake_disengage > SPEED_ASSIST_BRAKE_PROB_MAX:
      self.speed_assist_reason = "brake risk"
      return 0.0

    if output_a_target < SPEED_ASSIST_DECEL_BLOCK:
      self.speed_assist_reason = "decel"
      return 0.0
    if abs(output_a_target_e2e - output_a_target) > SPEED_ASSIST_MODEL_PLAN_GAP_MAX:
      self.speed_assist_reason = "mpc low" if output_a_target_e2e > output_a_target else "e2e low"
      return 0.0
    if output_a_target_e2e > SPEED_ASSIST_GENTLE_ACCEL_MAX or output_a_target > SPEED_ASSIST_GENTLE_ACCEL_MAX:
      self.speed_assist_reason = "already accel"
      return 0.0

    max_boost = SPEED_ASSIST_MAX_BOOSTS[int(np.clip(self.speed_assist_strength, 0, len(SPEED_ASSIST_MAX_BOOSTS) - 1))]
    full_gap_kph = max(float(self.speed_assist_start_gap_kph) + 1.0, SPEED_ASSIST_FULL_GAP_KPH)
    boost = float(np.interp(self.speed_assist_speed_gap_kph,
                            [self.speed_assist_start_gap_kph, full_gap_kph],
                            [0.05, max_boost]))
    self.speed_assist_eligible = True
    self.speed_assist_reason = "active" if self.speed_assist_mode == SPEED_ASSIST_ON else "readout"
    return boost

  def launch_assist_ready(self, sm) -> bool:
    if not self.launch_assist:
      self.launch_state = LAUNCH_READY
      return False

    CS = sm['carState']
    lead = sm['radarState'].leadOne
    stopped = CS.vEgo <= LAUNCH_MAX_EGO_SPEED
    v_thresh = float(np.interp(self.launch_eagerness, LAUNCH_VLEAD_BP, LAUNCH_VLEAD_V))
    lead_departing = lead.present and lead.vLead >= v_thresh

    if self.launch_state == LAUNCH_DONE and stopped and not lead_departing:
      self.launch_state = LAUNCH_READY

    can_fire = (self.launch_state in (LAUNCH_READY, LAUNCH_LAUNCHING) and
                lead.present and stopped and not CS.brakePressed and not CS.gasPressed and
                lead.dRel >= LAUNCH_MIN_DREL and lead_departing)

    if can_fire:
      self.launch_state = LAUNCH_LAUNCHING
      return True

    if self.launch_state == LAUNCH_LAUNCHING:
      self.launch_state = LAUNCH_DONE if not stopped else LAUNCH_READY

    return False

  def update(self, sm):
    LongitudinalPlannerSP.update(self, sm)

    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    if sm['controlsState'].forceDecel:
      v_cruise = 0.0

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET
    reset_state = reset_state or not v_cruise_initialized

    throttle_probs = sm['modelV2'].meta.disengagePredictions.gasPressProbs
    throttle_prob = throttle_probs[1] if len(throttle_probs) > 1 else 1.0
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['vehicleParameters'].angleOffsetDeg

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.output_a_target = np.clip(sm['carState'].aEgo, ACCEL_MIN, ACCEL_MAX)
      self.a_cruise = self.output_a_target

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    # Get new v_cruise and a_target from Smart Cruise Control and Speed Limit Assist
    v_cruise, self.output_a_target = LongitudinalPlannerSP.update_targets(self, sm, self.v_desired_filter.x, self.output_a_target, v_cruise)

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.output_a_target)
    self.read_tweaks_params()
    self.mpc.update(sm['radarState'], personality=sm['selfdriveState'].personality,
                    dynamic_follow=self.dynamic_follow,
                    t_follow_min=self.dynamic_follow_min, t_follow_max=self.dynamic_follow_max,
                    t_follow_curve=self.dynamic_follow_curve,
                    park_assist=self.park_assist, park_distance=self.park_distance,
                    park_mode=PARK_MODE_ALL_LOW_SPEED)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Save starting point for next iteration
    a_prev = self.output_a_target

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                              action_t=action_t)
    output_should_stop_mpc = should_stop(v_ego, output_a_target_mpc)
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    is_e2e = self.is_e2e(sm)

    self.a_cruise = get_cruise_accel(is_e2e, v_cruise, v_ego,
                                     self.a_cruise, steer_angle_without_offset, self.CP, self.dt,
                                     accel_coast, self.allow_throttle)
    cruise_should_stop = should_stop(v_ego, self.a_cruise)

    candidates = [(output_a_target_mpc, self.mpc.source, output_should_stop_mpc),
                  (self.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop)]
    if is_e2e:
      candidates.append((output_a_target_e2e, LongitudinalPlanSource.e2e, output_should_stop_e2e))

    output_a_target, self.mpc.source, _ = min(candidates, key=lambda c: c[0])
    self.output_should_stop = any(should_stop for _, _, should_stop in candidates)

    self.launch_assist_active = self.launch_assist_ready(sm)
    self.launch_assist_latched = self.launch_state == LAUNCH_DONE
    if self.launch_assist_active:
      output_a_target = output_a_target_mpc
      self.output_should_stop = output_should_stop_mpc

    if self.mpc.park_assist_active and sm['carState'].vEgo <= LAUNCH_MAX_EGO_SPEED:
      output_a_target = output_a_target_mpc
      self.output_should_stop = output_should_stop_mpc

    speed_assist_boost = self._speed_assist_desired_boost(sm, output_a_target, output_a_target_e2e)
    self.speed_assist_a_boost = float(self.speed_assist_boost_filter.update(speed_assist_boost))
    self.speed_assist_active = (self.speed_assist_mode == SPEED_ASSIST_ON and
                                self.speed_assist_eligible and self.speed_assist_a_boost > 0.01)
    if self.speed_assist_active:
      output_a_target += self.speed_assist_a_boost

    self.output_a_target = np.clip(output_a_target, ACCEL_MIN, ACCEL_MAX)

    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.output_a_target + a_prev) / 2.0

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks()

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.present
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)

    self.publish_longitudinal_plan_sp(sm, pm)
