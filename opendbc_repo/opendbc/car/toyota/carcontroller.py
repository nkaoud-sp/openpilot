import math
import numpy as np
from opendbc.car import Bus, make_tester_present_msg, rate_limit, structs, ACCELERATION_DUE_TO_GRAVITY, DT_CTRL
from opendbc.car.can_definitions import CanData
from opendbc.car.lateral import apply_meas_steer_torque_limits, apply_std_steer_angle_limits, common_fault_avoidance
from opendbc.car.carlog import carlog
from opendbc.car.common.filter_simple import FirstOrderFilter, HighPassFilter
from opendbc.car.common.pid import PIDController
from opendbc.car.secoc import add_mac, build_sync_mac
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.toyota import toyotacan
from opendbc.car.toyota.values import CAR, NO_STOP_TIMER_CAR, TSS2_CAR, \
                                        CarControllerParams, ToyotaFlags
from opendbc.can import CANPacker

from opendbc.sunnypilot.car.toyota.gas_interceptor import GasInterceptorCarController
from opendbc.sunnypilot.car.toyota.values import ToyotaFlagsSP

Ecu = structs.CarParams.Ecu
LongCtrlState = structs.CarControl.Actuators.LongControlState
SteerControlType = structs.CarParams.SteerControlType
VisualAlert = structs.CarControl.HUDControl.VisualAlert

# The up limit allows the brakes/gas to unwind quickly leaving a stop,
# the down limit roughly matches the rate of ACCEL_NET, reducing PCM compensation windup
ACCEL_WINDUP_LIMIT = 4.0 * DT_CTRL * 3  # m/s^2 / frame
ACCEL_WINDDOWN_LIMIT = -4.0 * DT_CTRL * 3  # m/s^2 / frame
ACCEL_PID_UNWIND = 0.03 * DT_CTRL * 3  # m/s^2 / frame

MAX_PITCH_COMPENSATION = 1.5  # m/s^2

# LKA limits
# EPS faults if you apply torque while the steering rate is above 100 deg/s for too long
MAX_STEER_RATE = 100  # deg/s
MAX_STEER_RATE_FRAMES = 17  # tx control frames needed before torque can be cut

# EPS allows user torque above threshold for 50 frames before permanently faulting
MAX_USER_TORQUE = 500

TOYOTA_TURN_SIGNAL_ADDR = 0x7C0
TOYOTA_TURN_SIGNAL_BUS = 0
TOYOTA_TURN_SIGNAL_PARAM = "ToyotaTurnSignalCommand"
TOYOTA_TURN_SIGNAL_BITS = {
  "left": 0x10,
  "right": 0x08,
  "hazard": 0x18,
}
TOYOTA_TURN_SIGNAL_OFF_MASK = 0x18
TOYOTA_EXTENDED_DIAGNOSTIC_SESSION = bytes.fromhex("1003")
TOYOTA_RETURN_TURN_SIGNAL_CONTROL = bytes.fromhex("2F291100")

# Onroad broadcast-candidate test: body frames captured from a 2020 Lexus ES350 that changed when the
# turn stalk was operated. These are below 0x600 so they cannot be injected via the offroad ELM327
# path; emit them here (Toyota safety, needs the addresses in the TX allowlist) to test whether the
# body ECU actually acts on any of them. Command string -> (addr, payload). ~10 Hz (frame % 10).
TOYOTA_LIGHTING_CANDIDATES = {
  "cand367": (0x367, b"\x08\x80"),
  "cand361": (0x361, bytes.fromhex("001700001600007f")),
  "cand2d8": (0x2D8, bytes.fromhex("00024000000a0000")),
}
TOYOTA_LIGHTING_CANDIDATE_RATE = 10

# Turn signal MITM test: replay the captured ES350 BLINKERS_STATE (0x614 = 29 80 8a <dir> 00 00 02 ce)
# with the direction nibble. Paired with the toyota.h fwd hook that drops the factory 0x614, so our
# frame is the only one the far side of the relay sees. dir: 0x10 left, 0x20 right, 0x38 hazard.
TOYOTA_BLINKER_STATE_ADDR = 0x614
TOYOTA_BLINKER_STATE_DIR = {"left": 0x10, "right": 0x20, "hazard": 0x38}


def toyota_blinker_state_frame(data3: int) -> bytes:
  return bytes([0x29, 0x80, 0x8A, data3, 0x00, 0x00, 0x02, 0xCE])


def get_long_tune(CP, params):
  if CP.flags & ToyotaFlags.TSS2:
    kiBP = [2., 5.]
    kiV = [0.5, 0.25]
  else:
    kiBP = [0., 5., 35.]
    kiV = [3.6, 2.4, 1.5]

  return PIDController(0.0, (kiBP, kiV), k_f=1.0,
                       pos_limit=params.ACCEL_MAX, neg_limit=params.ACCEL_MIN,
                       rate=1 / (DT_CTRL * 3))


def toyota_turn_signal_payload(command: str) -> bytes:
  state = bytearray(8)
  if command == "none":
    state[7] = TOYOTA_TURN_SIGNAL_OFF_MASK
  else:
    bits = TOYOTA_TURN_SIGNAL_BITS[command]
    state[3] = bits
    state[7] = bits
  return bytes.fromhex("2F291103") + bytes(state)


def toyota_turn_signal_isotp_frames(command: str) -> list[bytes]:
  payload = toyota_turn_signal_payload(command)
  return [
    bytes([0x10 | (len(payload) >> 8), len(payload) & 0xFF]) + payload[:6],
    (bytes([0x21]) + payload[6:]).ljust(8, b"\x00"),
  ]


def toyota_turn_signal_single_frame(payload: bytes) -> bytes:
  return (bytes([len(payload)]) + payload).ljust(8, b"\x00")


def toyota_turn_signal_sequence(command: str, was_active: bool) -> list[bytes]:
  frames = toyota_turn_signal_isotp_frames(command)
  if command == "none":
    return frames + [toyota_turn_signal_single_frame(TOYOTA_RETURN_TURN_SIGNAL_CONTROL)] if was_active else []
  return ([toyota_turn_signal_single_frame(TOYOTA_EXTENDED_DIAGNOSTIC_SESSION)] if not was_active else []) + frames


def get_turn_signal_command(CC_SP: structs.CarControlSP) -> str:
  command = str(CC_SP.turnSignalCommand)

  for param in CC_SP.params:
    if param.key == TOYOTA_TURN_SIGNAL_PARAM:
      command = param.value.decode(errors="ignore").strip().lower()
      break

  if command in ("off", "0"):
    return "none"
  if command in ("1", "left"):
    return "left"
  if command in ("2", "right"):
    return "right"
  if command in ("3", "hazard", "hazards"):
    return "hazard"
  return "none"


class CarController(CarControllerBase, GasInterceptorCarController):
  def __init__(self, dbc_names, CP, CP_SP):
    CarControllerBase.__init__(self, dbc_names, CP, CP_SP)
    GasInterceptorCarController.__init__(self, CP, CP_SP)
    self.params = CarControllerParams(self.CP)
    self.last_torque = 0
    self.last_angle = 0
    self.alert_active = False
    self.last_standstill = False
    self.standstill_req = False
    self.permit_braking = True
    self.steer_rate_counter = 0
    self.distance_button = 0

    # *** start long control state ***
    self.long_pid = get_long_tune(self.CP, self.params)
    self.aego = FirstOrderFilter(0.0, 0.25, DT_CTRL * 3)
    self.pitch = FirstOrderFilter(0, 0.5, DT_CTRL)
    self.pitch_hp = HighPassFilter(0.0, 0.25, 1.5, DT_CTRL)

    self.accel = 0
    self.prev_accel = 0
    # *** end long control state ***

    self.packer = CANPacker(dbc_names[Bus.pt])

    self.secoc_lka_message_counter = 0
    self.secoc_lta_message_counter = 0
    self.secoc_acc_message_counter = 0
    self.secoc_prev_reset_counter = 0
    self.last_turn_signal_command = "none"
    self.turn_signal_frames: list[bytes] = []

  def update(self, CC, CC_SP, CS, now_nanos):
    actuators = CC.actuators
    stopping = actuators.longControlState == LongCtrlState.stopping
    hud_control = CC.hudControl
    pcm_cancel_cmd = CC.cruiseControl.cancel
    lat_active = CC.latActive and abs(CS.out.steeringTorque) < MAX_USER_TORQUE

    if len(CC.orientationNED) == 3:
      self.pitch.update(CC.orientationNED[1])
      self.pitch_hp.update(CC.orientationNED[1])

    # *** control msgs ***
    can_sends = []

    turn_signal_command = get_turn_signal_command(CC_SP)
    if turn_signal_command != self.last_turn_signal_command:
      was_turn_signal_active = self.last_turn_signal_command != "none"
      self.last_turn_signal_command = turn_signal_command
      self.turn_signal_frames = toyota_turn_signal_sequence(turn_signal_command, was_turn_signal_active)

    if self.turn_signal_frames:
      can_sends.append(CanData(TOYOTA_TURN_SIGNAL_ADDR, self.turn_signal_frames.pop(0), TOYOTA_TURN_SIGNAL_BUS))

    # Onroad broadcast-candidate test: replay a captured body frame to probe whether the BCM acts on
    # it. get_turn_signal_command() returns "none" for these, so the 0x7C0 path above stays inactive.
    lighting_candidate = TOYOTA_LIGHTING_CANDIDATES.get(str(CC_SP.turnSignalCommand))
    if lighting_candidate is not None and self.frame % TOYOTA_LIGHTING_CANDIDATE_RATE == 0:
      cand_addr, cand_payload = lighting_candidate
      can_sends.append(CanData(cand_addr, cand_payload, TOYOTA_TURN_SIGNAL_BUS))

    # Turn signal MITM test: emit our own 0x614 for left/right/hazard (the factory 0x614 is dropped
    # by the fwd hook). turn_signal_command is already left/right/hazard/none from get_turn_signal_command.
    blinker_dir = TOYOTA_BLINKER_STATE_DIR.get(turn_signal_command)
    if blinker_dir is not None and self.frame % TOYOTA_LIGHTING_CANDIDATE_RATE == 0:
      can_sends.append(CanData(TOYOTA_BLINKER_STATE_ADDR, toyota_blinker_state_frame(blinker_dir), TOYOTA_TURN_SIGNAL_BUS))

    # *** handle secoc reset counter increase ***
    if self.CP.flags & ToyotaFlags.SECOC.value:
      if CS.secoc_synchronization['RESET_CNT'] != self.secoc_prev_reset_counter:
        self.secoc_lka_message_counter = 0
        self.secoc_lta_message_counter = 0
        self.secoc_acc_message_counter = 0
        self.secoc_prev_reset_counter = CS.secoc_synchronization['RESET_CNT']

        expected_mac = build_sync_mac(self.secoc_key, int(CS.secoc_synchronization['TRIP_CNT']), int(CS.secoc_synchronization['RESET_CNT']))
        if int(CS.secoc_synchronization['AUTHENTICATOR']) != expected_mac:
          carlog.error("SecOC synchronization MAC mismatch, wrong key?")

    # *** steer torque ***
    new_torque = int(round(actuators.torque * self.params.STEER_MAX))
    apply_torque = apply_meas_steer_torque_limits(new_torque, self.last_torque, CS.out.steeringTorqueEps, self.params)

    # >100 degree/sec steering fault prevention
    self.steer_rate_counter, apply_steer_req = common_fault_avoidance(abs(CS.out.steeringRateDeg) >= MAX_STEER_RATE, lat_active,
                                                                      self.steer_rate_counter, MAX_STEER_RATE_FRAMES)

    if not lat_active:
      apply_torque = 0

    # *** steer angle ***
    if self.CP.steerControlType == SteerControlType.angle:
      # If using LTA control, disable LKA and set steering angle command
      apply_torque = 0
      apply_steer_req = False
      if self.frame % 2 == 0:
        # EPS uses the torque sensor angle to control with, offset to compensate
        apply_angle = actuators.steeringAngleDeg + CS.out.steeringAngleOffsetDeg

        # Angular rate limit based on speed
        self.last_angle = apply_std_steer_angle_limits(apply_angle, self.last_angle, CS.out.vEgoRaw,
                                                       CS.out.steeringAngleDeg + CS.out.steeringAngleOffsetDeg,
                                                       CC.latActive, self.params.ANGLE_LIMITS)

    self.last_torque = apply_torque

    # toyota can trace shows STEERING_LKA at 42Hz, with counter adding alternatively 1 and 2;
    # sending it at 100Hz seem to allow a higher rate limit, as the rate limit seems imposed
    # on consecutive messages
    steer_command = toyotacan.create_steer_command(self.packer, apply_torque, apply_steer_req)
    if self.CP.flags & ToyotaFlags.SECOC.value:
      # TODO: check if this slow and needs to be done by the CANPacker
      steer_command = add_mac(self.secoc_key,
                              int(CS.secoc_synchronization['TRIP_CNT']),
                              int(CS.secoc_synchronization['RESET_CNT']),
                              self.secoc_lka_message_counter,
                              steer_command)
      self.secoc_lka_message_counter += 1
    can_sends.append(steer_command)

    # STEERING_LTA does not seem to allow more rate by sending faster, and may wind up easier
    if self.frame % 2 == 0 and self.CP.flags & ToyotaFlags.TSS2:
      lta_active = lat_active and self.CP.steerControlType == SteerControlType.angle
      # cut steering torque with TORQUE_WIND_DOWN when either EPS torque or driver torque is above
      # the threshold, to limit max lateral acceleration and for driver torque blending respectively.
      full_torque_condition = (abs(CS.out.steeringTorqueEps) < self.params.STEER_MAX and
                               abs(CS.out.steeringTorque) < self.params.MAX_LTA_DRIVER_TORQUE_ALLOWANCE)

      # TORQUE_WIND_DOWN at 0 ramps down torque at roughly the max down rate of 1500 units/sec
      torque_wind_down = 100 if lta_active and full_torque_condition else 0
      can_sends.append(toyotacan.create_lta_steer_command(self.packer, self.CP.steerControlType, self.last_angle,
                                                          lta_active, self.frame // 2, torque_wind_down))

      if self.CP.flags & ToyotaFlags.SECOC.value:
        lta_steer_2 = toyotacan.create_lta_steer_command_2(self.packer, self.frame // 2)
        lta_steer_2 = add_mac(self.secoc_key,
                              int(CS.secoc_synchronization['TRIP_CNT']),
                              int(CS.secoc_synchronization['RESET_CNT']),
                              self.secoc_lta_message_counter,
                              lta_steer_2)
        self.secoc_lta_message_counter += 1
        can_sends.append(lta_steer_2)

    # *** gas and brake ***

    # on entering standstill, send standstill request for older TSS-P cars that aren't designed to stay engaged at a stop
    if self.CP.carFingerprint not in NO_STOP_TIMER_CAR or self.CP_SP.enableGasInterceptor:
      if CS.out.standstill and not self.last_standstill and (self.CP_SP.enableGasInterceptor or not self.CP_SP.flags & ToyotaFlagsSP.STOP_AND_GO_HACK):
        self.standstill_req = True
      if CS.pcm_acc_status != 8:
        # pcm entered standstill or it's disabled
        self.standstill_req = False

    else:
      # if user engages at a stop with foot on brake, PCM starts in a special cruise standstill mode. on resume press,
      # brakes can take a while to ramp up causing a lurch forward. prevent resume press until planner wants to move.
      # don't use CC.cruiseControl.resume since it is gated on CS.cruiseState.standstill which goes false for 3s after resume press
      # whitelist hybrids as they do not have this issue and can stay stopped after resume press
      if not self.CP.flags & ToyotaFlags.HYBRID.value:
        should_resume = actuators.accel > 0
        if should_resume:
          self.standstill_req = False

        if not should_resume and CS.out.cruiseState.standstill:
          self.standstill_req = True

    self.last_standstill = CS.out.standstill

    # handle UI messages
    fcw_alert = hud_control.visualAlert == VisualAlert.fcw
    steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)
    lead = hud_control.leadVisible or CS.out.vEgo < 12.  # at low speed we always assume the lead is present so ACC can be engaged

    reverse_cruise = bool(self.CP_SP.flags & ToyotaFlagsSP.REVERSE_CRUISE)

    if self.CP.openpilotLongitudinalControl:
      if self.frame % 3 == 0:
        # Press distance button until we are at the correct bar length. Only change while enabled to avoid skipping startup popup
        if self.frame % 6 == 0 and self.CP.openpilotLongitudinalControl:
          desired_distance = 4 - hud_control.leadDistanceBars
          if CS.out.cruiseState.enabled and CS.pcm_follow_distance != desired_distance:
            self.distance_button = not self.distance_button
          else:
            self.distance_button = 0

        # internal PCM gas command can get stuck unwinding from negative accel so we apply a generous rate limit
        pcm_accel_cmd = actuators.accel
        if CC.longActive:
          pcm_accel_cmd = rate_limit(pcm_accel_cmd, self.prev_accel, ACCEL_WINDDOWN_LIMIT, ACCEL_WINDUP_LIMIT)
        self.prev_accel = pcm_accel_cmd

        # calculate amount of acceleration PCM should apply to reach target, given pitch.
        # clipped to only include downhill angles, avoids erroneously unsetting PERMIT_BRAKING when stopping on uphills
        accel_due_to_pitch = math.sin(min(self.pitch.x, 0.0)) * ACCELERATION_DUE_TO_GRAVITY
        # TODO: on uphills this sometimes sets PERMIT_BRAKING low not considering the creep force
        net_acceleration_request = pcm_accel_cmd + accel_due_to_pitch

        # GVC does not overshoot ego acceleration when starting from stop, but still has a similar delay
        if not self.CP.flags & ToyotaFlags.SECOC.value:
          a_ego_blended = float(np.interp(CS.out.vEgo, [1.0, 2.0], [CS.gvc, CS.out.aEgo]))
        else:
          a_ego_blended = CS.out.aEgo

        # wind down integral when approaching target for step changes and smooth ramps to reduce overshoot
        prev_aego = self.aego.x
        self.aego.update(a_ego_blended)
        j_ego = (self.aego.x - prev_aego) / (DT_CTRL * 3)

        future_t = float(np.interp(CS.out.vEgo, [2., 5.], [0.25, 0.5]))
        a_ego_future = a_ego_blended + j_ego * future_t

        if CC.longActive:
          # constantly slowly unwind integral to recover from large temporary errors
          self.long_pid.i -= ACCEL_PID_UNWIND * float(np.sign(self.long_pid.i))

          error_future = pcm_accel_cmd - a_ego_future

          if not stopping:
            # Toyota's PCM slowly responds to changes in pitch. On change, we amplify our
            # acceleration request to compensate for the undershoot and following overshoot
            pitch_compensation = float(np.clip(math.sin(self.pitch_hp.x) * ACCELERATION_DUE_TO_GRAVITY,
                                               -MAX_PITCH_COMPENSATION, MAX_PITCH_COMPENSATION))
            pcm_accel_cmd += pitch_compensation

          pcm_accel_cmd = self.long_pid.update(error_future,
                                               speed=CS.out.vEgo,
                                               feedforward=pcm_accel_cmd,
                                               freeze_integrator=actuators.longControlState != LongCtrlState.pid)
        else:
          self.long_pid.reset()

        # Along with rate limiting positive jerk above, this greatly improves gas response time
        # Consider the net acceleration request that the PCM should be applying (pitch included)
        net_acceleration_request_min = min(actuators.accel + accel_due_to_pitch, net_acceleration_request)
        if net_acceleration_request_min < 0.2 or stopping or not CC.longActive:
          self.permit_braking = True
        elif net_acceleration_request_min > 0.3:
          self.permit_braking = False

        pcm_accel_cmd = pcm_accel_cmd if self.CP.carFingerprint in TSS2_CAR else actuators.accel
        pcm_accel_cmd = float(np.clip(pcm_accel_cmd, self.params.ACCEL_MIN, self.params.ACCEL_MAX))

        main_accel_cmd = 0. if self.CP.flags & ToyotaFlags.SECOC.value else pcm_accel_cmd
        can_sends.append(toyotacan.create_accel_command(self.packer, main_accel_cmd, pcm_cancel_cmd, self.permit_braking, self.standstill_req, lead,
                                                        CS.acc_type, fcw_alert, self.distance_button, reverse_cruise))
        if self.CP.flags & ToyotaFlags.SECOC.value:
          acc_cmd_2 = toyotacan.create_accel_command_2(self.packer, pcm_accel_cmd)
          acc_cmd_2 = add_mac(self.secoc_key,
                              int(CS.secoc_synchronization['TRIP_CNT']),
                              int(CS.secoc_synchronization['RESET_CNT']),
                              self.secoc_acc_message_counter,
                              acc_cmd_2)
          self.secoc_acc_message_counter += 1
          can_sends.append(acc_cmd_2)

        self.accel = pcm_accel_cmd

    else:
      # we can spam can to cancel the system even if we are using lat only control
      if pcm_cancel_cmd:
        if self.CP.flags & ToyotaFlags.UNSUPPORTED_DSU:
          can_sends.append(toyotacan.create_acc_cancel_command(self.packer))
        else:
          can_sends.append(toyotacan.create_accel_command(self.packer, 0, pcm_cancel_cmd, True, False, lead, CS.acc_type, False, self.distance_button, reverse_cruise))

    can_sends.extend(GasInterceptorCarController.create_gas_command(self, CC, CS, actuators, self.packer, self.frame))

    # *** hud ui ***
    if self.CP.carFingerprint != CAR.TOYOTA_PRIUS_V:
      # ui mesg is at 1Hz but we send asap if:
      # - there is something to display
      # - there is something to stop displaying
      send_ui = False
      if ((fcw_alert or steer_alert) and not self.alert_active) or \
         (not (fcw_alert or steer_alert) and self.alert_active):
        send_ui = True
        self.alert_active = not self.alert_active
      elif pcm_cancel_cmd:
        # forcing the pcm to disengage causes a bad fault sound so play a good sound instead
        send_ui = True

      if self.frame % 20 == 0 or send_ui:
        can_sends.append(toyotacan.create_ui_command(self.packer, steer_alert, pcm_cancel_cmd, hud_control.leftLaneVisible,
                                                     hud_control.rightLaneVisible, hud_control.leftLaneDepart,
                                                     hud_control.rightLaneDepart, CC.latActive, CS.lkas_hud))

      if (self.frame % 100 == 0 or send_ui) and self.CP.flags & ToyotaFlags.DISABLE_RADAR.value:
        can_sends.append(toyotacan.create_fcw_command(self.packer, fcw_alert))

    # keep radar disabled
    if self.frame % 20 == 0 and self.CP.flags & ToyotaFlags.DISABLE_RADAR.value:
      can_sends.append(make_tester_present_msg(0x750, 0, 0xF))

    new_actuators = actuators.as_builder()
    new_actuators.torque = apply_torque / self.params.STEER_MAX
    new_actuators.torqueOutputCan = apply_torque
    new_actuators.steeringAngleDeg = self.last_angle
    new_actuators.accel = self.accel
    new_actuators.gas = self.gas

    self.frame += 1
    return new_actuators, can_sends
