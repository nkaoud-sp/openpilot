"""
desire_helper.py — nav v2 integration patch.

Changes from the current desire_helper.py:
  1. Remove NAV_LANE_CHANGE_DIRS — navd v2 never sends laneChangeLeft/Right.
  2. Remove the nav_lc_dir path from the LaneChangeState machine.
     Nav no longer drives the lane-change state machine directly.
  3. Simplify the nav desire override at the bottom of update():
     Just apply keepLeft/keepRight/turnLeft/turnRight when not mid-lane-change.
     Driver conflict is already resolved in navd before the desire is published,
     so no additional check is needed here.
  4. keep_pulse_timer behaviour is unchanged: it suppresses keepLeft/keepRight
     on alternate cycles, which applies to nav-sourced keep desires too.
     This is intentional — it prevents the model from being biased every tick.

The full updated update() method is below. Replace the existing one.
"""

from cereal import log, custom
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.selfdrive.controls.lib.auto_lane_change import AutoLaneChangeController, AutoLaneChangeMode
from openpilot.sunnypilot.selfdrive.controls.lib.lane_turn_desire import LaneTurnController

LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection
TurnDirection = custom.ModelDataV2SP.TurnDirection

# Nav v2: only keep* and turn* desires come from nav. No laneChange*.
NAV_DESIRE_MAP = {
  "none":       log.Desire.none,
  "turnLeft":   log.Desire.turnLeft,
  "turnRight":  log.Desire.turnRight,
  "keepLeft":   log.Desire.keepLeft,
  "keepRight":  log.Desire.keepRight,
}

LANE_CHANGE_SPEED_MIN = 20 * CV.MPH_TO_MS
LANE_CHANGE_TIME_MAX = 10.

DESIRES = {
  LaneChangeDirection.none: {
    LaneChangeState.off: log.Desire.none,
    LaneChangeState.preLaneChange: log.Desire.none,
    LaneChangeState.laneChangeStarting: log.Desire.none,
    LaneChangeState.laneChangeFinishing: log.Desire.none,
  },
  LaneChangeDirection.left: {
    LaneChangeState.off: log.Desire.none,
    LaneChangeState.preLaneChange: log.Desire.none,
    LaneChangeState.laneChangeStarting: log.Desire.laneChangeLeft,
    LaneChangeState.laneChangeFinishing: log.Desire.laneChangeLeft,
  },
  LaneChangeDirection.right: {
    LaneChangeState.off: log.Desire.none,
    LaneChangeState.preLaneChange: log.Desire.none,
    LaneChangeState.laneChangeStarting: log.Desire.laneChangeRight,
    LaneChangeState.laneChangeFinishing: log.Desire.laneChangeRight,
  },
}

TURN_DESIRES = {
  TurnDirection.none: log.Desire.none,
  TurnDirection.turnLeft: log.Desire.turnLeft,
  TurnDirection.turnRight: log.Desire.turnRight,
}


class DesireHelper:
  def __init__(self):
    self.lane_change_state = LaneChangeState.off
    self.lane_change_direction = LaneChangeDirection.none
    self.lane_change_timer = 0.0
    self.lane_change_ll_prob = 1.0
    self.keep_pulse_timer = 0.0
    self.prev_one_blinker = False
    self.desire = log.Desire.none
    self.alc = AutoLaneChangeController(self)
    self.lane_turn_controller = LaneTurnController(self)
    self.lane_turn_direction = TurnDirection.none

  @staticmethod
  def get_lane_change_direction(CS):
    return LaneChangeDirection.left if CS.leftBlinker else LaneChangeDirection.right

  def update(self, carstate, lateral_active, lane_change_prob, nav_desire="none"):
    self.alc.update_params()
    self.lane_turn_controller.update_params()
    v_ego = carstate.vEgo
    one_blinker = carstate.leftBlinker != carstate.rightBlinker
    below_lane_change_speed = v_ego < LANE_CHANGE_SPEED_MIN

    # Lane turn controller (unchanged)
    self.lane_turn_controller.update_lane_turn(
      blindspot_left=carstate.leftBlindspot,
      blindspot_right=carstate.rightBlindspot,
      left_blinker=carstate.leftBlinker,
      right_blinker=carstate.rightBlinker,
      v_ego=v_ego,
    )
    self.lane_turn_direction = self.lane_turn_controller.get_turn_direction()

    # ---------------------------------------------------------------------------
    # Lane change state machine — driver-initiated only (nav no longer drives this)
    # ---------------------------------------------------------------------------
    if not lateral_active or self.lane_change_timer > LANE_CHANGE_TIME_MAX or self.alc.lane_change_set_timer == AutoLaneChangeMode.OFF:
      self.lane_change_state = LaneChangeState.off
      self.lane_change_direction = LaneChangeDirection.none

    else:
      if self.lane_change_state == LaneChangeState.off:
        # Enter preLaneChange only on a fresh blinker flick from the driver.
        if one_blinker and not self.prev_one_blinker and not below_lane_change_speed:
          self.lane_change_state = LaneChangeState.preLaneChange
          self.lane_change_ll_prob = 1.0
          self.lane_change_direction = self.get_lane_change_direction(carstate)

      elif self.lane_change_state == LaneChangeState.preLaneChange:
        self.lane_change_direction = self.get_lane_change_direction(carstate)

        torque_applied = (
          carstate.steeringPressed and (
            (carstate.steeringTorque > 0 and self.lane_change_direction == LaneChangeDirection.left) or
            (carstate.steeringTorque < 0 and self.lane_change_direction == LaneChangeDirection.right)
          )
        )
        blindspot_detected = (
          (carstate.leftBlindspot and self.lane_change_direction == LaneChangeDirection.left) or
          (carstate.rightBlindspot and self.lane_change_direction == LaneChangeDirection.right)
        )

        self.alc.update_lane_change(blindspot_detected, carstate.brakePressed)

        if not one_blinker or below_lane_change_speed:
          self.lane_change_state = LaneChangeState.off
          self.lane_change_direction = LaneChangeDirection.none
        elif (torque_applied or self.alc.auto_lane_change_allowed) and not blindspot_detected:
          self.lane_change_state = LaneChangeState.laneChangeStarting

      elif self.lane_change_state == LaneChangeState.laneChangeStarting:
        self.lane_change_ll_prob = max(self.lane_change_ll_prob - 2 * DT_MDL, 0.0)
        if lane_change_prob < 0.02 and self.lane_change_ll_prob < 0.01:
          self.lane_change_state = LaneChangeState.laneChangeFinishing

      elif self.lane_change_state == LaneChangeState.laneChangeFinishing:
        self.lane_change_ll_prob = min(self.lane_change_ll_prob + DT_MDL, 1.0)
        if self.lane_change_ll_prob > 0.99:
          self.lane_change_direction = LaneChangeDirection.none
          self.lane_change_state = LaneChangeState.preLaneChange if one_blinker else LaneChangeState.off

    if self.lane_change_state in (LaneChangeState.off, LaneChangeState.preLaneChange):
      self.lane_change_timer = 0.0
    else:
      self.lane_change_timer += DT_MDL

    self.prev_one_blinker = one_blinker

    # ---------------------------------------------------------------------------
    # Base desire: lane turn controller takes priority, then lane change machine
    # ---------------------------------------------------------------------------
    if self.lane_turn_direction != TurnDirection.none:
      self.desire = TURN_DESIRES[self.lane_turn_direction]
    else:
      self.desire = DESIRES[self.lane_change_direction][self.lane_change_state]

    # ---------------------------------------------------------------------------
    # Nav desire override (v2: keep* and turn* only, never laneChange*)
    #
    # Rules:
    #   - Only applied when NOT in an active lane change (laneChangeStarting /
    #     laneChangeFinishing). We don't interrupt a lane change mid-execution.
    #   - turnLeft/turnRight override unconditionally (safety — sharp maneuver
    #     needed NOW). Gated by navd on correct lane already being confirmed.
    #   - keepLeft/keepRight are applied in off and preLaneChange states.
    #     In preLaneChange, they steer the model during the pre-phase, but do
    #     not prevent the driver from completing their lane change via blinker.
    #   - Driver conflict is already filtered in navd before publishing, so
    #     no additional check is required here.
    # ---------------------------------------------------------------------------
    nav_name = str(nav_desire)
    nav_mapped = NAV_DESIRE_MAP.get(nav_name)
    active_lc = self.lane_change_state in (LaneChangeState.laneChangeStarting,
                                           LaneChangeState.laneChangeFinishing)

    if nav_mapped and nav_name != "none" and not active_lc:
      self.desire = nav_mapped

    # ---------------------------------------------------------------------------
    # Keep pulse timer: suppress keepLeft/keepRight on alternate seconds during
    # preLaneChange. This applies to both driver-initiated and nav-sourced keeps,
    # giving the model breathing room instead of a continuous same-direction bias.
    # ---------------------------------------------------------------------------
    if self.lane_change_state in (LaneChangeState.off, LaneChangeState.laneChangeStarting):
      self.keep_pulse_timer = 0.0
    elif self.lane_change_state == LaneChangeState.preLaneChange:
      self.keep_pulse_timer += DT_MDL
      if self.keep_pulse_timer > 1.0:
        self.keep_pulse_timer = 0.0
      elif self.desire in (log.Desire.keepLeft, log.Desire.keepRight):
        self.desire = log.Desire.none

    self.alc.update_state()
