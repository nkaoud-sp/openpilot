import time

from cereal import log, custom
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.selfdrive.controls.lib.auto_lane_change import AutoLaneChangeController, AutoLaneChangeMode
from openpilot.sunnypilot.selfdrive.controls.lib.lane_turn_desire import LaneTurnController

LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection
TurnDirection = custom.ModelDataV2SP.TurnDirection

# nkaoud_nav: map our NavDesire enum onto the upstream log.Desire. Keys are
# the string names (pycapnp returns _EnumValueProxy objects on read that
# don't hash the same way as the schema constants used at write time, so
# we normalize via str()). The laneChange* values are NOT used here as a
# direct desire override -- they kick the LaneChangeState machine and let
# its existing logic produce the right desire as the lane change executes.
NAV_DESIRE_MAP = {
  "none": log.Desire.none,
  "turnLeft": log.Desire.turnLeft,
  "turnRight": log.Desire.turnRight,
  "keepLeft": log.Desire.keepLeft,
  "keepRight": log.Desire.keepRight,
}
NAV_LANE_CHANGE_DIRS = {
  "laneChangeLeft": LaneChangeDirection.left,
  "laneChangeRight": LaneChangeDirection.right,
}

LANE_CHANGE_SPEED_MIN = 20 * CV.MPH_TO_MS
LANE_CHANGE_TIME_MAX = 10.

# nkaoud_nav: extra clearance gate for NAV-initiated lane changes only (the
# driver-blinker path is unchanged). On top of BSM, the visual vehicle
# detector's car-probability on the target side must stay below the threshold,
# and BOTH signals (BSM clear AND visual clear) must hold continuously for
# NAV_LC_VISUAL_CLEAR_TIME before the change may start; any breach resets it.
# Only the probability is used here -- never the detector's block/clear boolean.
VISUAL_CONF_BLOCK_THRESHOLD = 0.60   # P(car present) >= this blocks the side ("< 60%")
VISUAL_STALE_TIME = 1.0              # s; detector state older than this counts as no signal
NAV_LC_VISUAL_CLEAR_TIME = 4.0       # s; BSM + visual must both stay clear this long

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
    self.nav_lc_clear_timer = 0.0
    self.nav_lc_initiated = False
    self.keep_pulse_timer = 0.0
    self.prev_one_blinker = False
    self.desire = log.Desire.none
    self.alc = AutoLaneChangeController(self)
    self.lane_turn_controller = LaneTurnController(self)
    self.lane_turn_direction = TurnDirection.none

  @staticmethod
  def get_lane_change_direction(CS):
    return LaneChangeDirection.left if CS.leftBlinker else LaneChangeDirection.right

  def _visual_side_clear(self, direction, vs) -> bool:
    """True when the visual vehicle detector does NOT see a likely car on the
    target side. Falls back to True (BSM-only) whenever there is no usable
    per-side signal: no message, a stale message, or no side probability.
    Only the car-probability is consulted -- never the block/clear boolean."""
    if vs is None or direction == LaneChangeDirection.none:
      return True
    # The detector stamps monotonicTime with time.monotonic(); CLOCK_MONOTONIC
    # is system-wide, so it is directly comparable across processes.
    if (time.monotonic() - float(vs.monotonicTime)) > VISUAL_STALE_TIME:
      return True
    side = "left" if direction == LaneChangeDirection.left else "right"
    # Worst (highest) car-probability across every zone reporting this side, over
    # the active classifier zones and the wide+driver dual-camera zones.
    worst = None
    for zones in (vs.classifier.zones, vs.wideZones, vs.driverZones):
      for z in zones:
        if str(z.name) == side and bool(z.hasProbability):
          p = float(z.probability)
          worst = p if worst is None else max(worst, p)
    if worst is None:
      return True  # no per-side probability -> fall back to BSM only
    return worst < VISUAL_CONF_BLOCK_THRESHOLD

  def update(self, carstate, lateral_active, lane_change_prob, nav_desire="none", visual_vehicle_state=None):
    self.alc.update_params()
    self.lane_turn_controller.update_params()
    v_ego = carstate.vEgo
    one_blinker = carstate.leftBlinker != carstate.rightBlinker
    below_lane_change_speed = v_ego < LANE_CHANGE_SPEED_MIN

    # nkaoud_nav: is the route asking for an active lane change right now?
    nav_name_pre = str(nav_desire)
    nav_lc_dir = NAV_LANE_CHANGE_DIRS.get(nav_name_pre, LaneChangeDirection.none)
    nav_requesting_lc = nav_lc_dir != LaneChangeDirection.none

    # Lane turn controller update
    self.lane_turn_controller.update_lane_turn(blindspot_left=carstate.leftBlindspot, blindspot_right=carstate.rightBlindspot,
                                               left_blinker=carstate.leftBlinker, right_blinker=carstate.rightBlinker, v_ego=v_ego)
    self.lane_turn_direction = self.lane_turn_controller.get_turn_direction()

    if not lateral_active or self.lane_change_timer > LANE_CHANGE_TIME_MAX or self.alc.lane_change_set_timer == AutoLaneChangeMode.OFF:
      self.lane_change_state = LaneChangeState.off
      self.lane_change_direction = LaneChangeDirection.none
    else:
      # LaneChangeState.off -- enter on driver blinker or nav request.
      if self.lane_change_state == LaneChangeState.off:
        driver_kicked = one_blinker and not self.prev_one_blinker
        if (driver_kicked or nav_requesting_lc) and not below_lane_change_speed:
          self.lane_change_state = LaneChangeState.preLaneChange
          self.lane_change_ll_prob = 1.0
          self.nav_lc_clear_timer = 0.0
          # Ownership: the clearance gate applies only when the route initiated
          # this maneuver. A simultaneous driver blinker keeps the driver path.
          self.nav_lc_initiated = nav_requesting_lc and not driver_kicked
          # Initialize lane change direction (nav wins if both are active)
          self.lane_change_direction = nav_lc_dir if nav_requesting_lc else self.get_lane_change_direction(carstate)

      # LaneChangeState.preLaneChange
      elif self.lane_change_state == LaneChangeState.preLaneChange:
        # Direction: nav request takes precedence (it doesn't toggle blinkers).
        if nav_requesting_lc:
          self.lane_change_direction = nav_lc_dir
        else:
          self.lane_change_direction = self.get_lane_change_direction(carstate)

        torque_applied = carstate.steeringPressed and \
                         ((carstate.steeringTorque > 0 and self.lane_change_direction == LaneChangeDirection.left) or
                          (carstate.steeringTorque < 0 and self.lane_change_direction == LaneChangeDirection.right))

        blindspot_detected = ((carstate.leftBlindspot and self.lane_change_direction == LaneChangeDirection.left) or
                              (carstate.rightBlindspot and self.lane_change_direction == LaneChangeDirection.right))

        self.alc.update_lane_change(blindspot_detected, carstate.brakePressed)

        # nkaoud_nav: sustained BSM + visual clearance gate for nav-initiated
        # lane changes. The change may only start once BOTH BSM is clear AND the
        # visual detector's target-side car-probability is below threshold, held
        # continuously for NAV_LC_VISUAL_CLEAR_TIME. Any breach of either resets
        # the timer. Visual falls back to BSM-only when it has no fresh reading.
        visual_clear = self._visual_side_clear(self.lane_change_direction, visual_vehicle_state)
        nav_gate_clear = (not blindspot_detected) and visual_clear
        if nav_gate_clear:
          self.nav_lc_clear_timer += DT_MDL
        else:
          self.nav_lc_clear_timer = 0.0

        # Exit only when neither driver nor nav is asking, or speed dropped.
        if ((not one_blinker and not nav_requesting_lc) or below_lane_change_speed):
          self.lane_change_state = LaneChangeState.off
          self.lane_change_direction = LaneChangeDirection.none
        elif torque_applied and not blindspot_detected:
          # Driver physically nudged: honor immediately (BSM-gated), even while a
          # nav request is active. Preserves the driver's instant-override path.
          self.lane_change_state = LaneChangeState.laneChangeStarting
        elif self.nav_lc_initiated:
          # Nav-initiated: gated by the sustained BSM + visual clearance above.
          if nav_gate_clear and self.nav_lc_clear_timer >= NAV_LC_VISUAL_CLEAR_TIME:
            self.lane_change_state = LaneChangeState.laneChangeStarting
        elif self.alc.auto_lane_change_allowed and not blindspot_detected:
          # Driver-blinker auto-timer path, unchanged.
          self.lane_change_state = LaneChangeState.laneChangeStarting

      # LaneChangeState.laneChangeStarting
      elif self.lane_change_state == LaneChangeState.laneChangeStarting:
        # fade out over .5s
        self.lane_change_ll_prob = max(self.lane_change_ll_prob - 2 * DT_MDL, 0.0)

        # 98% certainty
        if lane_change_prob < 0.02 and self.lane_change_ll_prob < 0.01:
          self.lane_change_state = LaneChangeState.laneChangeFinishing

      # LaneChangeState.laneChangeFinishing
      elif self.lane_change_state == LaneChangeState.laneChangeFinishing:
        # fade in laneline over 1s
        self.lane_change_ll_prob = min(self.lane_change_ll_prob + DT_MDL, 1.0)

        if self.lane_change_ll_prob > 0.99:
          self.lane_change_direction = LaneChangeDirection.none
          if one_blinker:
            self.lane_change_state = LaneChangeState.preLaneChange
          else:
            self.lane_change_state = LaneChangeState.off

    if self.lane_change_state in (LaneChangeState.off, LaneChangeState.preLaneChange):
      self.lane_change_timer = 0.0
    else:
      self.lane_change_timer += DT_MDL

    self.prev_one_blinker = one_blinker

    if self.lane_turn_direction != TurnDirection.none:
      self.desire = TURN_DESIRES[self.lane_turn_direction]
    else:
      self.desire = DESIRES[self.lane_change_direction][self.lane_change_state]

    # nkaoud_nav: when a route-derived desire is present (and isn't a
    # lane-change request -- those drive the LaneChangeState machine above),
    # override the desire here. Gated at navd by NkaoudNavControlSteer.
    # We leave actively-running lane changes alone so we don't yank the
    # wheel mid-maneuver.
    nav_name = str(nav_desire)
    if (nav_name in NAV_DESIRE_MAP and nav_name != "none"
        and self.lane_change_state in (LaneChangeState.off, LaneChangeState.preLaneChange)):
      if nav_name in ("keepLeft", "keepRight"):
        # keep* is navd's cautious lane-change bias, so hold it to the same
        # clearance signals as a real lane change: only bias toward a side while
        # that side's BSM is clear AND the visual detector is below threshold.
        # Drop the bias (desire stays none) the instant either isn't clear.
        keep_dir = LaneChangeDirection.left if nav_name == "keepLeft" else LaneChangeDirection.right
        keep_bsm = carstate.leftBlindspot if keep_dir == LaneChangeDirection.left else carstate.rightBlindspot
        if not keep_bsm and self._visual_side_clear(keep_dir, visual_vehicle_state):
          self.desire = NAV_DESIRE_MAP[nav_name]
      else:
        # turnLeft / turnRight cues are not lane changes -- apply directly.
        self.desire = NAV_DESIRE_MAP[nav_name]

    # Send keep pulse once per second during LaneChangeStart.preLaneChange
    if self.lane_change_state in (LaneChangeState.off, LaneChangeState.laneChangeStarting):
      self.keep_pulse_timer = 0.0
    elif self.lane_change_state == LaneChangeState.preLaneChange:
      self.keep_pulse_timer += DT_MDL
      if self.keep_pulse_timer > 1.0:
        self.keep_pulse_timer = 0.0
      elif self.desire in (log.Desire.keepLeft, log.Desire.keepRight):
        self.desire = log.Desire.none

    self.alc.update_state()
