import time

from cereal import log, custom
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.selfdrive.controls.lib.auto_lane_change import AutoLaneChangeController, AutoLaneChangeMode
from openpilot.sunnypilot.selfdrive.controls.lib.lane_turn_desire import LaneTurnController

LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection
TurnDirection = custom.ModelDataV2SP.TurnDirection

# nkaoud_nav: map our NavDesire enum onto the upstream log.Desire. Keys are
# the string names (pycapnp returns _EnumValueProxy objects on read that
# don't hash the same way as the schema constants used at write time, so
# we normalize via str()). navd publishes pure route INTENT (turn*/keep*
# only); every permission and safety gate is applied here, where carstate,
# BSM, the visual detector, and the lane-change state machine are all fresh
# at 20 Hz.
NAV_DESIRE_MAP = {
  "none": log.Desire.none,
  "turnLeft": log.Desire.turnLeft,
  "turnRight": log.Desire.turnRight,
  "keepLeft": log.Desire.keepLeft,
  "keepRight": log.Desire.keepRight,
}

LANE_CHANGE_SPEED_MIN = 20 * CV.MPH_TO_MS
LANE_CHANGE_TIME_MAX = 10.

# nkaoud_nav: clearance gate for the route's keep* bias. On top of BSM, the
# visual vehicle detector's car-probability on the target side must stay
# below the threshold. Only the probability is used here -- never the
# detector's block/clear boolean.
VISUAL_CONF_BLOCK_THRESHOLD_DEFAULT = 0.80
VISUAL_CONF_BLOCK_THRESHOLD_MIN = 0.05
VISUAL_CONF_BLOCK_THRESHOLD_MAX = 0.95
VISUAL_CONF_BLOCK_THRESHOLD_PARAM = "NkaoudNavVisualBlockThreshold"
VISUAL_STALE_TIME = 1.0              # s; detector state older than this counts as no signal

# nkaoud_nav: keep* episode limits. The bias is open-loop (nothing confirms a
# lane change completed), so cap any continuous episode, and hold off after
# ANY lane change finishes to give the model and navd's lane estimate time to
# settle -- otherwise a just-stale "wrong lane" reading chains into a second
# move.
NAV_KEEP_EPISODE_MAX_S = 10.0
NAV_KEEP_COOLDOWN_S = 4.0
NAV_PARAM_READ_FRAMES = 50           # re-read NkaoudNavControlSteer every ~2.5 s


def visual_conf_block_threshold(params: Params) -> float:
  try:
    raw = params.get(VISUAL_CONF_BLOCK_THRESHOLD_PARAM, return_default=True)
    threshold = float(raw)
  except (TypeError, ValueError):
    threshold = VISUAL_CONF_BLOCK_THRESHOLD_DEFAULT
  return max(VISUAL_CONF_BLOCK_THRESHOLD_MIN, min(VISUAL_CONF_BLOCK_THRESHOLD_MAX, threshold))


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
    # nkaoud_nav gating state
    self.params = Params()
    self.nav_steer_enabled = False
    self.visual_conf_block_threshold = VISUAL_CONF_BLOCK_THRESHOLD_DEFAULT
    self.nav_param_counter = 0
    self.nav_keep_timer = 0.0        # continuous keep* emission time
    self.nav_cooldown_timer = 0.0    # counts down after any lane change ends
    self.prev_lane_change_state = LaneChangeState.off
    self.prev_nav_keep = ""

  @staticmethod
  def get_lane_change_direction(CS):
    return LaneChangeDirection.left if CS.leftBlinker else LaneChangeDirection.right

  def _update_nav_params(self) -> None:
    if self.nav_param_counter % NAV_PARAM_READ_FRAMES == 0:
      self.nav_steer_enabled = self.params.get_bool("NkaoudNavControlSteer")
      self.visual_conf_block_threshold = visual_conf_block_threshold(self.params)
    self.nav_param_counter += 1

  def _update_nav_cooldown(self) -> None:
    """Arm the nav keep* cooldown when ANY lane change finishes (driver or
    otherwise) and count it down while the machine is idle."""
    if self.prev_lane_change_state != LaneChangeState.off and self.lane_change_state == LaneChangeState.off:
      self.nav_cooldown_timer = NAV_KEEP_COOLDOWN_S
    self.prev_lane_change_state = self.lane_change_state
    self.nav_cooldown_timer = max(0.0, self.nav_cooldown_timer - DT_MDL)

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
    return worst < self.visual_conf_block_threshold

  def update(self, carstate, lateral_active, lane_change_prob, nav_desire="none", visual_vehicle_state=None):
    self.alc.update_params()
    self.lane_turn_controller.update_params()
    self._update_nav_params()
    v_ego = carstate.vEgo
    one_blinker = carstate.leftBlinker != carstate.rightBlinker
    below_lane_change_speed = v_ego < LANE_CHANGE_SPEED_MIN

    # Lane turn controller update
    self.lane_turn_controller.update_lane_turn(blindspot_left=carstate.leftBlindspot, blindspot_right=carstate.rightBlindspot,
                                               left_blinker=carstate.leftBlinker, right_blinker=carstate.rightBlinker, v_ego=v_ego)
    self.lane_turn_direction = self.lane_turn_controller.get_turn_direction()

    if not lateral_active or self.lane_change_timer > LANE_CHANGE_TIME_MAX or self.alc.lane_change_set_timer == AutoLaneChangeMode.OFF:
      self.lane_change_state = LaneChangeState.off
      self.lane_change_direction = LaneChangeDirection.none
    else:
      # LaneChangeState.off -- enter on driver blinker only. Route keep*
      # intent never touches this machine; it is applied as a desire bias
      # below.
      if self.lane_change_state == LaneChangeState.off:
        if one_blinker and not self.prev_one_blinker and not below_lane_change_speed:
          self.lane_change_state = LaneChangeState.preLaneChange
          self.lane_change_ll_prob = 1.0
          self.lane_change_direction = self.get_lane_change_direction(carstate)

      # LaneChangeState.preLaneChange
      elif self.lane_change_state == LaneChangeState.preLaneChange:
        self.lane_change_direction = self.get_lane_change_direction(carstate)

        torque_applied = carstate.steeringPressed and \
                         ((carstate.steeringTorque > 0 and self.lane_change_direction == LaneChangeDirection.left) or
                          (carstate.steeringTorque < 0 and self.lane_change_direction == LaneChangeDirection.right))

        blindspot_detected = ((carstate.leftBlindspot and self.lane_change_direction == LaneChangeDirection.left) or
                              (carstate.rightBlindspot and self.lane_change_direction == LaneChangeDirection.right))

        self.alc.update_lane_change(blindspot_detected, carstate.brakePressed)

        if not one_blinker or below_lane_change_speed:
          self.lane_change_state = LaneChangeState.off
          self.lane_change_direction = LaneChangeDirection.none
        elif torque_applied and not blindspot_detected:
          self.lane_change_state = LaneChangeState.laneChangeStarting
        elif self.alc.auto_lane_change_allowed and not blindspot_detected:
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

    # nkaoud_nav: apply the route's lateral intent. navd publishes what the
    # route WANTS; every gate lives here. Nothing is applied unless the user
    # enabled NkaoudNavControlSteer, lateral is active, and the lane-change
    # state machine is idle (an in-flight driver maneuver always wins).
    self._update_nav_cooldown()
    nav_name = str(nav_desire)
    nav_keep = nav_name in ("keepLeft", "keepRight")
    # Track continuous keep* emission; reset whenever the intent stops or
    # flips sides so each new episode gets a fresh budget.
    if not nav_keep or nav_name != self.prev_nav_keep:
      self.nav_keep_timer = 0.0
    self.prev_nav_keep = nav_name if nav_keep else ""

    if (self.nav_steer_enabled and lateral_active
        and self.lane_change_state == LaneChangeState.off and nav_name != "none"):
      if nav_keep:
        # keep* is a cautious, open-loop lane-change bias. Gates: driver not
        # signaling, AutoLaneChange enabled, out of the post-change cooldown,
        # target-side BSM clear, visual detector below threshold, and the
        # episode budget not exhausted (a stuck "wrong lane" estimate must
        # not bias steering forever).
        keep_dir = LaneChangeDirection.left if nav_name == "keepLeft" else LaneChangeDirection.right
        keep_bsm = carstate.leftBlindspot if keep_dir == LaneChangeDirection.left else carstate.rightBlindspot
        if (not one_blinker
            and self.alc.lane_change_set_timer != AutoLaneChangeMode.OFF
            and self.nav_cooldown_timer <= 0.0
            and self.nav_keep_timer < NAV_KEEP_EPISODE_MAX_S
            and not keep_bsm
            and self._visual_side_clear(keep_dir, visual_vehicle_state)):
          self.desire = NAV_DESIRE_MAP[nav_name]
          self.nav_keep_timer += DT_MDL
      elif nav_name in ("turnLeft", "turnRight"):
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
