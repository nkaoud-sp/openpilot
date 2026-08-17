import json
import time

import numpy as np

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
NAV_PARAM_READ_FRAMES = 50           # native visual detector threshold refresh

# StarPilot navigation policy constants. These mirror the upstream source
# policy; only the setting transport is target-native.
STARPILOT_NAV_TURN_DISTANCE_SPEED_BREAKPOINTS = (0.0, 5.0, 10.0)
STARPILOT_NAV_TURN_DISTANCE_BREAKPOINTS = (20.0, 25.0, 30.0)
STARPILOT_NAV_KEEP_DISTANCE_SPEED_BREAKPOINTS = (0.0, 15.0, 30.0)
STARPILOT_NAV_KEEP_DISTANCE_BREAKPOINTS = (25.0, 90.0, 160.0)
STARPILOT_NAV_KEEP_AMBIGUOUS_SPLIT_DISTANCE_SCALE = 0.6
STARPILOT_NAV_KEEP_SMALL_SPLIT_MAX_OTHER_LANES = 2
STARPILOT_LANE_WIDTH_UPDATE_FRAMES = 4
STARPILOT_MIN_LANE_CHANGE_SPEED_DEFAULT_MPH = 20.0


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
    self.nav_starpilot_provider = False
    self.nav_starpilot_lane_positioning = False
    self.nav_starpilot_lane_detection_width = 0.0
    self.nav_starpilot_min_lane_change_speed = STARPILOT_MIN_LANE_CHANGE_SPEED_DEFAULT_MPH * CV.MPH_TO_MS
    self.nav_provider_matches = False
    self.visual_conf_block_threshold = VISUAL_CONF_BLOCK_THRESHOLD_DEFAULT
    self.nav_param_counter = 0
    self.nav_keep_timer = 0.0        # continuous keep* emission time
    self.nav_cooldown_timer = 0.0    # counts down after any lane change ends
    self.prev_lane_change_state = LaneChangeState.off
    self.prev_nav_keep = ""
    self.starpilot_lane_width_left = 0.0
    self.starpilot_lane_width_right = 0.0
    self.starpilot_lane_width_counter = 0

  @staticmethod
  def get_lane_change_direction(CS):
    return LaneChangeDirection.left if CS.leftBlinker else LaneChangeDirection.right

  def _update_nav_params(self) -> None:
    # Re-read the control master every model frame. The provider tag is checked
    # on the same frame, so disabling or switching can never leave an old route
    # desire active during the native/StarPilot handoff.
    self.nav_steer_enabled = self.params.get_bool("NkaoudNavControlSteer")
    if self.nav_param_counter % NAV_PARAM_READ_FRAMES == 0:
      self.visual_conf_block_threshold = visual_conf_block_threshold(self.params)
    # Provider selection changes which lateral policy is permissible. Read the
    # StarPilot settings every frame so an active route can't cross a stale
    # configuration cache window.
    try:
      self.nav_starpilot_provider = int(self.params.get("NkaoudNavRoutingProvider", return_default=True)) == 1
    except (TypeError, ValueError):
      self.nav_starpilot_provider = False
    self.nav_starpilot_lane_positioning = self.params.get_bool("NkaoudNavStarPilotLanePositioning")
    try:
      self.nav_starpilot_lane_detection_width = max(
        0.0, float(self.params.get("NkaoudNavStarPilotLaneDetectionWidth", return_default=True)),
      )
    except (TypeError, ValueError):
      self.nav_starpilot_lane_detection_width = 0.0
    try:
      speed_mph = max(0.0, float(self.params.get("NkaoudNavStarPilotMinimumLaneChangeSpeed", return_default=True)))
    except (TypeError, ValueError):
      speed_mph = STARPILOT_MIN_LANE_CHANGE_SPEED_DEFAULT_MPH
    self.nav_starpilot_min_lane_change_speed = speed_mph * CV.MPH_TO_MS
    self.nav_param_counter += 1

  @staticmethod
  def _starpilot_lane_width(lane, current_lane, road_edge) -> float:
    """Source-equivalent adjacent lane width from model geometry.

    StarPilot treats a road edge closer than the candidate lane line as an
    unavailable adjacent lane. Invalid/incomplete model geometry fails closed
    to zero width.
    """
    try:
      current_x = np.asarray(current_lane.x, dtype=float)
      current_y = np.asarray(current_lane.y, dtype=float)
      lane_x = np.asarray(lane.x, dtype=float)
      lane_y = np.asarray(lane.y, dtype=float)
      if current_x.size == 0 or current_y.size == 0 or lane_x.size == 0 or lane_y.size == 0:
        return 0.0
      lane_y_interp = np.interp(current_x, lane_x, lane_y)
      distance_to_lane = float(np.mean(np.abs(current_y - lane_y_interp)))
      if not np.isfinite(distance_to_lane):
        return 0.0

      if road_edge is None:
        return distance_to_lane
      edge_x = np.asarray(road_edge.x, dtype=float)
      edge_y = np.asarray(road_edge.y, dtype=float)
      if edge_x.size == 0 or edge_y.size == 0:
        return distance_to_lane
      edge_y_interp = np.interp(current_x, edge_x, edge_y)
      distance_to_edge = float(np.mean(np.abs(current_y - edge_y_interp)))
      if not np.isfinite(distance_to_edge) or distance_to_edge < distance_to_lane:
        return 0.0
      return distance_to_lane
    except (AttributeError, TypeError, ValueError):
      return 0.0

  def _update_starpilot_lane_widths(self, model_data, v_ego: float) -> None:
    """Refresh the StarPilot lane-width inputs every fourth model result.

    This matches StarPilot's planner cadence and deliberately does not use this
    fork's lane-count/edge estimator or its confidence value.
    """
    if model_data is None or v_ego < self.nav_starpilot_min_lane_change_speed:
      self.starpilot_lane_width_counter = 0
      self.starpilot_lane_width_left = 0.0
      self.starpilot_lane_width_right = 0.0
      return

    self.starpilot_lane_width_counter += 1
    if self.starpilot_lane_width_counter % STARPILOT_LANE_WIDTH_UPDATE_FRAMES:
      return
    try:
      lane_lines = model_data.laneLines
      road_edges = model_data.roadEdges
      self.starpilot_lane_width_left = self._starpilot_lane_width(lane_lines[0], lane_lines[1], road_edges[0])
      self.starpilot_lane_width_right = self._starpilot_lane_width(lane_lines[3], lane_lines[2], road_edges[1])
    except (IndexError, TypeError):
      self.starpilot_lane_width_left = 0.0
      self.starpilot_lane_width_right = 0.0

  @staticmethod
  def _parse_starpilot_instruction_state(raw_state) -> dict:
    if isinstance(raw_state, dict):
      return raw_state
    if not raw_state:
      return {}
    try:
      parsed = json.loads(str(raw_state))
    except (TypeError, ValueError):
      return {}
    return parsed if isinstance(parsed, dict) else {}

  @staticmethod
  def _starpilot_nav_keep_direction_is_clear(carstate, direction) -> bool:
    return not (
      (direction == LaneChangeDirection.left and carstate.leftBlindspot)
      or (direction == LaneChangeDirection.right and carstate.rightBlindspot)
    )

  @staticmethod
  def _starpilot_nav_torque_applied(carstate, direction) -> bool:
    return carstate.steeringPressed and (
      (direction == LaneChangeDirection.left and carstate.steeringTorque > 0)
      or (direction == LaneChangeDirection.right and carstate.steeringTorque < 0)
    )

  @staticmethod
  def _starpilot_nav_turn_is_imminent(v_ego: float, maneuver_distance) -> bool:
    try:
      distance = float(maneuver_distance)
    except (TypeError, ValueError):
      return False
    threshold = np.interp(
      v_ego, STARPILOT_NAV_TURN_DISTANCE_SPEED_BREAKPOINTS, STARPILOT_NAV_TURN_DISTANCE_BREAKPOINTS,
    )
    return distance <= float(threshold)

  @staticmethod
  def _starpilot_nav_int(value, default=0) -> int:
    try:
      return int(value or 0)
    except (TypeError, ValueError):
      return default

  @staticmethod
  def _starpilot_nav_should_delay_ambiguous_split(maneuver_type="", same_side_lane_count=0, lane_count=0) -> bool:
    maneuver_type = str(maneuver_type or "")
    same_side_lane_count = DesireHelper._starpilot_nav_int(same_side_lane_count)
    if maneuver_type not in ("off ramp", "fork") or same_side_lane_count <= 1:
      return False
    total_lanes = DesireHelper._starpilot_nav_int(lane_count)
    if total_lanes <= 0:
      return True
    return max(total_lanes - same_side_lane_count, 0) <= STARPILOT_NAV_KEEP_SMALL_SPLIT_MAX_OTHER_LANES

  @staticmethod
  def _starpilot_nav_keep_is_imminent(v_ego: float, maneuver_distance, maneuver_type="", same_side_lane_count=0, lane_count=0) -> bool:
    try:
      distance = float(maneuver_distance)
    except (TypeError, ValueError):
      return False
    threshold = np.interp(
      v_ego, STARPILOT_NAV_KEEP_DISTANCE_SPEED_BREAKPOINTS, STARPILOT_NAV_KEEP_DISTANCE_BREAKPOINTS,
    )
    if DesireHelper._starpilot_nav_should_delay_ambiguous_split(maneuver_type, same_side_lane_count, lane_count):
      threshold *= STARPILOT_NAV_KEEP_AMBIGUOUS_SPLIT_DISTANCE_SCALE
    return distance <= float(threshold)

  @staticmethod
  def _starpilot_nav_should_suppress_edge_lane_keep(state: dict) -> bool:
    maneuver_type = str(state.get("maneuverType", ""))
    if maneuver_type not in ("off ramp", "fork"):
      return False
    active_direction = str(state.get("activeLaneDirection", ""))
    if active_direction not in ("slightLeft", "left", "sharpLeft", "slightRight", "right", "sharpRight"):
      return False
    same_side_count = DesireHelper._starpilot_nav_int(state.get("sameSideLaneCount", 0))
    lane_count = DesireHelper._starpilot_nav_int(state.get("laneCount", 0))
    return (
      DesireHelper._starpilot_nav_should_delay_ambiguous_split(maneuver_type, same_side_count, lane_count)
      and bool(state.get("activeLaneAtRoadEdge", False))
      and bool(state.get("hasSharedSameSideLane", False))
    )

  @staticmethod
  def _starpilot_nav_effective_modifier(state: dict, v_ego: float, maneuver_distance) -> str:
    modifier = str(state.get("maneuverModifier", ""))
    maneuver_type = str(state.get("maneuverType", ""))
    same_side_count = DesireHelper._starpilot_nav_int(state.get("sameSideLaneCount", 0))
    lane_count = DesireHelper._starpilot_nav_int(state.get("laneCount", 0))
    if maneuver_type in ("off ramp", "fork") and modifier in (
        "slightLeft", "left", "sharpLeft", "slightRight", "right", "sharpRight"):
      if not DesireHelper._starpilot_nav_keep_is_imminent(
          v_ego, maneuver_distance, maneuver_type, same_side_count, lane_count):
        return ""
      if DesireHelper._starpilot_nav_should_suppress_edge_lane_keep(state):
        return ""
      active_direction = str(state.get("activeLaneDirection", ""))
      if active_direction in ("slightLeft", "left"):
        return "slightLeft"
      if active_direction in ("slightRight", "right"):
        return "slightRight"
      return ""
    return modifier

  def _starpilot_navigation_desire(self, carstate, lateral_active, state: dict):
    """Exact StarPilot navigation overlay: a model desire, never a direct
    lane-change-state transition or a curvature command."""
    if not self.nav_steer_enabled or not lateral_active or not bool(state.get("valid", False)):
      return log.Desire.none

    maneuver_distance = state.get("maneuverDistance", 0.0)
    modifier = self._starpilot_nav_effective_modifier(state, carstate.vEgo, maneuver_distance)
    if modifier == "slightLeft":
      if not self.nav_starpilot_lane_positioning:
        return log.Desire.none
      direction = LaneChangeDirection.left
      if not carstate.rightBlinker and self._starpilot_nav_keep_direction_is_clear(carstate, direction):
        if (self.starpilot_lane_width_left >= self.nav_starpilot_lane_detection_width
            and self._starpilot_nav_torque_applied(carstate, direction)):
          return log.Desire.keepLeft
    elif modifier == "slightRight":
      if not self.nav_starpilot_lane_positioning:
        return log.Desire.none
      direction = LaneChangeDirection.right
      if not carstate.leftBlinker and self._starpilot_nav_keep_direction_is_clear(carstate, direction):
        if (self.starpilot_lane_width_right >= self.nav_starpilot_lane_detection_width
            and self._starpilot_nav_torque_applied(carstate, direction)):
          return log.Desire.keepRight
    elif modifier in ("left", "sharpLeft"):
      allowed = not carstate.rightBlinker and not carstate.leftBlindspot
      allowed &= carstate.vEgo < self.nav_starpilot_min_lane_change_speed and not carstate.standstill
      if allowed and self._starpilot_nav_turn_is_imminent(carstate.vEgo, maneuver_distance):
        return log.Desire.turnLeft
    elif modifier in ("right", "sharpRight"):
      allowed = not carstate.leftBlinker and not carstate.rightBlindspot
      allowed &= carstate.vEgo < self.nav_starpilot_min_lane_change_speed and not carstate.standstill
      if allowed and self._starpilot_nav_turn_is_imminent(carstate.vEgo, maneuver_distance):
        return log.Desire.turnRight
    return log.Desire.none

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

  def update(self, carstate, lateral_active, lane_change_prob, nav_desire="none", nav_provider=0,
             starpilot_instruction_state="", model_data=None, visual_vehicle_state=None):
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

    # The nav message carries its source. A provider setting change reaches
    # this 20 Hz loop faster than navd's next 5 Hz publish, so do not apply a
    # retained old-provider output during that handoff.
    expected_nav_provider = 1 if self.nav_starpilot_provider else 0
    try:
      self.nav_provider_matches = int(nav_provider) == expected_nav_provider
    except (TypeError, ValueError):
      self.nav_provider_matches = False

    if self.nav_starpilot_provider:
      # These are native-only route-positioning state variables. Do not let a
      # mode switch import their visual detector, cooldown, or time-budget
      # semantics into the StarPilot path.
      self.nav_keep_timer = 0.0
      self.nav_cooldown_timer = 0.0
      self.prev_nav_keep = ""
      self.prev_lane_change_state = self.lane_change_state
    else:
      # Native nkaoud_nav policy remains unchanged. It consumes navd's compact
      # recommendedDesire and keeps its own visual/cooldown protections.
      self._update_nav_cooldown()
      nav_name = str(nav_desire) if self.nav_provider_matches else "none"
      nav_keep = nav_name in ("keepLeft", "keepRight")
      if not nav_keep or nav_name != self.prev_nav_keep:
        self.nav_keep_timer = 0.0
      self.prev_nav_keep = nav_name if nav_keep else ""

      if (self.nav_steer_enabled and lateral_active
          and self.lane_change_state == LaneChangeState.off and nav_name != "none"):
        if nav_keep:
          keep_dir = LaneChangeDirection.left if nav_name == "keepLeft" else LaneChangeDirection.right
          keep_bsm = carstate.leftBlindspot if keep_dir == LaneChangeDirection.left else carstate.rightBlindspot
          native_keep_allowed = (
            not one_blinker
            and self.alc.lane_change_set_timer != AutoLaneChangeMode.OFF
          )
          if (native_keep_allowed
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

    # In StarPilot mode the raw NavInstructionState is evaluated here, after
    # the regular lane-change and keep-pulse logic, exactly where StarPilot
    # overlays navigation on its DesireHelper. This path intentionally does
    # not consult the target visual detector, lane-count estimator, cooldown,
    # or keep-episode timer.
    if self.nav_starpilot_provider:
      self._update_starpilot_lane_widths(model_data, v_ego)
      if self.nav_provider_matches:
        state = self._parse_starpilot_instruction_state(starpilot_instruction_state)
        starpilot_desire = self._starpilot_navigation_desire(carstate, lateral_active, state)
        if starpilot_desire != log.Desire.none and self.lane_change_state == LaneChangeState.off:
          self.desire = starpilot_desire

    self.alc.update_state()
