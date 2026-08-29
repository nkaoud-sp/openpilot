"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.cereal import custom

from openpilot.common.constants import CV
from openpilot.common.params import Params

TurnDirection = custom.ModelDataV2SP.TurnDirection

LANE_CHANGE_SPEED_MIN = 20 * CV.MPH_TO_MS


# Manual on-road button override. The UI writes this param (0/1/2) on press and
# clears it (0) on release; the controller reads it every cycle so the request is
# responsive. Values match custom.ModelDataV2SP.TurnDirection.
LANE_TURN_BUTTON_PARAM = "LaneTurnButtonDirection"
BUTTON_TO_TURN = {
  1: TurnDirection.turnLeft,
  2: TurnDirection.turnRight,
}


class LaneTurnController:
  def __init__(self, desire_helper):
    self.DH = desire_helper
    self.turn_direction = TurnDirection.none
    self.button_direction = TurnDirection.none
    self.params = Params()
    self.lane_turn_value = float(self.params.get("LaneTurnValue", return_default=True)) * CV.MPH_TO_MS
    self.param_read_counter = 0
    self.enabled = self.params.get_bool("LaneTurnDesire")

  def read_params(self):
    self.enabled = self.params.get_bool("LaneTurnDesire")
    value = float(self.params.get("LaneTurnValue", return_default=True)) * CV.MPH_TO_MS
    self.lane_turn_value = min(float(LANE_CHANGE_SPEED_MIN), value)

  def update_params(self) -> None:
    if self.param_read_counter % 50 == 0:
      self.read_params()
    self.param_read_counter += 1

  def read_button(self) -> None:
    # Read every cycle for responsiveness; the button is a momentary manual request.
    value = self.params.get(LANE_TURN_BUTTON_PARAM, return_default=True)
    self.button_direction = BUTTON_TO_TURN.get(int(value or 0), TurnDirection.none)

  def update_lane_turn(self, blindspot_left: bool, blindspot_right: bool, left_blinker: bool, right_blinker: bool, v_ego: float) -> None:
    self.read_button()

    if left_blinker and not right_blinker and v_ego < self.lane_turn_value and not blindspot_left:
      self.turn_direction = TurnDirection.turnLeft
    elif right_blinker and not left_blinker and v_ego < self.lane_turn_value and not blindspot_right:
      self.turn_direction = TurnDirection.turnRight
    else:
      self.turn_direction = TurnDirection.none

  def get_turn_direction(self):
    # A held button is a manual override: it applies at any speed and regardless of
    # the LaneTurnDesire toggle, so it can be used to test turn desires directly.
    if self.button_direction != TurnDirection.none:
      return self.button_direction

    if not self.enabled:
      return TurnDirection.none
    return self.turn_direction
