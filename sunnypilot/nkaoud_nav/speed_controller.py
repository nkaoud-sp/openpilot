"""
Turn-slowdown speed controller for nkaoud_nav.

Consumes nkaoudNavigationSP.maneuverTargetSpeed (computed by nkaoud_navd)
and exposes it to LongitudinalPlannerSP.update_targets as another candidate
speed cap. The planner takes the min across all sources, so a non-zero
target here slows the car for the upcoming turn.

Gated on the NkaoudNavControlSpeed param.

This file lives under sunnypilot/nkaoud_nav/ so the entire experimental
navigation feature stays in one namespace.
"""
from __future__ import annotations

import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD


class NkaoudNavSpeedController:
  output_v_target: float = V_CRUISE_UNSET
  output_a_target: float = 0.0

  def __init__(self) -> None:
    self.params = Params()
    self.frame = -1
    self.enabled = self.params.get_bool("NkaoudNavControlSpeed")
    self.is_active = False  # currently capping the cruise target
    self.target_speed_ms = 0.0
    self.long_enabled = False
    self.long_override = False

  def update(self, sm: messaging.SubMaster, long_enabled: bool, long_override: bool,
             v_ego: float, a_ego: float, v_cruise: float) -> None:
    self.frame += 1
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.enabled = self.params.get_bool("NkaoudNavControlSpeed")

    self.long_enabled = long_enabled
    self.long_override = long_override
    self.is_active = False
    self.output_v_target = V_CRUISE_UNSET
    self.output_a_target = a_ego

    if not (self.enabled and long_enabled) or long_override:
      self.target_speed_ms = 0.0
      return

    nav = sm['nkaoudNavigationSP']
    if not nav.active:
      self.target_speed_ms = 0.0
      return

    self.target_speed_ms = float(nav.maneuverTargetSpeed)
    if self.target_speed_ms <= 0.0:
      return

    # Only cap if our target is below the cruise setpoint; otherwise we'd
    # actually raise the speed limit which is not what we want.
    if self.target_speed_ms < v_cruise:
      self.output_v_target = self.target_speed_ms
      self.is_active = True
