"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntEnum

from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.auto_lock_settings import AutoLockSettingsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.dynamic_follow_settings import DynamicFollowSettingsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.launch_assist_settings import LaunchAssistSettingsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.park_assist_settings import ParkAssistSettingsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.speed_assist_settings import SpeedAssistSettingsLayout
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import simple_button_item_sp, toggle_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller_tici import Scroller


class PanelType(IntEnum):
  TWEAKS = 0
  PARK = 1
  LAUNCH = 2
  DYNAMIC_FOLLOW = 3
  SPEED_ASSIST = 4
  AUTO_LOCK = 5


class TweaksLayout(Widget):
  def __init__(self):
    super().__init__()

    self._current_panel = PanelType.TWEAKS
    self._dynamic_follow_layout = DynamicFollowSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._launch_layout = LaunchAssistSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._park_layout = ParkAssistSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._speed_assist_layout = SpeedAssistSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._auto_lock_layout = AutoLockSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._remember_experimental_mode = toggle_item_sp(
      title=lambda: tr("Remember Experimental Mode Status"),
      description=lambda: tr("Keep Experimental Mode set the way you left it after rebooting. Cars without " +
                            "openpilot longitudinal control will still force Experimental Mode off."),
      param="RememberExperimentalModeStatus",
    )

    self._dynamic_follow = toggle_item_sp(
      title=lambda: tr("Dynamic Follow Distance"),
      description=lambda: tr("Vary the follow distance with vehicle speed instead of using the fixed driving " +
                            "personality gap. Requires openpilot longitudinal control."),
      param="DynamicFollow",
    )
    self._dynamic_follow_button = simple_button_item_sp(
      button_text=lambda: tr("Manage Dynamic Follow Settings"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.DYNAMIC_FOLLOW),
    )

    self._launch_assist = toggle_item_sp(
      title=lambda: tr("Lead Launch Assist"),
      description=lambda: tr("When stopped behind a lead that pulls away, launch sooner instead of waiting for " +
                            "the model. The radar-based planner still enforces the safe gap, and it only acts " +
                            "from a full stop and never overrides the brake or gas. Requires openpilot " +
                            "longitudinal control."),
      param="LaunchAssist",
    )
    self._launch_assist_button = simple_button_item_sp(
      button_text=lambda: tr("Manage Launch Assist Settings"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.LAUNCH),
    )

    self._park_assist = toggle_item_sp(
      title=lambda: tr("Lead Halt Assist"),
      description=lambda: tr("When stopped behind a stopped lead, settle at a closer gap than the default. The " +
                            "gap smoothly returns to normal once the lead moves. Only acts near a standstill. " +
                            "Requires openpilot longitudinal control."),
      param="ParkAssist",
    )
    self._park_assist_button = simple_button_item_sp(
      button_text=lambda: tr("Manage Halt Assist Settings"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.PARK),
    )

    self._speed_assist_button = simple_button_item_sp(
      button_text=lambda: tr("Experimental Speed Assist"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.SPEED_ASSIST),
    )

    self._reverse_cruise = toggle_item_sp(
      title=lambda: tr("Reverse Cruise Increase"),
      description=lambda: tr("Reverse the cruise control button behavior so a short press increases the set speed " +
                            "by 5 instead of 1. Lexus/Toyota only. Requires openpilot longitudinal control."),
      param="ToyotaReverseCruise",
      callback=self._on_reverse_cruise,
      enabled=lambda: not ui_state.engaged,
    )

    self._vision_lane_change_risk = toggle_item_sp(
      title=lambda: tr("Vision Lane Change Warning"),
      description=lambda: tr("Use an experimental camera-only common-frame tracker to warn before a lane change " +
                            "when persistent activity is seen in the intended side zone. Assistive warning only."),
      param="VisionLaneChangeRisk",
    )

    self._vision_lane_change_risk_debug = toggle_item_sp(
      title=lambda: tr("Vision Lane Change Debug Frames"),
      description=lambda: tr("Save one annotated grayscale tracking PNG per second while driving for calibration. " +
                            "Files are stored in /data/media/0/vision_lane_change_risk_debug and this resets after " +
                            "the route."),
      param="VisionLaneChangeRiskDebug",
      enabled=lambda: self._vision_lane_change_risk.action_item.get_state(),
    )

    self._force_onroad_mode = toggle_item_sp(
      title=lambda: tr("Force Onroad Mode"),
      description=lambda: tr("Temporarily start onroad processes without real ignition for bench testing camera " +
                            "pipelines and debug captures. Clears after manager restart and is ignored by Always " +
                            "Offroad mode."),
      param="ForceOnroadMode",
      enabled=lambda: not ui_state.engaged,
    )

    self._auto_lock_button = simple_button_item_sp(
      button_text=lambda: tr("Auto Door Lock"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.AUTO_LOCK),
    )

    return [
      self._remember_experimental_mode,
      self._dynamic_follow,
      self._dynamic_follow_button,
      self._launch_assist,
      self._launch_assist_button,
      self._park_assist,
      self._park_assist_button,
      self._speed_assist_button,
      self._reverse_cruise,
      self._vision_lane_change_risk,
      self._vision_lane_change_risk_debug,
      self._force_onroad_mode,
      self._auto_lock_button,
    ]

  def _on_reverse_cruise(self, state: bool):
    # The flag is read at car-process init, so request an onroad cycle to apply it without a full reboot.
    ui_state.params.put_bool("ToyotaReverseCruise", state)
    ui_state.params.put_bool("OnroadCycleRequested", True)

  def _render(self, rect):
    if self._current_panel == PanelType.DYNAMIC_FOLLOW:
      self._dynamic_follow_layout.render(rect)
    elif self._current_panel == PanelType.LAUNCH:
      self._launch_layout.render(rect)
    elif self._current_panel == PanelType.PARK:
      self._park_layout.render(rect)
    elif self._current_panel == PanelType.SPEED_ASSIST:
      self._speed_assist_layout.render(rect)
    elif self._current_panel == PanelType.AUTO_LOCK:
      self._auto_lock_layout.render(rect)
    else:
      self._scroller.render(rect)

  def show_event(self):
    self._set_current_panel(PanelType.TWEAKS)
    self._scroller.show_event()

  def _set_current_panel(self, panel: PanelType):
    self._current_panel = panel
    if panel == PanelType.DYNAMIC_FOLLOW:
      self._dynamic_follow_layout.show_event()
    elif panel == PanelType.LAUNCH:
      self._launch_layout.show_event()
    elif panel == PanelType.PARK:
      self._park_layout.show_event()
    elif panel == PanelType.SPEED_ASSIST:
      self._speed_assist_layout.show_event()
    elif panel == PanelType.AUTO_LOCK:
      self._auto_lock_layout.show_event()

  def _update_state(self):
    super()._update_state()
    self._dynamic_follow_button.action_item.set_enabled(self._dynamic_follow.action_item.get_state())
    self._launch_assist_button.action_item.set_enabled(self._launch_assist.action_item.get_state())
    self._park_assist_button.action_item.set_enabled(self._park_assist.action_item.get_state())
    self._vision_lane_change_risk_debug.action_item.set_enabled(self._vision_lane_change_risk.action_item.get_state())
