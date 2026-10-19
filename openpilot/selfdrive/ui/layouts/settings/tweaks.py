from enum import IntEnum

from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.layouts.settings.tweaks_sub_layouts.dynamic_follow_settings import DynamicFollowSettingsLayout
from openpilot.selfdrive.ui.layouts.settings.tweaks_sub_layouts.launch_assist_settings import LaunchAssistSettingsLayout
from openpilot.selfdrive.ui.layouts.settings.tweaks_sub_layouts.park_assist_settings import ParkAssistSettingsLayout
from openpilot.selfdrive.ui.layouts.settings.tweaks_sub_layouts.speed_assist_settings import SpeedAssistSettingsLayout
from openpilot.selfdrive.ui.layouts.settings.tweaks_sub_layouts.auto_lock_settings import AutoLockSettingsLayout
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.list_view import toggle_item, button_item
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.system.ui.lib.multilang import tr

MANAGE_BUTTON_WIDTH = 820


class PanelType(IntEnum):
  TWEAKS = 0
  DYNAMIC_FOLLOW = 1
  LAUNCH = 2
  PARK = 3
  SPEED_ASSIST = 4
  AUTO_LOCK = 5


class TweaksLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()
    self._current_panel = PanelType.TWEAKS

    self._dynamic_follow_layout = DynamicFollowSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._launch_layout = LaunchAssistSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._park_layout = ParkAssistSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._speed_assist_layout = SpeedAssistSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._auto_lock_layout = AutoLockSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))

    self._dynamic_follow = toggle_item(
      lambda: tr("Dynamic Follow"),
      description=lambda: tr("Vary the follow distance with vehicle speed instead of using the fixed driving " +
                             "personality gap. Requires openpilot longitudinal control."),
      initial_state=self._params.get_bool("DynamicFollow"),
      callback=lambda state: self._params.put_bool("DynamicFollow", state),
      icon="speed_limit.png",
    )
    self._launch_assist = toggle_item(
      lambda: tr("Lead Launch Assist"),
      description=lambda: tr("When stopped behind a lead that pulls away, launch sooner instead of waiting for " +
                             "the model. The radar planner still enforces the safe gap, it only acts from a full " +
                             "stop and never overrides the brake or gas. Requires openpilot longitudinal control."),
      initial_state=self._params.get_bool("LaunchAssist"),
      callback=lambda state: self._params.put_bool("LaunchAssist", state),
      icon="speed_limit.png",
    )
    self._park_assist = toggle_item(
      lambda: tr("Lead Halt Assist"),
      description=lambda: tr("When stopped behind a stopped lead, settle at a closer gap than the default. The gap " +
                             "smoothly returns to normal once the lead moves. Only acts near a standstill. " +
                             "Requires openpilot longitudinal control."),
      initial_state=self._params.get_bool("ParkAssist"),
      callback=lambda state: self._params.put_bool("ParkAssist", state),
      icon="speed_limit.png",
    )
    self._reverse_cruise = toggle_item(
      lambda: tr("Reverse Cruise Increase"),
      description=lambda: tr("Reverse the cruise control button behavior so a short press increases the set speed " +
                             "by 5 instead of 1. Lexus/Toyota only. Requires openpilot longitudinal control."),
      initial_state=self._params.get_bool("ToyotaReverseCruise"),
      callback=lambda state: self._params.put_bool("ToyotaReverseCruise", state),
      icon="speed_limit.png",
      enabled=lambda: not ui_state.engaged,
    )

    items = [
      self._dynamic_follow,
      button_item("", lambda: tr("Manage Dynamic Follow Settings"), button_width=MANAGE_BUTTON_WIDTH,
                  callback=lambda: self._set_current_panel(PanelType.DYNAMIC_FOLLOW)),
      self._launch_assist,
      button_item("", lambda: tr("Manage Launch Assist Settings"), button_width=MANAGE_BUTTON_WIDTH,
                  callback=lambda: self._set_current_panel(PanelType.LAUNCH)),
      self._park_assist,
      button_item("", lambda: tr("Manage Halt Assist Settings"), button_width=MANAGE_BUTTON_WIDTH,
                  callback=lambda: self._set_current_panel(PanelType.PARK)),
      button_item("", lambda: tr("Experimental Speed Assist"), button_width=MANAGE_BUTTON_WIDTH,
                  callback=lambda: self._set_current_panel(PanelType.SPEED_ASSIST)),
      self._reverse_cruise,
      button_item("", lambda: tr("Auto Door Lock"), button_width=MANAGE_BUTTON_WIDTH,
                  callback=lambda: self._set_current_panel(PanelType.AUTO_LOCK)),
    ]
    self._scroller = Scroller(items, line_separator=True, spacing=0)

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
    super().show_event()
    self._set_current_panel(PanelType.TWEAKS)

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
    else:
      # refresh the menu toggles from params to mirror any external changes
      self._dynamic_follow.action_item.set_state(self._params.get_bool("DynamicFollow"))
      self._launch_assist.action_item.set_state(self._params.get_bool("LaunchAssist"))
      self._park_assist.action_item.set_state(self._params.get_bool("ParkAssist"))
      self._reverse_cruise.action_item.set_state(self._params.get_bool("ToyotaReverseCruise"))
      self._scroller.show_event()
