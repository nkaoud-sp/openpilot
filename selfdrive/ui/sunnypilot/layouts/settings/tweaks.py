"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntEnum

from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.auto_lock_settings import AutoLockSettingsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.dynamic_follow_settings import DynamicFollowSettingsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.jerk_settings import JerkSettingsLayout
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, simple_button_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller_tici import Scroller


class PanelType(IntEnum):
  TWEAKS = 0
  AUTO_LOCK = 1
  DYNAMIC_FOLLOW = 2
  JERK = 3


class TweaksLayout(Widget):
  def __init__(self):
    super().__init__()

    self._current_panel = PanelType.TWEAKS
    self._auto_lock_layout = AutoLockSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._dynamic_follow_layout = DynamicFollowSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._jerk_layout = JerkSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    # Secure-on-exit (doorlockd). Toyota/Lexus only.
    # Detailed settings live in a sub-page reachable via the Manage button below.
    self._auto_lock = toggle_item_sp(
      title=lambda: tr("Auto Lock On Exit"),
      description=lambda: tr("Lock the doors (and optionally fold the mirrors / close the windows) once you leave " +
                            "the car. Toyota/Lexus only."),
      param="AutoLockEnabled",
    )
    self._auto_lock_button = simple_button_item_sp(
      button_text=lambda: tr("Manage Auto Lock Settings"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.AUTO_LOCK),
    )

    # Dynamic follow distance (speed-based gap, overrides the personality setting).
    # Detailed settings live in a sub-page reachable via the Manage button below.
    self._dynamic_follow = toggle_item_sp(
      title=lambda: tr("Dynamic Follow Distance"),
      description=lambda: tr("Vary the follow distance with vehicle speed instead of using the fixed driving " +
                            "personality gap. The follow time scales linearly from the low-speed value (at 0 km/h) " +
                            "to the high-speed value (at 130 km/h). Requires openpilot longitudinal control."),
      param="DynamicFollow",
    )
    self._dynamic_follow_button = simple_button_item_sp(
      button_text=lambda: tr("Manage Dynamic Follow Settings"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.DYNAMIC_FOLLOW),
    )

    # Asymmetric jerk (separate accel / decel ramp smoothness).
    # Detailed settings live in a sub-page reachable via the Manage button below.
    self._asymmetric_jerk = toggle_item_sp(
      title=lambda: tr("Asymmetric Accel / Decel Smoothness"),
      description=lambda: tr("Tune how gently acceleration and braking build up independently, so braking can " +
                            "ramp in more smoothly than acceleration (or vice-versa). Requires openpilot " +
                            "longitudinal control."),
      param="AsymmetricJerk",
    )
    self._asymmetric_jerk_button = simple_button_item_sp(
      button_text=lambda: tr("Manage Accel / Decel Smoothness"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.JERK),
    )

    items = [
      self._auto_lock,
      self._auto_lock_button,
      self._dynamic_follow,
      self._dynamic_follow_button,
      self._asymmetric_jerk,
      self._asymmetric_jerk_button,
    ]
    return items

  def _render(self, rect):
    if self._current_panel == PanelType.AUTO_LOCK:
      self._auto_lock_layout.render(rect)
    elif self._current_panel == PanelType.DYNAMIC_FOLLOW:
      self._dynamic_follow_layout.render(rect)
    elif self._current_panel == PanelType.JERK:
      self._jerk_layout.render(rect)
    else:
      self._scroller.render(rect)

  def show_event(self):
    self._set_current_panel(PanelType.TWEAKS)
    self._scroller.show_event()

  def _set_current_panel(self, panel: PanelType):
    self._current_panel = panel
    if panel == PanelType.AUTO_LOCK:
      self._auto_lock_layout.show_event()
    elif panel == PanelType.DYNAMIC_FOLLOW:
      self._dynamic_follow_layout.show_event()
    elif panel == PanelType.JERK:
      self._jerk_layout.show_event()

  def _update_state(self):
    super()._update_state()

    # the Manage buttons are only available once their feature is enabled
    self._auto_lock_button.action_item.set_enabled(self._auto_lock.action_item.get_state())
    self._dynamic_follow_button.action_item.set_enabled(self._dynamic_follow.action_item.get_state())
    self._asymmetric_jerk_button.action_item.set_enabled(self._asymmetric_jerk.action_item.get_state())
