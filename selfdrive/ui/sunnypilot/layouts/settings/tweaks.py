"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntEnum

from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.dynamic_follow_settings import DynamicFollowSettingsLayout
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, option_item_sp, simple_button_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller_tici import Scroller


class PanelType(IntEnum):
  TWEAKS = 0
  DYNAMIC_FOLLOW = 1


class TweaksLayout(Widget):
  def __init__(self):
    super().__init__()

    self._current_panel = PanelType.TWEAKS
    self._dynamic_follow_layout = DynamicFollowSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    # Secure-on-exit (doorlockd). Toyota/Lexus only.
    self._lock_doors_timer = option_item_sp(
      title=lambda: tr("Auto Lock After Exit"),
      param="LockDoorsTimer",
      description=lambda: tr("Lock the doors once you leave the car: no face seen in the driver-monitoring " +
                            "camera, all doors closed, and the ignition off for this many seconds. " +
                            "Set to Off to disable. Toyota/Lexus only."),
      min_value=0,
      max_value=180,
      value_change_step=2,
      label_callback=lambda value: tr("Off") if value == 0 else f"{value} s",
      inline=True,
    )
    self._fold_mirrors = toggle_item_sp(
      title=lambda: tr("Fold Mirrors On Exit"),
      description=lambda: tr("Also fold the side mirrors when the car is secured. Toyota/Lexus only."),
      param="FoldMirrors",
    )
    self._close_windows = toggle_item_sp(
      title=lambda: tr("Close Windows On Exit"),
      description=lambda: tr("Also roll up the windows when the car is secured. Toyota/Lexus only."),
      param="CloseWindows",
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

    items = [
      self._lock_doors_timer,
      self._fold_mirrors,
      self._close_windows,
      self._dynamic_follow,
      self._dynamic_follow_button,
    ]
    return items

  def _render(self, rect):
    if self._current_panel == PanelType.DYNAMIC_FOLLOW:
      self._dynamic_follow_layout.render(rect)
    else:
      self._scroller.render(rect)

  def show_event(self):
    self._set_current_panel(PanelType.TWEAKS)
    self._scroller.show_event()

  def _set_current_panel(self, panel: PanelType):
    self._current_panel = panel
    if panel == PanelType.DYNAMIC_FOLLOW:
      self._dynamic_follow_layout.show_event()

  def _update_state(self):
    super()._update_state()

    # the mirror / window options only do anything once auto-lock is enabled
    secure_enabled = self._lock_doors_timer.action_item.current_value > 0
    self._fold_mirrors.action_item.set_enabled(secure_enabled)
    self._close_windows.action_item.set_enabled(secure_enabled)

    # the Manage button is only available once dynamic follow is enabled
    dynamic_follow_enabled = self._dynamic_follow.action_item.get_state()
    self._dynamic_follow_button.action_item.set_enabled(dynamic_follow_enabled)
