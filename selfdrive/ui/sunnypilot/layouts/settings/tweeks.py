"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, option_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller_tici import Scroller


class TweeksLayout(Widget):
  def __init__(self):
    super().__init__()

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
      value_change_step=15,
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

    items = [
      self._lock_doors_timer,
      self._fold_mirrors,
      self._close_windows,
    ]
    return items

  def _update_state(self):
    super()._update_state()

    # the mirror / window options only do anything once auto-lock is enabled
    secure_enabled = self._lock_doors_timer.action_item.current_value > 0
    self._fold_mirrors.action_item.set_enabled(secure_enabled)
    self._close_windows.action_item.set_enabled(secure_enabled)

  def _render(self, rect):
    self._scroller.render(rect)

  def show_event(self):
    self._scroller.show_event()
