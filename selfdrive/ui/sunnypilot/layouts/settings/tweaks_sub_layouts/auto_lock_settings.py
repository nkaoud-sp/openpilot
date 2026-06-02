"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from collections.abc import Callable

import pyray as rl
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, option_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller


class AutoLockSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()

    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._lock_doors_timer = option_item_sp(
      title=lambda: tr("Auto Lock After Exit"),
      param="LockDoorsTimer",
      description=lambda: tr("Lock the doors this many seconds after you leave the car: no face seen in the " +
                            "driver-monitoring camera, all doors closed, and the ignition off. Toyota/Lexus only."),
      min_value=2,
      max_value=180,
      value_change_step=1,
      label_callback=lambda value: f"{value} s",
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

  def _render(self, rect):
    self._back_button.set_position(self._rect.x, self._rect.y + 20)
    self._back_button.render()

    content_rect = rl.Rectangle(rect.x, rect.y + self._back_button.rect.height + 40,
                                rect.width, rect.height - self._back_button.rect.height - 40)
    self._scroller.render(content_rect)

  def show_event(self):
    self._scroller.show_event()

  def hide_event(self):
    self._scroller.hide_event()
