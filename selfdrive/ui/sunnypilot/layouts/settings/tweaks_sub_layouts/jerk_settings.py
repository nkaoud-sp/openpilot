"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from collections.abc import Callable

import pyray as rl
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import option_item_sp, toggle_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller


class JerkSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()

    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._jerk_accel = option_item_sp(
      title=lambda: tr("Acceleration Smoothness"),
      param="JerkFactorAccel",
      description=lambda: tr("How gently acceleration builds. Higher is smoother and slower-building; " +
                            "lower is snappier and more immediate. 1.00x matches stock."),
      min_value=50,
      max_value=300,
      value_change_step=10,
      label_callback=lambda value: f"{value / 100:.2f}x",
      inline=True,
    )
    self._jerk_decel = option_item_sp(
      title=lambda: tr("Braking Smoothness"),
      param="JerkFactorDecel",
      description=lambda: tr("How gently braking builds. Higher is smoother and slower-building; " +
                            "lower is snappier and more immediate. 1.00x matches stock."),
      min_value=50,
      max_value=300,
      value_change_step=10,
      label_callback=lambda value: f"{value / 100:.2f}x",
      inline=True,
    )

    self._jerk_readout = toggle_item_sp(
      title=lambda: tr("Accel / Decel Readout"),
      description=lambda: tr("Display the current accel/decel mode, live acceleration, and the active " +
                            "smoothness factor on the driving screen. Useful for tuning."),
      param="JerkReadout",
    )

    items = [
      self._jerk_accel,
      self._jerk_decel,
      self._jerk_readout,
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
