"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from collections.abc import Callable

import pyray as rl
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import multiple_button_item_sp, option_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller


class ParkAssistSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()

    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._park_distance = option_item_sp(
      title=lambda: tr("Standstill Gap"),
      param="ParkDistance",
      description=lambda: tr("How close to stop behind a stopped lead. Only applies near a standstill and " +
                            "smoothly restores the normal gap once the lead starts moving."),
      min_value=100,
      max_value=300,
      value_change_step=10,
      label_callback=lambda value: f"{value / 100:.2f} m",
      inline=True,
    )

    self._park_mode = multiple_button_item_sp(
      title=lambda: tr("Engage When"),
      description=lambda: tr("From Full Stop: only after you stop behind a stopped lead, then holds through launch. " +
                            "Any Low Speed: also applies the closer gap while following at low speed."),
      buttons=[lambda: tr("From Full Stop"), lambda: tr("Any Low Speed")],
      param="ParkAssistMode",
      button_width=420,
      inline=False,
    )

    return [
      self._park_distance,
      self._park_mode,
    ]

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
