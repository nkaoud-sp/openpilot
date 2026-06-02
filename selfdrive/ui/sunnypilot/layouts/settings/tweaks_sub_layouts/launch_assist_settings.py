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


class LaunchAssistSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()

    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._eagerness = option_item_sp(
      title=lambda: tr("Launch Eagerness"),
      param="LaunchEagerness",
      description=lambda: tr("How quickly to launch once a stopped lead pulls away. Higher launches with less " +
                            "lead movement (sooner); lower waits until the lead is clearly moving. The car still " +
                            "only launches when the gap is actually opening, and never overrides the brake/gas."),
      min_value=1,
      max_value=10,
      value_change_step=1,
      label_callback=lambda value: tr("Level {}").format(value),
      inline=True,
    )
    self._launch_readout = toggle_item_sp(
      title=lambda: tr("Launch Assist Readout"),
      description=lambda: tr("Display the launch-assist state (armed / go), the lead's speed, and the trigger " +
                            "speed on the driving screen while stopped. Useful for tuning."),
      param="LaunchReadout",
    )

    items = [
      self._eagerness,
      self._launch_readout,
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
