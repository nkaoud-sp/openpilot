"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from collections.abc import Callable

import pyray as rl
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import option_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller


class DynamicFollowSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()

    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._dynamic_follow_min = option_item_sp(
      title=lambda: tr("Follow Time At 0 km/h"),
      param="DynamicFollowMinTime",
      description=lambda: tr("Follow time used at a standstill / low speed. Shorter means a closer gap."),
      min_value=30,
      max_value=150,
      value_change_step=5,
      label_callback=lambda value: f"{value / 100:.2f} s",
      inline=True,
    )
    self._dynamic_follow_max = option_item_sp(
      title=lambda: tr("Follow Time At 130 km/h"),
      param="DynamicFollowMaxTime",
      description=lambda: tr("Follow time used at highway speed. Longer means a larger gap."),
      min_value=50,
      max_value=250,
      value_change_step=5,
      label_callback=lambda value: f"{value / 100:.2f} s",
      inline=True,
    )
    self._dynamic_follow_curve = option_item_sp(
      title=lambda: tr("Curve Shape"),
      param="DynamicFollowCurve",
      description=lambda: tr("Bend the follow-time curve between the two endpoints. 1.00 is a straight line. " +
                            "Below 1.00 opens the gap earlier; above 1.00 stays tighter until highway speed."),
      min_value=50,
      max_value=300,
      value_change_step=5,
      label_callback=lambda value: f"{value / 100:.2f}",
      inline=True,
    )

    return [
      self._dynamic_follow_min,
      self._dynamic_follow_max,
      self._dynamic_follow_curve,
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
