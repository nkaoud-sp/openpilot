"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from collections.abc import Callable

import pyray as rl
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import multiple_button_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller


class LanePositionSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()

    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    # Edge-lane blocking filter selector. None preserves the base behaviour; the
    # other modes demote "fake" outer lanes (shoulders, gore stripes, off-ramps)
    # so the indicator only counts usable lanes. A 3-frame debounce gates every
    # mode.
    self._edge_filter_mode = multiple_button_item_sp(
      title=lambda: tr("Edge-Lane Filter"),
      description=lambda: tr("Demote fake outer lanes from the lane-position estimate. " +
                            "Width: outer lane narrower than the ego lane (ratio < 0.8). " +
                            "Separation: inner lane line strong while the outer line is weak. " +
                            "Both (AND): conservative — needs both. " +
                            "Both (OR): aggressive — either is enough."),
      buttons=[
        lambda: tr("None"),
        lambda: tr("Width"),
        lambda: tr("Separation"),
        lambda: tr("Both (AND)"),
        lambda: tr("Both (OR)"),
      ],
      param="LaneEdgeFilterMode",
      button_width=260,
      inline=False,
    )

    items = [
      self._edge_filter_mode,
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
