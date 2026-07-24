"""
Experimental longitudinal speed-assist tuning.
"""
from collections.abc import Callable

import pyray as rl
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import multiple_button_item_sp, option_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller


class SpeedAssistSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()

    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._mode = multiple_button_item_sp(
      title=lambda: tr("Mode"),
      description=lambda: tr("Readout shows when speed assist would apply without changing acceleration. On adds " +
                            "a small, smoothed acceleration nudge only in clean experimental open-road cases."),
      buttons=[lambda: tr("Off"), lambda: tr("Readout"), lambda: tr("On")],
      param="ExperimentalSpeedAssistMode",
      button_width=240,
      inline=False,
    )
    self._strength = multiple_button_item_sp(
      title=lambda: tr("Strength"),
      description=lambda: tr("Maximum added acceleration once far below the set speed: Low +0.15, Medium +0.25, " +
                            "High +0.35 m/s2. The nudge ramps down as you approach the target."),
      buttons=[lambda: tr("Low"), lambda: tr("Medium"), lambda: tr("High")],
      param="ExperimentalSpeedAssistStrength",
      button_width=250,
      inline=False,
    )
    self._min_speed = option_item_sp(
      title=lambda: tr("Minimum Speed"),
      description=lambda: tr("Speed assist is blocked below this speed."),
      param="ExperimentalSpeedAssistMinKph",
      min_value=30,
      max_value=100,
      value_change_step=5,
      label_callback=lambda value: f"{value} kph",
      inline=True,
    )
    self._max_speed = option_item_sp(
      title=lambda: tr("Maximum Speed"),
      description=lambda: tr("Speed assist is blocked above this speed."),
      param="ExperimentalSpeedAssistMaxKph",
      min_value=90,
      max_value=150,
      value_change_step=5,
      label_callback=lambda value: f"{value} kph",
      inline=True,
    )
    self._start_gap = option_item_sp(
      title=lambda: tr("Assist Starts Below Target"),
      description=lambda: tr("How far below the cruise set speed the nudge can begin. The boost ramps smoothly " +
                            "from a tiny assist here to full strength around 30 kph below target."),
      param="ExperimentalSpeedAssistStartGapKph",
      min_value=5,
      max_value=30,
      value_change_step=1,
      label_callback=lambda value: f"{value} kph",
      inline=True,
    )

    return [
      self._mode,
      self._strength,
      self._min_speed,
      self._max_speed,
      self._start_gap,
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
