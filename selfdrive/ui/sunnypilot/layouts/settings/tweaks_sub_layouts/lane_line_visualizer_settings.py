"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Settings submenu for the Lane Line Visualizer: a standalone solid-vs-broken
lane-line classifier (lane_line_classifierd) with an on-road debug readout.
"""
from collections.abc import Callable

import pyray as rl
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, option_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller


class LaneLineVisualizerSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()

    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._readout = toggle_item_sp(
      title=lambda: tr("Show Classifier Readout"),
      description=lambda: tr("Draw an on-road debug panel with each ego lane line's classification "
                            "(solid / broken / unknown), duty cycle, estimated dash period, confidence, "
                            "and the debounced crossable flags. Requires the Lane Line Visualizer master "
                            "toggle to be on so the classifier process runs."),
      param="LaneLineVisualizerReadout",
    )
    self._overlay = toggle_item_sp(
      title=lambda: tr("Show Lane Line Overlay"),
      description=lambda: tr("Recolor the two ego lane lines on the driving view by classification: "
                            "red = solid (not crossable), green = broken/dashed (crossable), gray = unknown. "
                            "Opacity scales with confidence."),
      param="LaneLineVisualizerOverlay",
    )

    # --- Tuning. These apply live (~1 s) to the classifier process. Watch the
    # readout's duty / period / confidence numbers to know which to move. ---
    self._min_contrast = option_item_sp(
      title=lambda: tr("Min Contrast"),
      description=lambda: tr("How much brighter than the road a marking must be to count as present "
                            "(luminance counts). Lower for faded paint; raise if glare causes false positives."),
      param="LaneLineMinContrast", min_value=5, max_value=60,
      label_callback=lambda v: str(v), inline=False,
    )
    self._solid_duty = option_item_sp(
      title=lambda: tr("Solid Duty"),
      description=lambda: tr("Fraction of the line that must read as painted to call it SOLID."),
      param="LaneLineSolidDuty", min_value=50, max_value=99,
      label_callback=lambda v: f"{v}%", inline=False,
    )
    self._min_autocorr = option_item_sp(
      title=lambda: tr("Min Periodicity"),
      description=lambda: tr("How clean the dash on/off rhythm must be to call a line BROKEN. "
                            "Raise if solid lines get misread as broken."),
      param="LaneLineMinAutocorr", min_value=10, max_value=90,
      label_callback=lambda v: f"{v}%", inline=False,
    )
    self._min_period = option_item_sp(
      title=lambda: tr("Min Dash Period"),
      description=lambda: tr("Shortest plausible dash cycle (paint + gap), in metres."),
      param="LaneLineMinPeriodM", min_value=1, max_value=10,
      label_callback=lambda v: f"{v} m", inline=False,
    )
    self._max_period = option_item_sp(
      title=lambda: tr("Max Dash Period"),
      description=lambda: tr("Longest plausible dash cycle (paint + gap), in metres. US highways ~12 m."),
      param="LaneLineMaxPeriodM", min_value=12, max_value=60,
      label_callback=lambda v: f"{v} m", inline=False,
    )
    self._sample_max = option_item_sp(
      title=lambda: tr("Classify Distance"),
      description=lambda: tr("How far ahead to classify, in metres. Lower it if the far field is noisy."),
      param="LaneLineSampleMaxM", min_value=20, max_value=100, value_change_step=5,
      label_callback=lambda v: f"{v} m", inline=False,
    )
    return [
      self._readout,
      self._overlay,
      self._min_contrast,
      self._solid_duty,
      self._min_autocorr,
      self._min_period,
      self._max_period,
      self._sample_max,
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
