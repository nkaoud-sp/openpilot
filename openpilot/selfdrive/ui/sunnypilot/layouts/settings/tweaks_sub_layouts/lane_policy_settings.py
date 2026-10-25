from collections.abc import Callable

import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import multiple_button_item_sp, toggle_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller


class LanePolicySettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()

    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)

    def enabled():
      return ui_state.params.get_bool("LanePolicyEnabled")
    self._two_line = toggle_item_sp(
      title=lambda: tr("Two-Line Centering"),
      description=lambda: tr("When both lane lines are clean, steer toward their midpoint. One-line hold builds " +
                            "on this lock, while lead fallback can work independently."),
      param="LanePolicyTwoLine",
      enabled=enabled,
    )
    self._intensity = multiple_button_item_sp(
      title=lambda: tr("Intensity"),
      description=lambda: tr("How firmly and quickly the correction pulls toward center. The hard safety limit " +
                            "is unchanged at every setting."),
      buttons=[lambda: tr("Relaxed"), lambda: tr("Normal"), lambda: tr("Aggressive")],
      param="LanePolicyIntensity",
      button_width=280,
      inline=False,
    )
    self._intensity.action_item.set_enabled(enabled)
    self._turn_centering = toggle_item_sp(
      title=lambda: tr("Turn Centering"),
      description=lambda: tr("During turns, target the geometric midpoint between reliable lane lines " +
                            "instead of following a turn-biased model path."),
      param="TurnCentering",
      enabled=enabled,
    )
    self._turn_centering_strength = multiple_button_item_sp(
      title=lambda: tr("Strength"),
      description=lambda: tr("How quickly and firmly the vehicle returns toward the lane midpoint during a turn."),
      buttons=[lambda: tr("Relaxed"), lambda: tr("Normal"), lambda: tr("Aggressive")],
      param="TurnCenteringStrength",
      button_width=280,
      inline=False,
    )
    self._turn_centering_strength.action_item.set_enabled(lambda: enabled() and ui_state.params.get_bool("TurnCentering"))
    self._one_line_fallback = toggle_item_sp(
      title=lambda: tr("One-Line Fallback"),
      description=lambda: tr("If one lane line briefly disappears, hold center using the remaining line and " +
                            "the learned lane width."),
      param="LanePolicyOneLineFallback",
      enabled=enabled,
    )
    self._e2e_blend = toggle_item_sp(
      title=lambda: tr("E2E Blend Mode"),
      description=lambda: tr("Blend the two-line lane target with the model path shape instead of using only " +
                            "lane geometry."),
      param="LanePolicyE2EBlend",
      enabled=enabled,
    )
    self._lead_fallback = toggle_item_sp(
      title=lambda: tr("Lead Fallback"),
      description=lambda: tr("When lane-line fallback is unavailable, use a detected lead vehicle as a " +
                            "low-authority lateral reference."),
      param="LanePolicyLeadFallback",
      enabled=enabled,
    )
    self._lead_close_range = toggle_item_sp(
      title=lambda: tr("Close Lead Centering"),
      description=lambda: tr("Allow nearby leads, including stop-and-go traffic under 5 m, to drive lead " +
                            "centering. Requires Lead Fallback."),
      param="LanePolicyLeadCloseRange",
      enabled=lambda: enabled() and ui_state.params.get_bool("LanePolicyLeadFallback"),
    )
    self._visual_indicator = toggle_item_sp(
      title=lambda: tr("Colored Lane-Line Indicator"),
      description=lambda: tr("Highlight the lane line the correction steers toward: green for two-line, yellow " +
                            "for one-line, and purple for lead fallback."),
      param="LanePolicyVisualIndicator",
      enabled=enabled,
    )

    self._scroller = Scroller([
      self._two_line,
      self._intensity,
      self._turn_centering,
      self._turn_centering_strength,
      self._one_line_fallback,
      self._e2e_blend,
      self._lead_fallback,
      self._lead_close_range,
      self._visual_indicator,
    ], line_separator=True, spacing=0)

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
