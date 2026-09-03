"""Toyota/Lexus turn signal diagnostic test controls."""
from collections.abc import Callable
import time

import pyray as rl
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import button_item_sp, multiple_button_item_sp, option_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller


SIGNALS = ("left", "right", "hazard")
SIGNAL_LABELS = (lambda: tr("Left"), lambda: tr("Right"), lambda: tr("Hazard"))


def _toyota_available() -> bool:
  return ui_state.CP is not None and ui_state.CP.brand == "toyota"


class TurnSignalTestSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()

    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)
    self._selected_signal = 0
    self._status = ""

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._direction = multiple_button_item_sp(
      title=lambda: tr("Signal"),
      description=lambda: tr("Select which exterior turn signal active test to run."),
      buttons=SIGNAL_LABELS,
      selected_index=self._selected_signal,
      button_width=250,
      callback=self._set_signal,
      inline=False,
    )
    self._duration = option_item_sp(
      title=lambda: tr("Duration"),
      description=lambda: tr("Approximate time to hold the selected signal before returning control to the car."),
      param="ToyotaTurnSignalTestDurationMs",
      min_value=500,
      max_value=5000,
      value_change_step=250,
      label_callback=lambda value: f"{value / 1000:.2f} s",
      inline=True,
    )
    self._run_test = button_item_sp(
      title=lambda: tr("Run Test"),
      button_text=lambda: tr("RUN"),
      description=lambda: tr("Requests the selected Toyota/Lexus turn signal command through the live car controller path. Onroad only."),
      callback=self._run_turn_signal_test,
      enabled=lambda: not ui_state.is_offroad() and _toyota_available(),
    )
    self._run_test.set_right_value(lambda: self._status)

    return [
      self._direction,
      self._duration,
      self._run_test,
    ]

  def _set_signal(self, selected_signal: int):
    self._selected_signal = selected_signal
    self._status = ""

  def _run_turn_signal_test(self):
    if ui_state.is_offroad() or not _toyota_available():
      self._status = tr("Unavailable")
      return

    signal = SIGNALS[self._selected_signal]
    duration_ms = int(ui_state.params.get("ToyotaTurnSignalTestDurationMs", return_default=True))
    ui_state.params.put("ToyotaTurnSignalTestRequest", {
      "signal": signal,
      "durationMs": duration_ms,
      "requestId": time.monotonic_ns(),
    })
    self._status = tr("Queued")

  def _render(self, rect):
    self._back_button.set_position(self._rect.x, self._rect.y + 20)
    self._back_button.render()

    content_rect = rl.Rectangle(rect.x, rect.y + self._back_button.rect.height + 40,
                                rect.width, rect.height - self._back_button.rect.height - 40)
    self._scroller.render(content_rect)

  def show_event(self):
    self._status = ""
    self._scroller.show_event()

  def hide_event(self):
    self._scroller.hide_event()
