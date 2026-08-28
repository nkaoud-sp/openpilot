"""Developer CAN test panel: fire a single hardcoded CAN frame on demand."""
from collections.abc import Callable

import pyray as rl
from openpilot.common.params import Params
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import simple_button_item_sp
from openpilot.system.ui.widgets.list_view import text_item
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller


class CanTestSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()

    self._params = Params()
    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._send_button = simple_button_item_sp(
      button_text=lambda: tr("Send Test CAN Frame"),
      button_width=800,
      callback=self._on_send,
    )
    return [
      text_item(
        title=lambda: tr("CAN Frame"),
        value=lambda: "0x750 · bus 0",
        description=lambda: tr("Sends 40 05 30 11 00 80 00 00 to address 0x750 on bus 0 once per press. Works " +
                              "offroad: pandad briefly switches the panda to ELM327 diagnostic mode, sends the " +
                              "frame, then reverts. Requires a panda connected to a live CAN bus."),
      ),
      self._send_button,
    ]

  def _on_send(self):
    # card.py reads this trigger each control cycle, injects the frame, and clears it.
    self._params.put_bool("CanTestTrigger", True)

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
