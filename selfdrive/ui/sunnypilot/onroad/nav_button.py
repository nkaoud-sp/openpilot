"""
Onroad NAV button for the experimental nkaoud_nav.

Tapping it pushes a MultiOptionDialog with the hard-coded presets
(Home / Work / School + Clear). The selection is written to the
NkaoudNavDestination param as JSON; nkaoud_navd reads that param to
decide what to route to.

Only visible when NkaoudNavEnabled is set.
"""
from __future__ import annotations

import json
import pyray as rl
from openpilot.common.params import Params
from openpilot.sunnypilot.nkaoud_nav.destinations import PRESETS
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.option_dialog import MultiOptionDialog


CLEAR_LABEL = "Clear destination"


class NavButton(Widget):
  def __init__(self, button_size: int):
    super().__init__()
    self._params = Params()
    self._rect = rl.Rectangle(0, 0, button_size, button_size)
    self._black_bg = rl.Color(0, 0, 0, 166)
    self._white = rl.Color(255, 255, 255, 255)
    self._font = gui_app.font(FontWeight.BOLD)
    self._enabled = False
    self._has_destination = False

  def set_rect(self, rect: rl.Rectangle) -> None:
    self._rect.x, self._rect.y = rect.x, rect.y

  def _update_state(self) -> None:
    self._enabled = self._params.get_bool("NkaoudNavEnabled")
    dest_raw = self._params.get("NkaoudNavDestination")
    self._has_destination = bool(dest_raw)

  def _handle_mouse_release(self, _):
    super()._handle_mouse_release(_)
    if not self._enabled:
      return
    self._open_picker()

  def _open_picker(self) -> None:
    options = [d.label for d in PRESETS]
    if self._has_destination:
      options.append(CLEAR_LABEL)

    dialog = MultiOptionDialog(tr("Select destination"), options, current="",
                               callback=self._on_picker_result)
    gui_app.push_widget(dialog)
    self._picker_ref = dialog  # keep alive

  def _on_picker_result(self, result: DialogResult) -> None:
    if result != DialogResult.CONFIRM:
      return
    selection = self._picker_ref.selection
    if selection == CLEAR_LABEL:
      self._params.remove("NkaoudNavDestination")
      return
    for preset in PRESETS:
      if preset.label == selection:
        self._params.put("NkaoudNavDestination", json.dumps(preset.as_dict()))
        return

  def _render(self, rect: rl.Rectangle) -> None:
    if not self._enabled:
      return

    cx = int(self._rect.x + self._rect.width // 2)
    cy = int(self._rect.y + self._rect.height // 2)
    self._white.a = 180 if self.is_pressed else 255

    rl.draw_circle(cx, cy, self._rect.width / 2, self._black_bg)

    label = "NAV"
    font_size = 56
    text_w = measure_text_cached(self._font, label, font_size).x
    rl.draw_text_ex(self._font, label,
                    rl.Vector2(cx - text_w / 2, cy - font_size / 2),
                    font_size, 0, self._white)

    if self._has_destination:
      dot_color = rl.Color(0x80, 0xd8, 0xa6, 255)
      rl.draw_circle(cx, int(cy + font_size / 2 + 18), 8, dot_color)
