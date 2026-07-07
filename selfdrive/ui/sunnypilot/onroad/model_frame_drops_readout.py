"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.text_measure import measure_text_cached

_GREEN = rl.Color(0, 200, 90, 255)
_AMBER = rl.Color(255, 180, 0, 255)
_ORANGE = rl.Color(255, 115, 0, 255)
_RED = rl.Color(255, 70, 70, 255)
_DIM = rl.Color(180, 180, 180, 255)


class ModelFrameDropsReadout:
  def __init__(self):
    self._alpha: float = 0.0
    self._cap_font = gui_app.font(FontWeight.SEMI_BOLD)
    self._val_font = gui_app.font(FontWeight.BOLD)

  def _update_alpha(self, visible: bool):
    if visible:
      self._alpha = min(1.0, self._alpha + 0.1)
    else:
      self._alpha = max(0.0, self._alpha - 0.05)

  @staticmethod
  def _drop_color(drop_perc: float) -> rl.Color:
    if drop_perc >= 10.0:
      return _RED
    if drop_perc >= 6.0:
      return _ORANGE
    if drop_perc >= 2.0:
      return _AMBER
    return _GREEN

  @staticmethod
  def _drop_cell(sm, service: str, label: str) -> tuple[str, str, rl.Color]:
    if not sm.alive[service] or sm.recv_frame[service] < ui_state.started_frame:
      return label, "--", _DIM

    drop_perc = float(sm[service].frameDropPerc)
    return label, f"{drop_perc:.1f}%", ModelFrameDropsReadout._drop_color(drop_perc)

  def draw(self, sm, rect: rl.Rectangle):
    visible = ui_state.model_frame_drops_readout
    self._update_alpha(visible)
    if self._alpha <= 0.0 or not visible:
      return

    cells = [
      self._drop_cell(sm, "modelV2", "MODEL V2"),
      self._drop_cell(sm, "drivingModelData", "DRIVING"),
    ]
    self._render(rect, cells)

  def _render(self, rect: rl.Rectangle, cells: list[tuple[str, str, rl.Color]]):
    a = self._alpha
    cap_size = 22
    val_size = 36
    pad_x = 18
    pad_y = 14
    cell_gap = 24
    cap_val_gap = 4

    def fade(c: rl.Color) -> rl.Color:
      return rl.Color(c.r, c.g, c.b, int(255 * a))

    cell_w = 0.0
    for cap, val, _c in cells:
      cell_w = max(
        cell_w,
        measure_text_cached(self._cap_font, cap, cap_size, 0).x,
        measure_text_cached(self._val_font, val, val_size, 0).x,
      )

    row_h = cap_size + cap_val_gap + val_size
    content_w = len(cells) * cell_w + (len(cells) - 1) * cell_gap
    panel_w = pad_x + content_w + pad_x
    panel_h = pad_y + row_h + pad_y

    x = rect.x + (rect.width - panel_w) / 2
    y = rect.y + rect.height * 0.63

    rl.draw_rectangle_rounded(rl.Rectangle(x, y, panel_w, panel_h), 0.18, 10, rl.Color(0, 0, 0, int(120 * a)))

    row_y = y + pad_y
    for idx, (cap, val, color) in enumerate(cells):
      cx = x + pad_x + idx * (cell_w + cell_gap)

      cap_w = measure_text_cached(self._cap_font, cap, cap_size, 0).x
      rl.draw_text_ex(
        self._cap_font,
        cap,
        rl.Vector2(int(cx + (cell_w - cap_w) / 2), int(row_y)),
        cap_size,
        0,
        fade(_DIM),
      )

      val_w = measure_text_cached(self._val_font, val, val_size, 0).x
      rl.draw_text_ex(
        self._val_font,
        val,
        rl.Vector2(int(cx + (cell_w - val_w) / 2), int(row_y + cap_size + cap_val_gap)),
        val_size,
        0,
        fade(color),
      )
