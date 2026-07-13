"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

On-road debug readout for the solid-vs-broken lane-line classifier. Reads the
laneLineClassificationSP message published by lane_line_classifierd and draws a
panel with the two ego lane lines' type / duty / period / confidence and the
debounced crossable flags. UI/debug only.
"""
import time

import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached

STALE_AFTER_S = 1.0
# Nudge the panel left off the right edge. ~1 cm on the comma3/3X display.
PANEL_LEFT_SHIFT_PX = 90

_GREEN = rl.Color(0, 220, 110, 255)
_AMBER = rl.Color(255, 180, 0, 255)
_RED = rl.Color(255, 70, 70, 255)
_WHITE = rl.Color(255, 255, 255, 255)
_DIM = rl.Color(190, 190, 190, 255)
_PURPLE = rl.Color(190, 90, 240, 255)
_BG = rl.Color(0, 0, 0, 175)

# LaneLineType (see lane_line_classifier.py): 0=unknown 1=broken 2=solid 3=double
# Solid is purple to match the overlay (red is used for road edges).
_TYPE_NAME = {0: "UNKNOWN", 1: "BROKEN", 2: "SOLID", 3: "DOUBLE"}
_TYPE_COLOR = {0: _DIM, 1: _GREEN, 2: _PURPLE, 3: _PURPLE}


class LaneLineVisualizerReadout:
  def __init__(self):
    self._alpha = 0.0
    self._title_font = gui_app.font(FontWeight.BOLD)
    self._cap_font = gui_app.font(FontWeight.SEMI_BOLD)
    self._val_font = gui_app.font(FontWeight.BOLD)

  def _update_alpha(self, visible: bool):
    if visible:
      self._alpha = min(1.0, self._alpha + 0.1)
    else:
      self._alpha = max(0.0, self._alpha - 0.05)

  @staticmethod
  def _line_summary(line) -> str:
    t = _TYPE_NAME.get(int(line.lineType), "?")
    period = f"{line.periodM:4.1f}" if line.periodM > 0 else "  -- "
    return f"{t:7s} d={line.duty:.2f} p={period} c={line.confidence:.2f}"

  def draw(self, rect: rl.Rectangle):
    visible = bool(getattr(ui_state, "lane_line_visualizer_readout", False))
    self._update_alpha(visible)
    if self._alpha <= 0.0:
      return

    sm = ui_state.sm
    have = sm.recv_frame.get("laneLineClassificationSP", 0) > 0 and sm.valid.get("laneLineClassificationSP", False)
    if not have:
      self._render_panel(rect, [("STATUS", "NO SIGNAL", _AMBER)], "LANE LINES")
      return

    msg = sm["laneLineClassificationSP"]
    age = max(0.0, time.monotonic() - float(msg.monotonicTime or 0.0))
    stale = age > STALE_AFTER_S
    reason = str(msg.reason)
    ok = msg.valid and reason == "ok" and not stale

    left_c = _GREEN if msg.leftCrossable else _RED
    right_c = _GREEN if msg.rightCrossable else _RED

    rows = [
      ("LEFT", self._line_summary(msg.left), _TYPE_COLOR.get(int(msg.left.lineType), _DIM)),
      ("RIGHT", self._line_summary(msg.right), _TYPE_COLOR.get(int(msg.right.lineType), _DIM)),
      ("CROSS", f"L={int(msg.leftCrossable)}  R={int(msg.rightCrossable)}", left_c if msg.leftCrossable == msg.rightCrossable else _WHITE),
      ("STATUS", "STALE" if stale else reason.upper(), _GREEN if ok else _AMBER),
      ("RATE", f"{msg.hz:.1f} Hz", _WHITE),
      ("AGE", f"{age:.1f}s", _AMBER if stale else _WHITE),
      ("FRAME", str(msg.frameId), _DIM),
    ]
    # subtle: recolor CROSS row per-side via the two flags is folded into the label
    _ = (left_c, right_c)
    self._render_panel(rect, rows, "LANE LINES")

  def _render_panel(self, rect: rl.Rectangle, rows, title: str):
    a = self._alpha
    title_size = 34
    cap_size = 23
    val_size = 34
    pad = 22
    row_gap = 10
    col_gap = 28

    def fade(c: rl.Color) -> rl.Color:
      return rl.Color(c.r, c.g, c.b, int(c.a * a))

    title_w = measure_text_cached(self._title_font, title, title_size, 0).x
    cap_w = max(measure_text_cached(self._cap_font, cap, cap_size, 0).x for cap, _, _ in rows)
    val_w = max(measure_text_cached(self._val_font, val, val_size, 0).x for _, val, _ in rows)

    row_h = max(cap_size, val_size)
    content_w = max(title_w, cap_w + col_gap + val_w)
    panel_w = pad + content_w + pad
    panel_h = pad + title_size + 16 + len(rows) * row_h + (len(rows) - 1) * row_gap + pad

    x = rect.x + rect.width - panel_w - 36 - PANEL_LEFT_SHIFT_PX
    y = rect.y + 120

    rl.draw_rectangle_rounded(rl.Rectangle(x, y, panel_w, panel_h), 0.16, 10, fade(_BG))
    rl.draw_rectangle_rounded_lines_ex(rl.Rectangle(x, y, panel_w, panel_h), 0.16, 10, 3, fade(_DIM))
    rl.draw_text_ex(self._title_font, title, rl.Vector2(int(x + pad), int(y + pad)), title_size, 0, fade(_WHITE))

    start_y = y + pad + title_size + 16
    for i, (cap, val, color) in enumerate(rows):
      row_y = start_y + i * (row_h + row_gap)
      cap_y = row_y + (row_h - cap_size) / 2
      val_y = row_y + (row_h - val_size) / 2
      rl.draw_text_ex(self._cap_font, cap, rl.Vector2(int(x + pad), int(cap_y)), cap_size, 0, fade(_DIM))
      rl.draw_text_ex(self._val_font, val, rl.Vector2(int(x + pad + cap_w + col_gap), int(val_y)), val_size, 0, fade(color))
