"""Large on-road UI/debug readout for the standalone visual vehicle detector."""
import json
import time
from pathlib import Path

import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached

STATE_PATH = Path("/tmp/nkaoud_visual_vehicle_detector.json")
STALE_AFTER_S = 1.0

_GREEN = rl.Color(0, 220, 110, 255)
_AMBER = rl.Color(255, 180, 0, 255)
_RED = rl.Color(255, 70, 70, 255)
_WHITE = rl.Color(255, 255, 255, 255)
_DIM = rl.Color(190, 190, 190, 255)
_BG = rl.Color(0, 0, 0, 175)


class VisualVehicleReadout:
  def __init__(self):
    self._alpha = 0.0
    self._last_read_t = 0.0
    self._state = {"left": False, "right": False, "monotonic_time": 0.0, "debug": {"reason": "not_started"}}
    self._title_font = gui_app.font(FontWeight.BOLD)
    self._cap_font = gui_app.font(FontWeight.SEMI_BOLD)
    self._val_font = gui_app.font(FontWeight.BOLD)

  def _update_alpha(self, visible: bool):
    if visible:
      self._alpha = min(1.0, self._alpha + 0.1)
    else:
      self._alpha = max(0.0, self._alpha - 0.05)

  def _read_state(self):
    now = time.monotonic()
    if now - self._last_read_t < 0.1:
      return self._state
    self._last_read_t = now
    try:
      self._state = json.loads(STATE_PATH.read_text())
    except Exception:
      self._state = {"left": False, "right": False, "monotonic_time": 0.0, "debug": {"reason": "state_missing"}}
    return self._state

  @staticmethod
  def _status_color(active: bool, stale: bool, reason: str) -> rl.Color:
    if stale or reason not in ("ok",):
      return _AMBER if reason in ("state_missing", "inactive", "waiting_for_camera", "waiting_for_vipc", "pkl_missing") else _RED
    return _RED if active else _GREEN

  def draw(self, rect: rl.Rectangle):
    visible = bool(getattr(ui_state, "visual_vehicle_detector_readout", False))
    self._update_alpha(visible)
    if self._alpha <= 0.0:
      return

    state = self._read_state()
    debug = state.get("debug", {}) or {}
    age = max(0.0, time.monotonic() - float(state.get("monotonic_time", 0.0) or 0.0))
    stale = age > STALE_AFTER_S
    reason = str(debug.get("reason", "unknown"))
    runtime = str(debug.get("runtime", "--"))
    input_shape = debug.get("input_shape", [])
    if isinstance(input_shape, list) and len(input_shape) >= 4:
      input_shape_text = f"{input_shape[3]}x{input_shape[2]}"
    elif isinstance(input_shape, list) and len(input_shape) >= 2:
      input_shape_text = f"{input_shape[-1]}x{input_shape[-2]}"
    else:
      input_shape_text = "--"
    left = bool(state.get("left", False))
    right = bool(state.get("right", False))

    rows = [
      ("LEFT", "VEHICLE" if left else "CLEAR", self._status_color(left, stale, reason)),
      ("RIGHT", "VEHICLE" if right else "CLEAR", self._status_color(right, stale, reason)),
      ("STATUS", "STALE" if stale else reason.upper(), _AMBER if stale or reason != "ok" else _GREEN),
      ("RUNTIME", runtime.upper(), _GREEN if runtime == "tinygrad_pkl" else (_AMBER if runtime == "onnx_cpu" else _DIM)),
      ("INPUT", input_shape_text, _WHITE),
      ("AGE", f"{age:.1f}s", _AMBER if stale else _WHITE),
      ("DETS", str(debug.get("detections", "--")), _WHITE),
      ("BEST", str(debug.get("best_conf", "--")), _WHITE),
      ("SCORES", f"L{debug.get('left_score', '--')} / R{debug.get('right_score', '--')}", _WHITE),
      ("FRAME", str(debug.get("frame_id", "--")), _DIM),
    ]
    self._render(rect, rows)

  def _render(self, rect: rl.Rectangle, rows):
    a = self._alpha
    title_size = 34
    cap_size = 23
    val_size = 38
    pad = 22
    row_gap = 10
    col_gap = 28

    def fade(c: rl.Color) -> rl.Color:
      return rl.Color(c.r, c.g, c.b, int(c.a * a))

    title = "VISUAL VEHICLE DETECTOR"
    title_w = measure_text_cached(self._title_font, title, title_size, 0).x
    cap_w = max(measure_text_cached(self._cap_font, cap, cap_size, 0).x for cap, _, _ in rows)
    val_w = max(measure_text_cached(self._val_font, val, val_size, 0).x for _, val, _ in rows)

    row_h = max(cap_size, val_size)
    content_w = max(title_w, cap_w + col_gap + val_w)
    panel_w = pad + content_w + pad
    panel_h = pad + title_size + 16 + len(rows) * row_h + (len(rows) - 1) * row_gap + pad

    x = rect.x + rect.width - panel_w - 36
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
