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

  @staticmethod
  def _timing_breakdown(timing: dict) -> str:
    """Per-stage ms: C=crop->RGB, P=preprocess, I=inference, W=state write."""
    return "C{} P{} I{} W{}".format(
      timing.get("crop_rgb_ms", "--"),
      timing.get("preprocess_ms", "--"),
      timing.get("infer_ms", "--"),
      timing.get("state_write_ms", "--"),
    )

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
    timing = debug.get("timing", {}) or {}
    input_shape = debug.get("input_shape", [])
    if isinstance(input_shape, list) and len(input_shape) >= 4:
      input_shape_text = f"{input_shape[3]}x{input_shape[2]}"
    elif isinstance(input_shape, list) and len(input_shape) >= 2:
      input_shape_text = f"{input_shape[-1]}x{input_shape[-2]}"
    else:
      input_shape_text = "--"
    left = bool(state.get("left", False))
    right = bool(state.get("right", False))

    capture = debug.get("capture", {}) or {}
    cap_on = bool(capture.get("on"))
    classifier = debug.get("classifier", {}) or {}

    if debug.get("dual"):
      # wide+driver: one combined panel, both cameras' zones (WIDE-L/R, DM-L/R).
      cameras = debug.get("cameras", {}) or {}

      def _zcolor(blk: bool) -> rl.Color:
        if stale or reason != "ok":
          return _AMBER
        return _RED if blk else _GREEN

      def _pp(v) -> str:
        return f"{v:.2f}" if isinstance(v, (int, float)) else "--"

      labels = {"wide": "WIDE", "driver": "DM"}
      rows = [
        ("MODE", "WIDE+DRIVER", _WHITE),
        ("CAPTURE", (f"REC {capture.get('saved', 0)}" if cap_on else "OFF"), _RED if cap_on else _DIM),
      ]
      probs = []
      for cam in ("wide", "driver"):
        for z in (cameras.get(cam, {}) or {}).get("zones", []) or []:
          zn = str(z.get("name", "?"))
          label = labels.get(cam, cam.upper()) + ("" if zn == "center" else f"-{zn[:1].upper()}")
          rows.append((label, ("BLOCKED" if z.get("blocked") else "CLEAR"), _zcolor(bool(z.get("blocked")))))
          probs.append(_pp(z.get("p")))
      rows.append(("P", " ".join(probs) or "--", _WHITE))
      rows.extend([
        ("EVAL", str(debug.get("side", "--")).upper(), _DIM),
        ("STATUS", "STALE" if stale else reason.upper(), _AMBER if stale or reason != "ok" else _GREEN),
        ("RUNTIME", runtime.upper(), _GREEN if runtime == "tinygrad_pkl" else (_AMBER if runtime == "onnx_cpu" else _DIM)),
        ("RATE", f"{timing.get('measured_hz', '--')} / {debug.get('hz', '--')} Hz", _WHITE),
        ("TIMING", self._timing_breakdown(timing), _WHITE),
        ("AGE", f"{age:.1f}s", _AMBER if stale else _WHITE),
        ("FRAME", str(debug.get("frame_id", "--")), _DIM),
      ])
      self._render_panel(rect, rows, "CAR OCCUPANCY (DUAL)", side="right")
      return

    if classifier.get("active") or reason in ("classifier_missing", "classifier_error"):
      # Car classifier: per-zone occupancy. Zones come from the model recipe --
      # left/right (driver, alternating) or a single 'center' (wide).
      def _side_color(blk: bool) -> rl.Color:
        if stale or reason != "ok":
          return _AMBER
        return _RED if blk else _GREEN

      def _p(v) -> str:
        return f"{v:.2f}" if isinstance(v, (int, float)) else "--"

      zones = classifier.get("zones") or []
      rows = [
        ("CAMERA", str(debug.get("camera", "--")).upper(), _WHITE),
        ("CAPTURE", (f"REC {capture.get('saved', 0)}" if cap_on else "OFF"), _RED if cap_on else _DIM),
      ]
      for z in zones:
        label = "CAR" if z.get("name") == "center" else str(z.get("name", "?")).upper()
        blk = bool(z.get("blocked"))
        rows.append((label, ("BLOCKED" if blk else "CLEAR"), _side_color(blk)))
      rows.append(("P", " / ".join(_p(z.get("p")) for z in zones) or "--", _WHITE))
      rows.extend([
        ("EVAL", str(classifier.get("side", "--")).upper(), _DIM),
        ("THRESH", str(classifier.get("threshold", "--")), _DIM),
        ("STATUS", "STALE" if stale else reason.upper(), _AMBER if stale or reason != "ok" else _GREEN),
        ("RUNTIME", runtime.upper(), _GREEN if runtime == "tinygrad_pkl" else (_AMBER if runtime == "onnx_cpu" else _DIM)),
        ("INPUT", input_shape_text, _WHITE),
        ("RATE", f"{timing.get('measured_hz', '--')} / {debug.get('hz', '--')} Hz", _WHITE),
        ("TIMING", self._timing_breakdown(timing), _WHITE),
        ("AGE", f"{age:.1f}s", _AMBER if stale else _WHITE),
        ("FRAME", str(debug.get("frame_id", "--")), _DIM),
      ])
      self._render_panel(rect, rows, "CAR OCCUPANCY", side="right")
      return

    rows = [
      ("CAMERA", str(debug.get("camera", "--")).upper(), _WHITE),
      ("CAPTURE", (f"REC {capture.get('saved', 0)}" if cap_on else "OFF"), _RED if cap_on else _DIM),
      ("LEFT", "VEHICLE" if left else "CLEAR", self._status_color(left, stale, reason)),
      ("RIGHT", "VEHICLE" if right else "CLEAR", self._status_color(right, stale, reason)),
      ("STATUS", "STALE" if stale else reason.upper(), _AMBER if stale or reason != "ok" else _GREEN),
      ("RUNTIME", runtime.upper(), _GREEN if runtime == "tinygrad_pkl" else (_AMBER if runtime == "onnx_cpu" else _DIM)),
      ("INPUT", input_shape_text, _WHITE),
      ("RATE", f"{timing.get('measured_hz', '--')} / {debug.get('hz', '--')} Hz", _WHITE),
      ("INF ms", str(timing.get("infer_ms", "--")), _WHITE),
      ("AGE", f"{age:.1f}s", _AMBER if stale else _WHITE),
      ("DETS", str(debug.get("detections", "--")), _WHITE),
      ("BEST", str(debug.get("best_conf", "--")), _WHITE),
      ("SCORES", f"L{debug.get('left_score', '--')} / R{debug.get('right_score', '--')}", _WHITE),
      ("FRAME", str(debug.get("frame_id", "--")), _DIM),
    ]
    raw_rows = [
      ("PARSER", str(debug.get("parser", "--")).upper(), _WHITE),
      ("OUT", self._format_shape(debug.get("output_shape")), _WHITE),
      ("OBJ", str(debug.get("raw_best_obj", "--")), _WHITE),
      ("CLASS", str(debug.get("raw_best_cls", "--")), _WHITE),
      ("RAWCONF", str(debug.get("raw_best_conf", "--")), _WHITE),
      ("BESTCLS", str(debug.get("raw_best_class_id", "--")), _WHITE),
      ("VEH?", "YES" if debug.get("raw_best_vehicle") else "NO", _WHITE),
      ("ROI", f"L{'Y' if debug.get('raw_best_left_roi') else 'N'}/R{'Y' if debug.get('raw_best_right_roi') else 'N'}", _WHITE),
      ("BOX", self._format_box(debug.get("raw_best_box")), _WHITE),
      ("PKL", "YES" if debug.get("pkl_exists", True) else "NO", _WHITE),
      ("ONNX", "YES" if debug.get("onnx_exists", True) else "NO", _WHITE),
      ("MODEL", str(timing.get("model", "--")), _WHITE),
      ("LOAD ms", str(timing.get("model_load_ms", "--")), _WHITE),
      ("INF1 ms", str(timing.get("first_inf_ms", "--")), _WHITE),
      ("CONN ms", str(timing.get("cam_connect_ms", "--")), _WHITE),
    ]
    self._render_panel(rect, rows, "VISUAL VEHICLE DETECTOR", side="right")
    self._render_panel(rect, raw_rows, "VISUAL RAW DEBUG", side="left")
    self._render_marker(rect, debug)

  @staticmethod
  def _format_shape(shape) -> str:
    if isinstance(shape, list) and shape:
      return "x".join(str(v) for v in shape)
    return "--"

  @staticmethod
  def _format_box(box) -> str:
    if isinstance(box, list) and len(box) == 4:
      return f"{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}"
    return "--"

  def _render_marker(self, rect: rl.Rectangle, debug: dict):
    cx = debug.get("raw_best_center_x")
    cy = debug.get("raw_best_center_y")
    if not isinstance(cx, (int, float)) or not isinstance(cy, (int, float)):
      return

    x = rect.x + float(cx) * rect.width
    y = rect.y + float(cy) * rect.height
    color = _GREEN if debug.get("raw_best_vehicle") else _AMBER
    color = rl.Color(color.r, color.g, color.b, int(color.a * self._alpha))
    rl.draw_circle_lines(int(x), int(y), 18, color)
    rl.draw_line(int(x) - 8, int(y), int(x) + 8, int(y), color)
    rl.draw_line(int(x), int(y) - 8, int(x), int(y) + 8, color)

  def _render_panel(self, rect: rl.Rectangle, rows, title: str, side: str = "right"):
    a = self._alpha
    title_size = 34
    cap_size = 23
    val_size = 38
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

    if side == "left":
      x = rect.x + 36
      y = rect.y + (rect.height - panel_h) / 2
    else:
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
