"""Large on-road UI/debug readout for the standalone visual vehicle detector."""
import time

import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached

STALE_AFTER_S = 1.0

_GREEN = rl.Color(0, 220, 110, 255)
_AMBER = rl.Color(255, 180, 0, 255)
_RED = rl.Color(255, 70, 70, 255)
_WHITE = rl.Color(255, 255, 255, 255)
_DIM = rl.Color(190, 190, 190, 255)
_BG = rl.Color(0, 0, 0, 175)
_SOFT_GREY = rl.Color(216, 216, 216, 210)
_MID_GREY = rl.Color(148, 148, 148, 220)
_DARK_GREY = rl.Color(78, 78, 78, 220)
_CAR_FILL = rl.Color(224, 229, 232, 255)
_CAR_EDGE = rl.Color(86, 94, 102, 255)
_GLASS = rl.Color(30, 36, 42, 225)


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

    sm = ui_state.sm
    if sm.recv_frame.get("visualVehicleDetectorStateSP", 0) <= 0 or not sm.valid.get("visualVehicleDetectorStateSP", False):
      self._state = {"left": False, "right": False, "monotonic_time": 0.0, "debug": {"reason": "state_missing"}}
      return self._state

    try:
      self._state = self._message_to_state(sm["visualVehicleDetectorStateSP"])
    except Exception:
      self._state = {"left": False, "right": False, "monotonic_time": 0.0, "debug": {"reason": "state_decode_failed"}}
    return self._state

  @staticmethod
  def _zones_to_dict(zones) -> list[dict]:
    out = []
    for zone in zones:
      out.append({
        "name": str(zone.name),
        "blocked": bool(zone.blocked),
        "p": float(zone.probability) if bool(zone.hasProbability) else None,
      })
    return out

  def _message_to_state(self, msg) -> dict:
    debug = {
      "reason": str(msg.reason),
      "runtime": str(msg.runtime),
      "camera": str(msg.camera),
      "side": str(msg.side),
      "hz": float(msg.hz),
      "frame_id": int(msg.frameId),
      "dual": bool(msg.dual),
      "input_shape": list(msg.inputShape),
      "pkl_path": str(msg.pklPath),
      "onnx_path": str(msg.onnxPath),
      "pkl_exists": bool(msg.pklExists),
      "onnx_exists": bool(msg.onnxExists),
      "parser": str(msg.parser),
      "output_shape": list(msg.outputShape),
      "detections": int(msg.detections),
      "best_conf": round(float(msg.bestConf), 3),
      "left_score": int(msg.leftScore),
      "right_score": int(msg.rightScore),
      "timing": {
        "crop_rgb_ms": round(float(msg.timing.cropRgbMs), 1),
        "preprocess_ms": round(float(msg.timing.preprocessMs), 1),
        "infer_ms": round(float(msg.timing.inferMs), 1),
        "state_write_ms": round(float(msg.timing.stateWriteMs), 1),
        "measured_hz": round(float(msg.timing.measuredHz), 1),
        "model_load_ms": round(float(msg.timing.modelLoadMs), 1),
        "first_inf_ms": round(float(msg.timing.firstInfMs), 1),
        "cam_connect_ms": round(float(msg.timing.camConnectMs), 1),
        "model": str(msg.modelName),
      },
      "crop": {
        "crop_x": int(msg.crop.cropX),
        "crop_y": int(msg.crop.cropY),
        "crop_w": int(msg.crop.cropW),
        "crop_h": int(msg.crop.cropH),
        "frame_w": int(msg.crop.frameW),
        "frame_h": int(msg.crop.frameH),
      },
      "capture": {
        "on": bool(msg.capture.on),
        "saved": int(msg.capture.saved),
      },
      "raw_best_obj": round(float(msg.rawBestObj), 5),
      "raw_best_cls": round(float(msg.rawBestCls), 5),
      "raw_best_conf": round(float(msg.rawBestConf), 5),
      "raw_best_class_id": int(msg.rawBestClassId),
      "raw_best_vehicle": bool(msg.rawBestVehicle),
      "raw_best_left_roi": bool(msg.rawBestLeftRoi),
      "raw_best_right_roi": bool(msg.rawBestRightRoi),
      "raw_best_box": list(msg.rawBestBox),
      "error": str(msg.error),
    }

    if bool(msg.rawBestCenterValid):
      debug["raw_best_center_x"] = float(msg.rawBestCenterX)
      debug["raw_best_center_y"] = float(msg.rawBestCenterY)

    classifier_zones = self._zones_to_dict(msg.classifier.zones)
    if bool(msg.classifier.active) or classifier_zones:
      debug["classifier"] = {
        "active": bool(msg.classifier.active),
        "side": str(msg.classifier.side),
        "threshold": round(float(msg.classifier.threshold), 3),
        "left_blocked": bool(msg.classifier.leftBlocked),
        "right_blocked": bool(msg.classifier.rightBlocked),
        "zones": classifier_zones,
      }

    wide_zones = self._zones_to_dict(msg.wideZones)
    driver_zones = self._zones_to_dict(msg.driverZones)
    if bool(msg.dual) or wide_zones or driver_zones:
      debug["cameras"] = {
        "wide": {"zones": wide_zones},
        "driver": {"zones": driver_zones},
      }

    return {
      "left": bool(msg.leftBlocked),
      "right": bool(msg.rightBlocked),
      "monotonic_time": float(msg.monotonicTime),
      "debug": debug,
    }

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
    widget_mode = bool(getattr(ui_state, "visual_vehicle_detector_car_widget", False))

    if widget_mode:
      self._render_car_widget(rect, state, debug, stale, reason)
      return

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

  def _render_car_widget(self, rect: rl.Rectangle, state: dict, debug: dict, stale: bool, reason: str) -> None:
    a = self._alpha
    panel_w = min(540.0, rect.width * 0.42)
    panel_h = min(920.0, rect.height * 0.74)
    x = rect.x + rect.width - panel_w - 56
    y = rect.y + max(110.0, (rect.height - panel_h) * 0.48)
    panel = rl.Rectangle(x, y, panel_w, panel_h)

    def fade(c: rl.Color, alpha_scale: float = 1.0) -> rl.Color:
      return rl.Color(c.r, c.g, c.b, int(c.a * a * alpha_scale))

    rl.draw_rectangle_rounded(panel, 0.14, 12, fade(_BG, 0.92))
    rl.draw_rectangle_rounded_lines_ex(panel, 0.14, 12, 3, fade(_DIM, 0.7))

    status_text = "STALE" if stale else reason.upper()
    status_color = _AMBER if stale or reason != "ok" else _GREEN
    rl.draw_text_ex(self._title_font, "VEHICLE VIEW", rl.Vector2(int(x + 28), int(y + 24)), 30, 0, fade(_WHITE))
    rl.draw_text_ex(self._cap_font, status_text, rl.Vector2(int(x + panel_w - 170), int(y + 28)), 24, 0, fade(status_color))

    zones = self._widget_zones(state, debug)
    center = rl.Vector2(panel.x + panel.width / 2, panel.y + panel.height / 2 + 18)
    ring_outer = min(panel.width * 0.43, panel.height * 0.28)
    ring_inner = ring_outer * 0.52
    gap = 14
    zone_specs = [
      ("front_left", 186, 264),
      ("front_right", 276, 354),
      ("rear_right", 6, 84),
      ("rear_left", 96, 174),
    ]
    base_zone = fade(_SOFT_GREY, 0.78)
    alert_zone = fade(_RED, 0.96)
    for name, start, end in zone_specs:
      color = alert_zone if zones[name] else base_zone
      rl.draw_ring(center, ring_inner, ring_outer, start + gap / 2, end - gap / 2, 40, color)

    self._draw_car_body(center, panel.width * 0.28, panel.height * 0.58, fade)
    self._draw_zone_labels(panel, zones, fade)

    active_count = sum(1 for active in zones.values() if active)
    footer = f"{active_count} BLOCKED" if active_count else "CLEAR"
    footer_color = _RED if active_count else (_AMBER if stale or reason != "ok" else _GREEN)
    rl.draw_text_ex(self._val_font, footer, rl.Vector2(int(x + 28), int(y + panel.height - 60)), 34, 0, fade(footer_color))

  @staticmethod
  def _zone_blocked(zone: dict | None) -> bool:
    return bool((zone or {}).get("blocked", False))

  def _widget_zones(self, state: dict, debug: dict) -> dict[str, bool]:
    zones = {
      "front_left": False,
      "front_right": False,
      "rear_left": False,
      "rear_right": False,
    }
    camera = str(debug.get("camera", "") or "")

    if debug.get("dual"):
      cameras = debug.get("cameras", {}) or {}
      wide_zones = {str(z.get("name")): z for z in (cameras.get("wide", {}) or {}).get("zones", []) or []}
      driver_zones = {str(z.get("name")): z for z in (cameras.get("driver", {}) or {}).get("zones", []) or []}
      zones["front_left"] = self._zone_blocked(wide_zones.get("left"))
      zones["front_right"] = self._zone_blocked(wide_zones.get("right"))
      zones["rear_left"] = self._zone_blocked(driver_zones.get("left"))
      zones["rear_right"] = self._zone_blocked(driver_zones.get("right"))
      return zones

    classifier = debug.get("classifier", {}) or {}
    classifier_zones = {str(z.get("name")): z for z in classifier.get("zones", []) or []}
    if classifier.get("active"):
      if camera == "driver":
        zones["rear_left"] = self._zone_blocked(classifier_zones.get("left"))
        zones["rear_right"] = self._zone_blocked(classifier_zones.get("right"))
        if "center" in classifier_zones:
          blocked = self._zone_blocked(classifier_zones.get("center"))
          zones["rear_left"] = blocked
          zones["rear_right"] = blocked
      else:
        zones["front_left"] = self._zone_blocked(classifier_zones.get("left"))
        zones["front_right"] = self._zone_blocked(classifier_zones.get("right"))
        if "center" in classifier_zones:
          blocked = self._zone_blocked(classifier_zones.get("center"))
          zones["front_left"] = blocked
          zones["front_right"] = blocked
      return zones

    left = bool(state.get("left", False))
    right = bool(state.get("right", False))
    if camera == "driver":
      zones["rear_left"] = left
      zones["rear_right"] = right
    else:
      zones["front_left"] = left
      zones["front_right"] = right
    return zones

  def _draw_car_body(self, center: rl.Vector2, width: float, height: float, fade) -> None:
    body = rl.Rectangle(center.x - width / 2, center.y - height / 2, width, height)
    rl.draw_rectangle_rounded(body, 0.34, 18, fade(_CAR_FILL))
    rl.draw_rectangle_rounded_lines_ex(body, 0.34, 18, 3, fade(_CAR_EDGE))

    roof = rl.Rectangle(body.x + width * 0.18, body.y + height * 0.15, width * 0.64, height * 0.48)
    rl.draw_rectangle_rounded(roof, 0.26, 16, fade(_GLASS))
    hood = rl.Rectangle(body.x + width * 0.17, body.y + height * 0.02, width * 0.66, height * 0.12)
    rear = rl.Rectangle(body.x + width * 0.17, body.y + height * 0.84, width * 0.66, height * 0.08)
    rl.draw_rectangle_rounded(hood, 0.4, 12, fade(_MID_GREY, 0.55))
    rl.draw_rectangle_rounded(rear, 0.4, 12, fade(_MID_GREY, 0.55))

    mirror_w = width * 0.12
    mirror_h = height * 0.08
    left_mirror = rl.Rectangle(body.x - mirror_w * 0.55, body.y + height * 0.36, mirror_w, mirror_h)
    right_mirror = rl.Rectangle(body.x + width - mirror_w * 0.45, body.y + height * 0.36, mirror_w, mirror_h)
    rl.draw_rectangle_rounded(left_mirror, 0.5, 10, fade(_CAR_EDGE, 0.85))
    rl.draw_rectangle_rounded(right_mirror, 0.5, 10, fade(_CAR_EDGE, 0.85))

    rl.draw_line_ex(rl.Vector2(body.x + width * 0.14, body.y + height * 0.74),
                    rl.Vector2(body.x + width * 0.86, body.y + height * 0.74), 2, fade(_DARK_GREY))
    rl.draw_line_ex(rl.Vector2(body.x + width * 0.32, body.y + height * 0.16),
                    rl.Vector2(body.x + width * 0.32, body.y + height * 0.62), 2, fade(_MID_GREY))
    rl.draw_line_ex(rl.Vector2(body.x + width * 0.68, body.y + height * 0.16),
                    rl.Vector2(body.x + width * 0.68, body.y + height * 0.62), 2, fade(_MID_GREY))

  def _draw_zone_labels(self, panel: rl.Rectangle, zones: dict[str, bool], fade) -> None:
    labels = [
      ("FL", zones["front_left"], panel.x + 34, panel.y + 86),
      ("FR", zones["front_right"], panel.x + panel.width - 68, panel.y + 86),
      ("RL", zones["rear_left"], panel.x + 34, panel.y + panel.height - 112),
      ("RR", zones["rear_right"], panel.x + panel.width - 68, panel.y + panel.height - 112),
    ]
    for text, active, x, y in labels:
      color = _RED if active else _DIM
      rl.draw_text_ex(self._cap_font, text, rl.Vector2(int(x), int(y)), 24, 0, fade(color))
