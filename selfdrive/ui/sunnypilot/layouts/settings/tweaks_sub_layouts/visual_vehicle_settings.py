"""
Settings submenu for the standalone visual vehicle detector test.
"""
from __future__ import annotations

from collections.abc import Callable
import json
import time

import pyray as rl
from openpilot.common.params import Params
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, button_item_sp, ListItemSP
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller


class VisualVehicleSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()
    self._params = Params()
    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)
    self._scroller = Scroller(self._initialize_items(), line_separator=True, spacing=0)

  def _status(self) -> dict:
    try:
      raw = self._params.get("VisualVehicleDetectorManagerStatus") or "{}"
      return json.loads(raw)
    except Exception:
      return {}

  def _status_title(self) -> str:
    st = self._status()
    state = str(st.get("state", "idle")).upper()
    return tr("Model Manager Status: {}").format(state)

  def _status_description(self) -> str:
    st = self._status()
    msg = st.get("message", "No status yet.")
    onnx = "yes" if st.get("onnx_exists") else "no"
    pkl = "yes" if st.get("pkl_exists") else "no"
    meta = "yes" if st.get("meta_exists") else "no"
    onnx_mb = st.get("onnx_size_mb", 0)
    pkl_mb = st.get("pkl_size_mb", 0)
    updated_at = float(st.get("updated_at", 0) or 0)
    age = max(0.0, time.time() - updated_at) if updated_at else 0.0
    return tr("{}\nONNX: {} ({} MB)  |  PKL: {} ({} MB)  |  META: {}\nUpdated {:.0f}s ago").format(
      msg, onnx, onnx_mb, pkl, pkl_mb, meta, age
    )

  def _queue_action(self, trigger_param: str, message: str) -> None:
    status = self._status()
    status.update({
      "state": "queued",
      "message": message,
      "updated_at": time.time(),
    })
    self._params.put("VisualVehicleDetectorManagerStatus", json.dumps(status, separators=(",", ":")))
    self._params.put(trigger_param, str(time.time_ns()))

  def _trigger_download(self) -> None:
    self._queue_action("VisualVehicleDetectorDownloadTrigger", "Download queued. Waiting for model manager...")

  def _trigger_compile(self) -> None:
    self._queue_action("VisualVehicleDetectorCompileTrigger", "Compile queued. Waiting for model manager...")

  def _clear_status(self) -> None:
    self._params.remove("VisualVehicleDetectorManagerStatus")

  def _initialize_items(self):
    self._status_item = ListItemSP(
      title=lambda: self._status_title(),
      description=lambda: self._status_description(),
      description_visible=True,
      inline=False,
    )

    self._download_model = button_item_sp(
      title=lambda: tr("Detector ONNX Model"),
      button_text=lambda: tr("Download ONNX"),
      description=lambda: tr("Downloads the default tiny COCO vehicle detector ONNX into selfdrive/modeld/models. "
                             "Use this before compiling if the ONNX file is missing."),
      callback=lambda: self._trigger_download(),
    )

    self._compile_model = button_item_sp(
      title=lambda: tr("Tinygrad PKL"),
      button_text=lambda: tr("Compile PKL"),
      description=lambda: tr("Compiles the downloaded ONNX to visual_vehicle_detector_tinygrad.pkl on this device. "
                             "Keep the car offroad; this may take several minutes and the UI status will update."),
      callback=lambda: self._trigger_compile(),
    )

    self._clear_status_button = button_item_sp(
      title=lambda: tr("Status"),
      button_text=lambda: tr("Clear Status"),
      description=lambda: tr("Clears the model manager status text. It will repopulate on the next manager tick."),
      callback=lambda: self._clear_status(),
    )

    self._readout = toggle_item_sp(
      title=lambda: tr("Show Detector Readout"),
      description=lambda: tr("Draw a large on-road debug panel with left/right vehicle status, stale state, "
                            "detector reason, detection count, confidence and frame id."),
      param="VisualVehicleDetectorReadout",
    )
    self._allow_onnx = toggle_item_sp(
      title=lambda: tr("Allow ONNX Fallback (debug only)"),
      description=lambda: tr("If the PKL is missing, allow the detector daemon to try ONNX Runtime. Leave OFF on "
                            "comma3x unless you are only debugging process/UI behavior."),
      param="VisualVehicleDetectorAllowOnnx",
    )
    self._debug_log = toggle_item_sp(
      title=lambda: tr("Log Detector Debug"),
      description=lambda: tr("Write detector debug lines to cloudlog. Useful while tuning the model, but leave "
                            "off for normal driving tests."),
      param="VisualVehicleDetectorLogDebug",
    )
    return [
      self._status_item,
      self._download_model,
      self._compile_model,
      self._clear_status_button,
      self._readout,
      self._allow_onnx,
      self._debug_log,
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
