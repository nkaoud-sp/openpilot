"""
Settings submenu for the standalone visual vehicle detector test.
"""
from __future__ import annotations

from collections.abc import Callable
import os
import threading

import pyray as rl
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, ListItemSP, SimpleButtonActionSP
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller


class VisualVehicleSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()
    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)
    self._status = ""
    self._lock = threading.Lock()
    self._worker: threading.Thread | None = None
    self._scroller = Scroller(self._initialize_items(), line_separator=True, spacing=0)

  def _download_button_label(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import ONNX_PATH
    with self._lock:
      status = self._status
    if status.startswith("Downloading"):
      return tr("Downloading ONNX...")
    if status.startswith("error"):
      return tr("Retry Download")
    if os.path.exists(ONNX_PATH):
      return tr("Re-download ONNX")
    return tr("Download ONNX")

  def _compile_button_label(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import ONNX_PATH, PKL_PATH
    with self._lock:
      status = self._status
    if status.startswith("Compiling"):
      return tr("Compiling PKL...")
    if status.startswith("error"):
      return tr("Retry Compile")
    if os.path.exists(PKL_PATH):
      return tr("Recompile PKL")
    if os.path.exists(ONNX_PATH):
      return tr("Compile PKL")
    return tr("Compile PKL")

  def _download_description(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import ONNX_PATH, PKL_PATH
    with self._lock:
      status = self._status
    if status:
      return status
    onnx = "yes" if os.path.exists(ONNX_PATH) else "no"
    pkl = "yes" if os.path.exists(PKL_PATH) else "no"
    return tr("Downloads the default tiny COCO vehicle detector ONNX into selfdrive/modeld/models. ONNX: {}  PKL: {}").format(onnx, pkl)

  def _compile_description(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import ONNX_PATH, PKL_PATH
    with self._lock:
      status = self._status
    if status:
      return status
    if os.path.exists(PKL_PATH):
      return tr("Tinygrad PKL is present. Tap to rebuild it on this device.")
    if os.path.exists(ONNX_PATH):
      return tr("Compiles the downloaded ONNX to visual_vehicle_detector_tinygrad.pkl on this device. Keep the car offroad.")
    return tr("Download the ONNX first, then compile it to a tinygrad PKL.")

  def _set_status(self, status: str) -> None:
    with self._lock:
      self._status = status

  def _run_download(self) -> None:
    try:
      from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import ensure_onnx, ONNX_PATH
      self._set_status("Downloading ONNX...")
      if ONNX_PATH.exists():
        ONNX_PATH.unlink()
      ensure_onnx()
      self._set_status("ONNX download complete.")
    except Exception as e:
      self._set_status(f"error: download failed: {e}")
    finally:
      with self._lock:
        self._worker = None

  def _run_compile(self) -> None:
    try:
      from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import ONNX_PATH, compile_pkl
      if not os.path.exists(ONNX_PATH):
        self._set_status("error: ONNX file is missing")
        return
      self._set_status("Compiling PKL...")
      compile_pkl()
      self._set_status("PKL compile complete.")
    except Exception as e:
      self._set_status(f"error: compile failed: {e}")
    finally:
      with self._lock:
        self._worker = None

  def _start_worker(self, target: Callable[[], None]) -> None:
    with self._lock:
      if self._worker is not None and self._worker.is_alive():
        self._status = "Another setup task is already running."
        return
      self._worker = threading.Thread(target=target, daemon=True)
      self._worker.start()

  def _trigger_download(self) -> None:
    self._start_worker(self._run_download)

  def _trigger_compile(self) -> None:
    self._start_worker(self._run_compile)

  def _initialize_items(self):
    self._download_model = ListItemSP(
      title=lambda: tr("Detector ONNX Model"),
      description=lambda: self._download_description(),
      description_visible=True,
      inline=False,
      action_item=SimpleButtonActionSP(
        button_text=lambda: self._download_button_label(),
        button_width=800,
        callback=lambda: self._trigger_download(),
      ),
    )

    self._compile_model = ListItemSP(
      title=lambda: tr("Tinygrad PKL"),
      description=lambda: self._compile_description(),
      description_visible=True,
      inline=False,
      action_item=SimpleButtonActionSP(
        button_text=lambda: self._compile_button_label(),
        button_width=800,
        callback=lambda: self._trigger_compile(),
      ),
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
      self._download_model,
      self._compile_model,
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
