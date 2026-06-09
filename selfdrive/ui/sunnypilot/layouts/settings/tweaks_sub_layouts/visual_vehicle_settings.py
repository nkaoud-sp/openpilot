"""
Settings submenu for the standalone visual vehicle detector test.
"""
from __future__ import annotations

from collections.abc import Callable
import os
import threading
import time

import pyray as rl
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, ListItemSP, SimpleButtonActionSP
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller

from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.nav_token_qr_dialog import (
  VisualVehiclePreviewQrDialog,
)


class VisualVehicleSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()
    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)
    self._status = self._load_status_message()
    self._lock = threading.Lock()
    self._worker: threading.Thread | None = None
    self._scroller = Scroller(self._initialize_items(), line_separator=True, spacing=0)

  def _status_data(self) -> dict:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import read_status
    return read_status()

  def _load_status_message(self) -> str:
    status = self._status_data()
    return str(status.get("message", "")) if status else ""

  def _idle_status_line(self) -> str:
    status = self._status_data()
    if status:
      onnx = "yes" if status.get("onnx_exists") else "no"
      pkl = "yes" if status.get("pkl_exists") else "no"
      onnx_mb = status.get("onnx_size_mb", 0)
      pkl_mb = status.get("pkl_size_mb", 0)
      updated_at = float(status.get("updated_at", 0) or 0)
      age_s = max(0, int(time.time() - updated_at)) if updated_at else 0
      message = str(status.get("message", ""))
      return tr("{} ONNX: {} ({} MB)  PKL: {} ({} MB)  Updated {}s ago").format(
        message, onnx, onnx_mb, pkl, pkl_mb, age_s
      )
    return ""

  def _download_button_label(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import ONNX_PATH
    with self._lock:
      status = self._status
    if status.startswith("Downloading"):
      return tr("Downloading ONNX...")
    if status.startswith("error"):
      return tr("Retry Download")
    if os.path.exists(ONNX_PATH):
      return tr("Re-download 640 ONNX")
    return tr("Download 640 ONNX")

  def _download_320_button_label(self) -> str:
    with self._lock:
      status = self._status
    if status.startswith("Downloading"):
      return tr("Downloading 320 ONNX...")
    if status.startswith("error"):
      return tr("Retry 320 Download")
    return tr("Download 320 ONNX")

  def _download_256_button_label(self) -> str:
    with self._lock:
      status = self._status
    if status.startswith("Downloading"):
      return tr("Downloading 256 ONNX...")
    if status.startswith("error"):
      return tr("Retry 256 Download")
    return tr("Download 256 ONNX")

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
    idle = self._idle_status_line()
    if idle:
      return idle
    onnx = "yes" if os.path.exists(ONNX_PATH) else "no"
    pkl = "yes" if os.path.exists(PKL_PATH) else "no"
    return tr("Downloads the default YOLOv5n ONNX (640x640). ONNX: {}  PKL: {}").format(onnx, pkl)

  def _download_320_description(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import DEFAULT_MODEL_320_URL
    if DEFAULT_MODEL_320_URL:
      return tr("Downloads a hosted 320x320 ONNX export and replaces the current ONNX before compile.")
    return tr("No 320x320 download URL is configured yet. Export yolov5n at 320x320, host it, or manually place it at /data/visual_vehicle_detector/visual_vehicle_detector.onnx, then tap Compile PKL.")

  def _download_256_description(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import DEFAULT_MODEL_256_URL
    if DEFAULT_MODEL_256_URL:
      return tr("Downloads a hosted 256x256 ONNX export and replaces the current ONNX before compile.")
    return tr("No 256x256 download URL is configured yet.")

  def _compile_description(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import ONNX_PATH, PKL_PATH
    with self._lock:
      status = self._status
    if status:
      return status
    idle = self._idle_status_line()
    if idle:
      return idle
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
      from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import ensure_onnx_640
      self._set_status("Downloading 640 ONNX...")
      ensure_onnx_640()
      self._set_status(self._load_status_message())
    except Exception as e:
      self._set_status(f"error: download failed: {e}")
    finally:
      with self._lock:
        self._worker = None

  def _run_download_320(self) -> None:
    try:
      from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import ensure_onnx_320
      self._set_status("Downloading 320 ONNX...")
      ensure_onnx_320()
      self._set_status(self._load_status_message())
    except Exception as e:
      self._set_status(f"error: download failed: {e}")
    finally:
      with self._lock:
        self._worker = None

  def _run_download_256(self) -> None:
    try:
      from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import ensure_onnx_256
      self._set_status("Downloading 256 ONNX...")
      ensure_onnx_256()
      self._set_status(self._load_status_message())
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
      self._set_status(self._load_status_message())
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

  def _trigger_download_320(self) -> None:
    self._start_worker(self._run_download_320)

  def _trigger_download_256(self) -> None:
    self._start_worker(self._run_download_256)

  def _trigger_compile(self) -> None:
    self._start_worker(self._run_compile)

  def _initialize_items(self):
    self._download_model = ListItemSP(
      title=lambda: tr("Detector ONNX (640)"),
      description=lambda: self._download_description(),
      description_visible=True,
      inline=False,
      action_item=SimpleButtonActionSP(
        button_text=lambda: self._download_button_label(),
        button_width=800,
        callback=lambda: self._trigger_download(),
      ),
    )

    self._download_model_320 = ListItemSP(
      title=lambda: tr("Detector ONNX (320)"),
      description=lambda: self._download_320_description(),
      description_visible=True,
      inline=False,
      action_item=SimpleButtonActionSP(
        button_text=lambda: self._download_320_button_label(),
        button_width=800,
        callback=lambda: self._trigger_download_320(),
      ),
    )

    self._download_model_256 = ListItemSP(
      title=lambda: tr("Detector ONNX (256)"),
      description=lambda: self._download_256_description(),
      description_visible=True,
      inline=False,
      action_item=SimpleButtonActionSP(
        button_text=lambda: self._download_256_button_label(),
        button_width=800,
        callback=lambda: self._trigger_download_256(),
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
    self._live_preview = ListItemSP(
      title=lambda: tr("Live Preview"),
      description=lambda: tr("Shows a QR + URL for a phone browser view of the exact 320x320 RGB tensor the "
                             "detector sees, so you can sanity-check the NV12->RGB conversion and letterbox. "
                             "The preview is written only while this dialog is open."),
      description_visible=True,
      inline=False,
      action_item=SimpleButtonActionSP(
        button_text=lambda: tr("Open Detector Preview"),
        button_width=800,
        callback=lambda: gui_app.push_widget(VisualVehiclePreviewQrDialog()),
      ),
    )
    return [
      self._download_model,
      self._download_model_320,
      self._download_model_256,
      self._compile_model,
      self._readout,
      self._allow_onnx,
      self._debug_log,
      self._live_preview,
    ]

  def _render(self, rect):
    self._back_button.set_position(self._rect.x, self._rect.y + 20)
    self._back_button.render()
    content_rect = rl.Rectangle(rect.x, rect.y + self._back_button.rect.height + 40,
                                rect.width, rect.height - self._back_button.rect.height - 40)
    self._scroller.render(content_rect)

  def show_event(self):
    with self._lock:
      if self._worker is None or not self._worker.is_alive():
        self._status = self._load_status_message()
    self._scroller.show_event()

  def hide_event(self):
    self._scroller.hide_event()
