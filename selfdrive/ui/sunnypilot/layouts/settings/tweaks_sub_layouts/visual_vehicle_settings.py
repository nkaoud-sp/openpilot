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
from openpilot.system.ui.sunnypilot.widgets.list_view import (
  multiple_button_item_sp, toggle_item_sp, ListItemSP, SimpleButtonActionSP,
)
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller

from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.nav_token_qr_dialog import (
  VisualVehicleCaptureQrDialog,
  VisualVehicleCropQrDialog,
  VisualVehiclePreviewQrDialog,
  VisualVehicleStagesQrDialog,
  VisualVehicleTuningQrDialog,
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

  def _download_480_button_label(self) -> str:
    with self._lock:
      status = self._status
    if status.startswith("Downloading"):
      return tr("Downloading 480 ONNX...")
    if status.startswith("error"):
      return tr("Retry 480 Download")
    return tr("Download 480 ONNX")

  def _download_480x224_button_label(self) -> str:
    with self._lock:
      status = self._status
    if status.startswith("Downloading"):
      return tr("Downloading 480x224 ONNX...")
    if status.startswith("error"):
      return tr("Retry 480x224 Download")
    return tr("Download 480x224 ONNX")

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

  def _compile_driver_button_label(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import DRIVER_PKL_PATH
    with self._lock:
      status = self._status
    if status.startswith("Compiling"):
      return tr("Compiling DM PKL...")
    if status.startswith("error"):
      return tr("Retry DM Compile")
    if os.path.exists(DRIVER_PKL_PATH):
      return tr("Recompile DM PKL")
    return tr("Compile DM PKL")

  def _compile_driver_description(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import DRIVER_PKL_PATH
    with self._lock:
      status = self._status
    if status:
      return status
    present = "yes" if os.path.exists(DRIVER_PKL_PATH) else "no"
    return tr("Compiles the current ONNX into a separate driver-camera model "
              "(visual_vehicle_detector_driver_tinygrad.pkl), used only when the Driver camera is selected. "
              "Download a square model (320 or 256) first to avoid letterbox waste. DM PKL present: {}").format(present)

  def _run_compile_driver(self) -> None:
    try:
      from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import ONNX_PATH, compile_pkl_driver
      if not os.path.exists(ONNX_PATH):
        self._set_status("error: ONNX file is missing")
        return
      self._set_status("Compiling DM PKL...")
      compile_pkl_driver()
      self._set_status(self._load_status_message())
    except Exception as e:
      self._set_status(f"error: compile failed: {e}")
    finally:
      with self._lock:
        self._worker = None

  def _trigger_compile_driver(self) -> None:
    self._start_worker(self._run_compile_driver)

  def _download_classifier_button_label(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import CLASSIFIER_ONNX_PATH
    with self._lock:
      status = self._status
    if status.startswith("Downloading"):
      return tr("Downloading DM Classifier...")
    if status.startswith("error"):
      return tr("Retry DM Classifier Download")
    if os.path.exists(CLASSIFIER_ONNX_PATH):
      return tr("Re-download DM Classifier")
    return tr("Download DM Classifier")

  def _download_classifier_description(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import CLASSIFIER_ONNX_PATH, CLASSIFIER_PKL_PATH
    with self._lock:
      status = self._status
    if status:
      return status
    onnx = "yes" if os.path.exists(CLASSIFIER_ONNX_PATH) else "no"
    pkl = "yes" if os.path.exists(CLASSIFIER_PKL_PATH) else "no"
    return tr("Downloads the hosted driver-cam car classifier (320x320 MobileNetV3-Small, 2-class). Used only "
              "when the Driver camera is selected; replaces YOLO there. Enable 'Allow ONNX Fallback' to run it "
              "without compiling, or compile it below. ONNX: {}  PKL: {}").format(onnx, pkl)

  def _run_download_classifier(self) -> None:
    try:
      from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import ensure_classifier_onnx
      self._set_status("Downloading DM classifier ONNX...")
      ensure_classifier_onnx()
      self._set_status(self._load_status_message())
    except Exception as e:
      self._set_status(f"error: download failed: {e}")
    finally:
      with self._lock:
        self._worker = None

  def _trigger_download_classifier(self) -> None:
    self._start_worker(self._run_download_classifier)

  def _compile_classifier_button_label(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import CLASSIFIER_PKL_PATH
    with self._lock:
      status = self._status
    if status.startswith("Compiling"):
      return tr("Compiling DM Classifier PKL...")
    if status.startswith("error"):
      return tr("Retry DM Classifier Compile")
    if os.path.exists(CLASSIFIER_PKL_PATH):
      return tr("Recompile DM Classifier PKL")
    return tr("Compile DM Classifier PKL")

  def _compile_classifier_description(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import CLASSIFIER_ONNX_PATH, CLASSIFIER_PKL_PATH
    with self._lock:
      status = self._status
    if status:
      return status
    if os.path.exists(CLASSIFIER_PKL_PATH):
      return tr("DM classifier PKL present. Tap to rebuild it on this device.")
    if os.path.exists(CLASSIFIER_ONNX_PATH):
      return tr("Compiles the DM classifier ONNX to its tinygrad PKL on this device. Keep the car offroad.")
    return tr("Download the DM classifier ONNX first, then compile it.")

  def _run_compile_classifier(self) -> None:
    try:
      from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import (
        CLASSIFIER_ONNX_PATH, compile_classifier_pkl, ensure_classifier_onnx,
      )
      if not os.path.exists(CLASSIFIER_ONNX_PATH):
        self._set_status("Installing DM classifier ONNX...")
        ensure_classifier_onnx()
      self._set_status("Compiling DM Classifier PKL...")
      compile_classifier_pkl()
      self._set_status(self._load_status_message())
    except Exception as e:
      self._set_status(f"error: compile failed: {e}")
    finally:
      with self._lock:
        self._worker = None

  def _trigger_compile_classifier(self) -> None:
    self._start_worker(self._run_compile_classifier)

  # ---- Wide-camera car classifier (replaces YOLO on the wide cam) ----

  def _download_wide_classifier_button_label(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import WIDE_CLASSIFIER_ONNX_PATH
    with self._lock:
      status = self._status
    if status.startswith("Downloading"):
      return tr("Downloading Wide Classifier...")
    if status.startswith("error"):
      return tr("Retry Wide Classifier Download")
    if os.path.exists(WIDE_CLASSIFIER_ONNX_PATH):
      return tr("Re-download Wide Classifier")
    return tr("Download Wide Classifier")

  def _download_wide_classifier_description(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import (
      WIDE_CLASSIFIER_ONNX_PATH, WIDE_CLASSIFIER_PKL_PATH, MODEL_CONFIG_PATH,
    )
    with self._lock:
      status = self._status
    if status:
      return status
    onnx = "yes" if os.path.exists(WIDE_CLASSIFIER_ONNX_PATH) else "no"
    pkl = "yes" if os.path.exists(WIDE_CLASSIFIER_PKL_PATH) else "no"
    cfg = "yes" if os.path.exists(MODEL_CONFIG_PATH) else "no"
    return tr("Downloads the hosted wide-cam car classifier (320x128 MobileNetV3-Small, single-zone) and the "
              "model_config.json. Used only when the Wide camera is selected; replaces YOLO there. Tune the wide "
              "crop to 854x280. ONNX: {}  PKL: {}  Config: {}").format(onnx, pkl, cfg)

  def _run_download_wide_classifier(self) -> None:
    try:
      from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import ensure_wide_classifier_onnx
      self._set_status("Downloading wide classifier ONNX...")
      ensure_wide_classifier_onnx()
      self._set_status(self._load_status_message())
    except Exception as e:
      self._set_status(f"error: download failed: {e}")
    finally:
      with self._lock:
        self._worker = None

  def _trigger_download_wide_classifier(self) -> None:
    self._start_worker(self._run_download_wide_classifier)

  def _compile_wide_classifier_button_label(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import WIDE_CLASSIFIER_PKL_PATH
    with self._lock:
      status = self._status
    if status.startswith("Compiling"):
      return tr("Compiling Wide Classifier PKL...")
    if status.startswith("error"):
      return tr("Retry Wide Classifier Compile")
    if os.path.exists(WIDE_CLASSIFIER_PKL_PATH):
      return tr("Recompile Wide Classifier PKL")
    return tr("Compile Wide Classifier PKL")

  def _compile_wide_classifier_description(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import WIDE_CLASSIFIER_ONNX_PATH, WIDE_CLASSIFIER_PKL_PATH
    with self._lock:
      status = self._status
    if status:
      return status
    if os.path.exists(WIDE_CLASSIFIER_PKL_PATH):
      return tr("Wide classifier PKL present. Tap to rebuild it on this device.")
    if os.path.exists(WIDE_CLASSIFIER_ONNX_PATH):
      return tr("Compiles the wide classifier ONNX to its tinygrad PKL on this device. Keep the car offroad.")
    return tr("Download the wide classifier ONNX first, then compile it.")

  def _run_compile_wide_classifier(self) -> None:
    try:
      from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import (
        WIDE_CLASSIFIER_ONNX_PATH, compile_wide_classifier_pkl, ensure_wide_classifier_onnx,
      )
      if not os.path.exists(WIDE_CLASSIFIER_ONNX_PATH):
        self._set_status("Installing wide classifier ONNX...")
        ensure_wide_classifier_onnx()
      self._set_status("Compiling Wide Classifier PKL...")
      compile_wide_classifier_pkl()
      self._set_status(self._load_status_message())
    except Exception as e:
      self._set_status(f"error: compile failed: {e}")
    finally:
      with self._lock:
        self._worker = None

  def _trigger_compile_wide_classifier(self) -> None:
    self._start_worker(self._run_compile_wide_classifier)

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

  def _download_480_description(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import DEFAULT_MODEL_480_URL
    if DEFAULT_MODEL_480_URL:
      return tr("Downloads a hosted 480x480 ONNX export and replaces the current ONNX before compile.")
    return tr("No 480x480 download URL is configured yet.")

  def _download_480x224_description(self) -> str:
    from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import DEFAULT_MODEL_480X224_URL
    if DEFAULT_MODEL_480X224_URL:
      return tr("Downloads a hosted 480x224 ONNX export and replaces the current ONNX before compile.")
    return tr("No 480x224 download URL is configured yet.")

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

  def _run_download_480(self) -> None:
    try:
      from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import ensure_onnx_480
      self._set_status("Downloading 480 ONNX...")
      ensure_onnx_480()
      self._set_status(self._load_status_message())
    except Exception as e:
      self._set_status(f"error: download failed: {e}")
    finally:
      with self._lock:
        self._worker = None

  def _run_download_480x224(self) -> None:
    try:
      from openpilot.sunnypilot.nkaoud_nav.visual_vehicle_setup import ensure_onnx_480x224
      self._set_status("Downloading 480x224 ONNX...")
      ensure_onnx_480x224()
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

  def _trigger_download_480(self) -> None:
    self._start_worker(self._run_download_480)

  def _trigger_download_480x224(self) -> None:
    self._start_worker(self._run_download_480x224)

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

    self._download_model_480 = ListItemSP(
      title=lambda: tr("Detector ONNX (480)"),
      description=lambda: self._download_480_description(),
      description_visible=True,
      inline=False,
      action_item=SimpleButtonActionSP(
        button_text=lambda: self._download_480_button_label(),
        button_width=800,
        callback=lambda: self._trigger_download_480(),
      ),
    )

    self._download_model_480x224 = ListItemSP(
      title=lambda: tr("Detector ONNX (480x224)"),
      description=lambda: self._download_480x224_description(),
      description_visible=True,
      inline=False,
      action_item=SimpleButtonActionSP(
        button_text=lambda: self._download_480x224_button_label(),
        button_width=800,
        callback=lambda: self._trigger_download_480x224(),
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

    self._compile_driver_model = ListItemSP(
      title=lambda: tr("Driver-cam PKL"),
      description=lambda: self._compile_driver_description(),
      description_visible=True,
      inline=False,
      action_item=SimpleButtonActionSP(
        button_text=lambda: self._compile_driver_button_label(),
        button_width=800,
        callback=lambda: self._trigger_compile_driver(),
      ),
    )

    self._download_classifier = ListItemSP(
      title=lambda: tr("DM Classifier ONNX (320)"),
      description=lambda: self._download_classifier_description(),
      description_visible=True,
      inline=False,
      action_item=SimpleButtonActionSP(
        button_text=lambda: self._download_classifier_button_label(),
        button_width=800,
        callback=lambda: self._trigger_download_classifier(),
      ),
    )
    self._compile_classifier = ListItemSP(
      title=lambda: tr("DM Classifier PKL"),
      description=lambda: self._compile_classifier_description(),
      description_visible=True,
      inline=False,
      action_item=SimpleButtonActionSP(
        button_text=lambda: self._compile_classifier_button_label(),
        button_width=800,
        callback=lambda: self._trigger_compile_classifier(),
      ),
    )

    self._download_wide_classifier = ListItemSP(
      title=lambda: tr("Wide Classifier ONNX (320x128)"),
      description=lambda: self._download_wide_classifier_description(),
      description_visible=True,
      inline=False,
      action_item=SimpleButtonActionSP(
        button_text=lambda: self._download_wide_classifier_button_label(),
        button_width=800,
        callback=lambda: self._trigger_download_wide_classifier(),
      ),
    )
    self._compile_wide_classifier = ListItemSP(
      title=lambda: tr("Wide Classifier PKL"),
      description=lambda: self._compile_wide_classifier_description(),
      description_visible=True,
      inline=False,
      action_item=SimpleButtonActionSP(
        button_text=lambda: self._compile_wide_classifier_button_label(),
        button_width=800,
        callback=lambda: self._trigger_compile_wide_classifier(),
      ),
    )

    self._camera_source = multiple_button_item_sp(
      title=lambda: tr("Camera Source"),
      description=lambda: tr("Which camera the detector runs on. Each camera keeps its own crop / ROI / gate "
                             "profile, and the tuning portals edit whichever camera is selected here. "
                             "Road = normal forward, Wide = wide angle, Driver = cabin. Wide+Driver runs both "
                             "classifier cams at once (wide-L/R + driver-L/R), one inference per frame."),
      buttons=[lambda: tr("Road"), lambda: tr("Wide"), lambda: tr("Driver"), lambda: tr("Wide+Driver")],
      param="VisualVehicleDetectorCamera",
    )
    self._readout = toggle_item_sp(
      title=lambda: tr("Show Detector Readout"),
      description=lambda: tr("Draw a large on-road debug panel with left/right vehicle status, stale state, "
                            "detector reason, detection count, confidence and frame id."),
      param="VisualVehicleDetectorReadout",
    )
    self._car_widget = toggle_item_sp(
      title=lambda: tr("Use Car Widget"),
      description=lambda: tr("Replace the detector readout panels with a top-down car widget. In Wide+Driver mode, "
                            "each blocked zone lights the matching corner red."),
      param="VisualVehicleDetectorCarWidget",
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
    self._stages_preview = ListItemSP(
      title=lambda: tr("Pipeline Stages"),
      description=lambda: tr("Shows a QR + URL for a phone browser view of every stage: the full camera frame "
                             "with the crop box, the crop before YOLO, and the letterboxed model input with ROI "
                             "and detection boxes. The images are written only while this dialog is open."),
      description_visible=True,
      inline=False,
      action_item=SimpleButtonActionSP(
        button_text=lambda: tr("Open Detector Stages"),
        button_width=800,
        callback=lambda: gui_app.push_widget(VisualVehicleStagesQrDialog()),
      ),
    )
    self._live_tuning = ListItemSP(
      title=lambda: tr("Live Tuning"),
      description=lambda: tr("Shows a QR + URL for a phone browser view, over the model input image, with sliders "
                             "for the right ROI band, the size/position gate and detection confidence. Green boxes "
                             "trip the right-lane flag, gray are ignored; changes apply live and persist."),
      description_visible=True,
      inline=False,
      action_item=SimpleButtonActionSP(
        button_text=lambda: tr("Open Live Tuning"),
        button_width=800,
        callback=lambda: gui_app.push_widget(VisualVehicleTuningQrDialog()),
      ),
    )
    self._crop_tuning = ListItemSP(
      title=lambda: tr("Crop & Rate"),
      description=lambda: tr("Shows a QR + URL for a phone browser view, over the full camera frame, with sliders "
                             "for the crop box (x/y/width/height) and the detector rate. Drag the yellow crop box "
                             "over the area YOLO should see; changes apply live and persist."),
      description_visible=True,
      inline=False,
      action_item=SimpleButtonActionSP(
        button_text=lambda: tr("Open Crop & Rate"),
        button_width=800,
        callback=lambda: gui_app.push_widget(VisualVehicleCropQrDialog()),
      ),
    )
    self._image_capture = ListItemSP(
      title=lambda: tr("Image Capture (training)"),
      description=lambda: tr("Records the selected camera's crop to the device while this portal's QR dialog is "
                             "open and the car is onroad (offroad is paused). No live preview. Scan the QR to "
                             "download all images as a ZIP or delete them. Capped to protect storage."),
      description_visible=True,
      inline=False,
      action_item=SimpleButtonActionSP(
        button_text=lambda: tr("Open Image Capture"),
        button_width=800,
        callback=lambda: gui_app.push_widget(VisualVehicleCaptureQrDialog()),
      ),
    )
    return [
      self._download_model,
      self._download_model_480,
      self._download_model_480x224,
      self._download_model_320,
      self._download_model_256,
      self._compile_model,
      self._compile_driver_model,
      self._download_classifier,
      self._compile_classifier,
      self._download_wide_classifier,
      self._compile_wide_classifier,
      self._camera_source,
      self._readout,
      self._car_widget,
      self._allow_onnx,
      self._debug_log,
      self._live_preview,
      self._stages_preview,
      self._live_tuning,
      self._crop_tuning,
      self._image_capture,
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
