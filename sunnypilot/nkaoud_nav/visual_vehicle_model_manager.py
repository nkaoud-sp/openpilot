#!/usr/bin/env python3
"""
Offroad model manager for the standalone visual vehicle detector.

The Tweaks UI writes one-shot Params:
  - VisualVehicleDetectorDownloadTrigger
  - VisualVehicleDetectorCompileTrigger

This daemon runs offroad, performs the requested job, and writes a status JSON
to VisualVehicleDetectorManagerStatus. It does not interact with controls or
navigation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "selfdrive/modeld/models"
ONNX_PATH = MODEL_DIR / "visual_vehicle_detector.onnx"
PKL_PATH = MODEL_DIR / "visual_vehicle_detector_tinygrad.pkl"
META_PATH = MODEL_DIR / "visual_vehicle_detector_tinygrad.json"
COMPILE_SCRIPT = REPO_ROOT / "tools/nkaoud/compile_visual_vehicle_detector_tinygrad.py"

# Default to a tiny COCO detector. It can be replaced later by changing the
# VisualVehicleDetectorModelUrl param, or by editing this constant.
DEFAULT_MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"

STATUS_PARAM = "VisualVehicleDetectorManagerStatus"
DOWNLOAD_TRIGGER_PARAM = "VisualVehicleDetectorDownloadTrigger"
COMPILE_TRIGGER_PARAM = "VisualVehicleDetectorCompileTrigger"
MODEL_URL_PARAM = "VisualVehicleDetectorModelUrl"


class VisualVehicleModelManager:
  def __init__(self) -> None:
    self.params = Params()

  @staticmethod
  def _size_mb(path: Path) -> float:
    try:
      return round(path.stat().st_size / (1024 * 1024), 2)
    except Exception:
      return 0.0

  def _status_payload(self, state: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {
      "state": state,
      "message": message,
      "updated_at": time.time(),
      "onnx_exists": ONNX_PATH.exists(),
      "pkl_exists": PKL_PATH.exists(),
      "meta_exists": META_PATH.exists(),
      "onnx_size_mb": self._size_mb(ONNX_PATH),
      "pkl_size_mb": self._size_mb(PKL_PATH),
      "onnx_path": str(ONNX_PATH),
      "pkl_path": str(PKL_PATH),
    }
    payload.update(extra)
    return payload

  def _put_status(self, state: str, message: str, **extra: Any) -> None:
    payload = self._status_payload(state, message, **extra)
    try:
      self.params.put(STATUS_PARAM, json.dumps(payload, separators=(",", ":")))
    except Exception:
      cloudlog.exception("visual vehicle model manager failed to write status")

  def _model_url(self) -> str:
    url = (self.params.get(MODEL_URL_PARAM) or "").strip()
    return url or DEFAULT_MODEL_URL

  def _download(self) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = self._model_url()
    tmp_path = ONNX_PATH.with_suffix(".onnx.download")

    self._put_status("downloading", "Downloading visual detector ONNX...", url=url)
    cloudlog.warning("visual vehicle model download: %s", url)

    try:
      req = urllib.request.Request(url, headers={"User-Agent": "openpilot-visual-vehicle-detector"})
      with urllib.request.urlopen(req, timeout=45) as resp, open(tmp_path, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last_status = time.monotonic()
        while True:
          chunk = resp.read(1024 * 512)
          if not chunk:
            break
          f.write(chunk)
          done += len(chunk)
          now = time.monotonic()
          if now - last_status > 1.0:
            percent = (done / total * 100.0) if total else 0.0
            self._put_status("downloading", f"Downloading ONNX... {done / (1024 * 1024):.1f} MB", url=url,
                             download_percent=round(percent, 1))
            last_status = now

      if tmp_path.stat().st_size < 1024 * 1024:
        raise RuntimeError(f"Downloaded file is unexpectedly small: {tmp_path.stat().st_size} bytes")

      os.replace(tmp_path, ONNX_PATH)
      self._put_status("downloaded", "ONNX download complete.", url=url)
    except Exception as e:
      try:
        if tmp_path.exists():
          tmp_path.unlink()
      except Exception:
        pass
      self._put_status("error", f"Download failed: {e}", url=url)
      cloudlog.exception("visual vehicle model download failed")

  def _compile(self) -> None:
    if not ONNX_PATH.exists():
      self._put_status("error", "Cannot compile: ONNX file is missing. Tap Download ONNX first.")
      return

    if not COMPILE_SCRIPT.exists():
      self._put_status("error", f"Compile script missing: {COMPILE_SCRIPT}")
      return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_pkl = PKL_PATH.with_suffix(".pkl.tmp")
    tmp_meta = META_PATH.with_suffix(".json.tmp")

    # Clear stale temp files from previous failed runs.
    for p in (tmp_pkl, tmp_meta):
      try:
        if p.exists():
          p.unlink()
      except Exception:
        pass

    imgsz = "320"
    self._put_status("compiling", "Compiling ONNX to tinygrad PKL. Keep the device offroad.", imgsz=imgsz)

    cmd = [
      sys.executable,
      str(COMPILE_SCRIPT),
      "--onnx", str(ONNX_PATH),
      "--out", str(tmp_pkl),
      "--metadata", str(tmp_meta),
      "--imgsz", imgsz,
    ]

    try:
      env = os.environ.copy()
      # Keep aligned with modeld on comma3x when possible.
      env.setdefault("DEV", "QCOM")
      proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=45 * 60)
      output_tail = (proc.stdout or "")[-3000:]
      if proc.returncode != 0:
        raise RuntimeError(f"compile failed rc={proc.returncode}: {output_tail}")

      if not tmp_pkl.exists():
        raise RuntimeError("compile finished but did not create PKL")

      os.replace(tmp_pkl, PKL_PATH)
      if tmp_meta.exists():
        os.replace(tmp_meta, META_PATH)

      self._put_status("compiled", "Tinygrad PKL compile complete.", output=output_tail)
    except Exception as e:
      self._put_status("error", f"Compile failed: {e}")
      cloudlog.exception("visual vehicle model compile failed")

  def _idle_status(self) -> None:
    current = self.params.get(STATUS_PARAM)
    if current:
      return

    if PKL_PATH.exists():
      self._put_status("idle", "PKL ready.")
    elif ONNX_PATH.exists():
      self._put_status("idle", "ONNX ready. Tap Compile PKL.")
    else:
      self._put_status("idle", "No detector model yet. Tap Download ONNX.")

  def update(self) -> None:
    download_trigger = self.params.get(DOWNLOAD_TRIGGER_PARAM) or ""
    compile_trigger = self.params.get(COMPILE_TRIGGER_PARAM) or ""

    if download_trigger:
      self.params.remove(DOWNLOAD_TRIGGER_PARAM)
      self._download()
      return

    if compile_trigger:
      self.params.remove(COMPILE_TRIGGER_PARAM)
      self._compile()
      return

    self._idle_status()


def main() -> None:
  mgr = VisualVehicleModelManager()
  rk = Ratekeeper(1.0)
  while True:
    try:
      mgr.update()
    except Exception:
      cloudlog.exception("visual vehicle model manager loop failed")
    rk.keep_time()


if __name__ == "__main__":
  main()
