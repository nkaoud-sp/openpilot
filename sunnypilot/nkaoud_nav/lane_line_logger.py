"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Per-engagement troubleshooting logger for the lane-line classifier.

While openpilot is actively engaged (and LaneLineVisualizerLogging is on),
lane_line_classifierd feeds every published assessment into a session:

  /data/media/0/lane_line_logs/session_<stamp>/
      config.json       classifier config + camera offset at session start
      assessments.csv   one row per publish: both sides' label/duty/period/...
      summary.json      per-side label histograms (written at session end)
      <t>_L_unknown_d032.raw.png  lossless, unannotated classifier input
      <t>_L_unknown_d032.jpg   rate-limited annotated snapshots per label
      <t>_L_unknown_d032.json  raw inputs to replay that frame offline
      ...

Snapshots are the road-camera Y plane with the scan corridor drawn on: the two
rails as white lines and one square per along-line sample - white where the
classifier saw marking, black where it saw none - so a single image shows both
where it looked and what it detected. Each annotated JPEG has a paired,
lossless unannotated Y-plane PNG and a matching .json sidecar. The PNG is the
exact image fed to the classifier; replay must never sample the overlay because
its rails and black/white detection squares would contaminate the result. The
sidecar contains both ego lane-line polylines (x/y/z), the calibration rpy,
the camera intrinsics and offset, so ``classify_line`` can be re-run offline
to tell an algorithm miss from genuinely worn/absent paint.

On disengage the session directory is zipped, queued on the nkaoud_nav
mailer's pending queue, and the directory removed; the mailer emails the zip
and deletes it only after the send succeeds (sunnypilot/nkaoud_nav/mailer.py).
Sessions orphaned by a mid-drive shutdown are zipped and queued by
``finalize_orphan_sessions`` (called at classifier start and by the mailer at
drive end).
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import time

import numpy as np

from openpilot.common.swaglog import cloudlog

LOG_DIR = os.environ.get("LANE_LINE_LOG_DIR", "/data/media/0/lane_line_logs")
SESSION_PREFIX = "session_"
ZIP_PREFIX = "lane_line_log_"
MAX_STORED_ZIPS = 10          # keep disk bounded if email is failing/off
SNAPSHOT_MIN_INTERVAL_S = 25.0  # per (side, label)
SNAPSHOT_SESSION_CAP = 24       # bounds the emailed zip to a few MB

LABELS = {0: "unknown", 1: "broken", 2: "solid", 3: "double"}

CSV_FIELDS = [
  "wall_time", "mono_time", "frame_id", "v_ego",
  "left_type", "left_reason", "left_conf", "left_duty", "left_period_m", "left_periodicity", "left_valid_frac", "left_offset_m", "left_n", "left_valid_n", "left_present_n", "left_low_contrast_n", "left_low_snr_n",
  "right_type", "right_reason", "right_conf", "right_duty", "right_period_m", "right_periodicity", "right_valid_frac", "right_offset_m", "right_n", "right_valid_n", "right_present_n", "right_low_contrast_n", "right_low_snr_n",
  "left_crossable", "right_crossable", "snapshots",
]


def _stamp() -> str:
  return time.strftime("%Y%m%d_%H%M%S", time.localtime())


class LaneLineSessionLogger:
  """One instance per daemon; start()/end() bracket each engagement."""

  def __init__(self):
    self._dir: str | None = None
    self._csv_file = None
    self._csv = None
    self._snap_last: dict[tuple[str, str], float] = {}
    self._snap_count = 0
    self._label_counts: dict[str, dict[str, int]] = {}

  @property
  def is_active(self) -> bool:
    return self._dir is not None

  def start(self, config: dict) -> None:
    if self.is_active:
      return
    try:
      session_dir = os.path.join(LOG_DIR, f"{SESSION_PREFIX}{_stamp()}")
      os.makedirs(session_dir, exist_ok=True)
      with open(os.path.join(session_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)
      self._csv_file = open(os.path.join(session_dir, "assessments.csv"), "w", newline="")
      self._csv = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS)
      self._csv.writeheader()
      self._dir = session_dir
      self._snap_last.clear()
      self._snap_count = 0
      self._label_counts = {"L": dict.fromkeys(LABELS.values(), 0),
                            "R": dict.fromkeys(LABELS.values(), 0)}
    except OSError:
      cloudlog.exception("lane_line_logger: could not start session")
      self._close_files()
      self._dir = None

  def log_row(self, mono_time: float, frame_id: int, v_ego: float | None, gate, snapshots: list[str]) -> None:
    if not self.is_active or self._csv is None:
      return
    try:
      row = {
        "wall_time": time.strftime("%H:%M:%S", time.localtime()),
        "mono_time": f"{mono_time:.2f}",
        "frame_id": frame_id,
        "v_ego": "" if v_ego is None else f"{v_ego:.1f}",
        "left_crossable": int(gate.left_crossable),
        "right_crossable": int(gate.right_crossable),
        "snapshots": ";".join(snapshots),
      }
      for side, res in (("left", gate.left), ("right", gate.right)):
        label = LABELS.get(int(res.line_type), "unknown")
        row[f"{side}_type"] = label
        row[f"{side}_reason"] = res.reason
        row[f"{side}_conf"] = f"{res.confidence:.2f}"
        row[f"{side}_duty"] = f"{res.duty:.2f}"
        row[f"{side}_period_m"] = f"{res.period_m:.1f}"
        row[f"{side}_periodicity"] = f"{res.periodicity:.2f}"
        row[f"{side}_valid_frac"] = f"{res.valid_frac:.2f}"
        row[f"{side}_offset_m"] = f"{res.lateral_offset_m:+.2f}"
        row[f"{side}_n"] = res.n_samples
        row[f"{side}_valid_n"] = res.n_valid
        row[f"{side}_present_n"] = res.n_present
        row[f"{side}_low_contrast_n"] = res.n_low_contrast
        row[f"{side}_low_snr_n"] = res.n_low_snr
        self._label_counts[side[0].upper()][label] += 1
      self._csv.writerow(row)
      self._csv_file.flush()
    except (OSError, ValueError):
      cloudlog.exception("lane_line_logger: csv write failed")

  def snapshot_due(self, side: str, label: str) -> bool:
    if not self.is_active or self._snap_count >= SNAPSHOT_SESSION_CAP:
      return False
    return time.monotonic() - self._snap_last.get((side, label), 0.0) >= SNAPSHOT_MIN_INTERVAL_S

  def save_snapshot(self, frame_y: np.ndarray, side: str, label: str, duty: float,
                    geometry, present: np.ndarray) -> str | None:
    """Save an untouched classifier input plus an annotated diagnostic view."""
    if not self.snapshot_due(side, label):
      return None
    try:
      from PIL import Image, ImageDraw
      name = f"{time.strftime('%H%M%S')}_{side}_{label}_d{int(round(duty * 100)):03d}.jpg"
      base = os.path.splitext(name)[0]
      # Save the exact frame first, before any overlay pixels are drawn. PNG is
      # lossless: JPEG artefacts can move a borderline contrast decision.
      raw = Image.fromarray(np.ascontiguousarray(frame_y), "L")
      raw.save(os.path.join(self._dir, base + ".raw.png"), "PNG")

      img = raw.copy()
      draw = ImageDraw.Draw(img)
      if geometry is not None:
        centre, rails = geometry
        for rail in rails:
          self._draw_polyline(draw, rail, fill=255, width=2)
        n = min(len(centre), len(present))
        for i in range(n):
          u, v = centre[i]
          if np.isfinite(u) and np.isfinite(v):
            fill = 255 if present[i] else 0
            draw.rectangle((u - 2, v - 2, u + 2, v + 2), fill=fill)

      img.save(os.path.join(self._dir, name), "JPEG", quality=80)
      self._snap_last[(side, label)] = time.monotonic()
      self._snap_count += 1
      return name
    except (OSError, ValueError, ImportError):
      cloudlog.exception("lane_line_logger: snapshot failed")
      return None

  def save_raw(self, snapshot_name: str, raw: dict) -> None:
    """Write the classifier inputs for a snapshot as a matching .json sidecar.

    Small (a couple of polylines + calib) and only written when a snapshot is,
    so it inherits the snapshot rate-limit and session cap.
    """
    if not self.is_active:
      return
    try:
      base = os.path.splitext(snapshot_name)[0]
      with open(os.path.join(self._dir, base + ".json"), "w") as f:
        json.dump(raw, f, separators=(",", ":"), default=float)
    except (OSError, TypeError, ValueError):
      cloudlog.exception("lane_line_logger: raw sidecar write failed")

  @staticmethod
  def _draw_polyline(draw, uv: np.ndarray, fill: int, width: int) -> None:
    """Draw a projected polyline, splitting at NaN (behind-camera) points."""
    run: list[tuple[float, float]] = []
    for u, v in uv:
      if np.isfinite(u) and np.isfinite(v):
        run.append((float(u), float(v)))
      else:
        if len(run) >= 2:
          draw.line(run, fill=fill, width=width)
        run = []
    if len(run) >= 2:
      draw.line(run, fill=fill, width=width)

  def end(self) -> str | None:
    """Zip the session, queue it for email, remove the directory."""
    if not self.is_active:
      return None
    session_dir, self._dir = self._dir, None
    try:
      with open(os.path.join(session_dir, "summary.json"), "w") as f:
        json.dump({"labels": self._label_counts, "snapshots": self._snap_count}, f, indent=2)
    except OSError:
      cloudlog.exception("lane_line_logger: summary write failed")
    self._close_files()
    return _zip_and_queue(session_dir)

  def _close_files(self) -> None:
    if self._csv_file is not None:
      try:
        self._csv_file.close()
      except OSError:
        pass
    self._csv_file = None
    self._csv = None


def _zip_and_queue(session_dir: str) -> str | None:
  try:
    if not os.path.isdir(session_dir):
      return None
    # nothing beyond config.json -> no assessments were logged; drop silently
    if not os.path.isfile(os.path.join(session_dir, "assessments.csv")):
      shutil.rmtree(session_dir, ignore_errors=True)
      return None
    stamp = os.path.basename(session_dir)[len(SESSION_PREFIX):]
    zip_base = os.path.join(LOG_DIR, f"{ZIP_PREFIX}{stamp}")
    zip_path = shutil.make_archive(zip_base, "zip", session_dir)
    shutil.rmtree(session_dir, ignore_errors=True)
    _prune_zips()

    # lazy import: mailer also (lazily) imports this module for orphan cleanup
    from sunnypilot.nkaoud_nav import mailer
    mailer.queue_log(zip_path)
    return zip_path
  except (OSError, ValueError):
    cloudlog.exception("lane_line_logger: zip/queue failed")
    return None


def _prune_zips() -> None:
  """Drop the oldest zips beyond MAX_STORED_ZIPS so a broken email setup can't
  fill the drive. The mailer tolerates queued paths that have vanished."""
  try:
    zips = sorted(e.path for e in os.scandir(LOG_DIR)
                  if e.is_file() and e.name.startswith(ZIP_PREFIX) and e.name.endswith(".zip"))
    for path in zips[:-MAX_STORED_ZIPS]:
      os.remove(path)
  except OSError:
    cloudlog.exception("lane_line_logger: prune failed")


def finalize_orphan_sessions() -> int:
  """Zip + queue session dirs left behind by a mid-drive shutdown."""
  count = 0
  try:
    if not os.path.isdir(LOG_DIR):
      return 0
    for entry in sorted(os.scandir(LOG_DIR), key=lambda e: e.name):
      if entry.is_dir() and entry.name.startswith(SESSION_PREFIX):
        if _zip_and_queue(entry.path):
          count += 1
  except OSError:
    cloudlog.exception("lane_line_logger: orphan finalize failed")
  return count
