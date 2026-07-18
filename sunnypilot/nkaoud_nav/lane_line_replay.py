#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Offline replay for lane-line troubleshooting sessions.

Each snapshot in a session (see lane_line_logger.py) has a matching ``.json``
sidecar carrying the exact classifier inputs for that frame: both ego lane-line
polylines, the calibration rpy, the camera intrinsics and offset. This tool
rebuilds the device->image transform and re-runs ``classify_line`` against the
saved Y-plane image, so a borderline result can be diagnosed as an algorithm
miss vs. genuinely worn/absent paint - and tuning changes can be checked
against real frames instead of synthetic ones.

Pure numpy + Pillow; no cereal/openpilot runtime needed.

Usage:
    python -m sunnypilot.nkaoud_nav.lane_line_replay <session_dir_or_zip> [--cfg]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import tempfile
import zipfile

import numpy as np

from openpilot.common.transformations.camera import view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler
from sunnypilot.selfdrive.controls.lib.lane_line_classifier import (
  classify_line, LaneLineClassifierConfig, LaneLineType,
)


def build_transform(rpy_calib, intrinsics) -> np.ndarray:
  """device/calib frame -> image pixel 3x3, identical to lane_line_classifierd."""
  view_from_calib = view_frame_from_device_frame @ rot_from_euler(np.asarray(rpy_calib, dtype=np.float64))
  return np.asarray(intrinsics, dtype=np.float64) @ view_from_calib


def config_from_session(session_dir: str) -> LaneLineClassifierConfig:
  """Rebuild the classifier config the session ran with (config.json)."""
  path = os.path.join(session_dir, "config.json")
  if not os.path.isfile(path):
    return LaneLineClassifierConfig()
  with open(path) as f:
    cfg = json.load(f).get("config", {})
  fields = {f.name for f in dataclasses.fields(LaneLineClassifierConfig)}
  return LaneLineClassifierConfig(**{k: v for k, v in cfg.items() if k in fields})


def replay_frame(session_dir: str, sidecar: str, cfg: LaneLineClassifierConfig):
  """Re-run both ego lines for one sidecar; returns a list of result rows."""
  from PIL import Image

  with open(os.path.join(session_dir, sidecar)) as f:
    raw = json.load(f)
  jpg = os.path.splitext(sidecar)[0] + ".jpg"
  frame_y = np.asarray(Image.open(os.path.join(session_dir, jpg)).convert("L"))
  transform = build_transform(raw["rpy_calib"], raw["intrinsics"])
  offset = float(raw.get("camera_offset", 0.0))

  rows = []
  for side in ("L", "R"):
    line = raw["lines"].get(side)
    if line is None:
      continue
    res = classify_line(frame_y, line["x"], line["y"], line["z"], transform, offset, cfg)
    rows.append({
      "sidecar": sidecar, "side": side,
      "type": LaneLineType(res.line_type).name,
      "duty": res.duty, "period_m": res.period_m,
      "valid_frac": res.valid_frac, "offset_m": res.lateral_offset_m,
      "prob": line.get("prob"),
    })
  return rows


def _as_session_dir(path: str, stack):
  if os.path.isdir(path):
    return path
  if zipfile.is_zipfile(path):
    tmp = stack.enter_context(tempfile.TemporaryDirectory())
    with zipfile.ZipFile(path) as z:
      z.extractall(tmp)
    # a session zip may contain the files at the root or under one dir
    entries = os.listdir(tmp)
    if len(entries) == 1 and os.path.isdir(os.path.join(tmp, entries[0])):
      return os.path.join(tmp, entries[0])
    return tmp
  raise SystemExit(f"not a session dir or zip: {path}")


def main() -> None:
  import contextlib

  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("session", help="session directory or .zip")
  ap.add_argument("--cfg", action="store_true", help="print the replayed config")
  args = ap.parse_args()

  with contextlib.ExitStack() as stack:
    session_dir = _as_session_dir(args.session, stack)
    cfg = config_from_session(session_dir)
    if args.cfg:
      print(json.dumps(dataclasses.asdict(cfg), indent=2, default=float))
    sidecars = sorted(f for f in os.listdir(session_dir) if f.endswith(".json")
                      and f not in ("config.json", "summary.json"))
    if not sidecars:
      raise SystemExit("no raw sidecars found (needs a session logged with raw capture)")

    print(f"{'sidecar':<34} {'side':<4} {'type':<8} {'duty':>5} {'period':>7} "
          f"{'validf':>6} {'offset':>7} {'prob':>5}")
    for sc in sidecars:
      for r in replay_frame(session_dir, sc, cfg):
        prob = "-" if r["prob"] is None else f"{r['prob']:.2f}"
        print(f"{r['sidecar']:<34} {r['side']:<4} {r['type']:<8} {r['duty']:>5.2f} "
              f"{r['period_m']:>7.1f} {r['valid_frac']:>6.2f} {r['offset_m']:>+7.2f} {prob:>5}")


if __name__ == "__main__":
  main()
