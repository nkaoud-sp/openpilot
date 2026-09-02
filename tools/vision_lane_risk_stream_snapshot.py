#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from openpilot.cereal.visionipc import VisionStreamType  # noqa: E402
from msgq.visionipc import VisionIpcClient  # noqa: E402


STREAM_SERVER_NAME = "vision_lane_change_riskd"
STREAM_TYPE = VisionStreamType.VISION_STREAM_MAP


def yuv420_to_rgb(buf) -> np.ndarray:
  width = buf.width
  height = buf.height
  y_size = width * height
  uv_size = y_size // 4
  data = np.frombuffer(buf.data, dtype=np.uint8)
  y = data[:y_size].reshape((height, width)).astype(np.float32)
  u = data[y_size:y_size + uv_size].reshape((height // 2, width // 2)).astype(np.float32)
  v = data[y_size + uv_size:y_size + uv_size * 2].reshape((height // 2, width // 2)).astype(np.float32)

  u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1) - 128.0
  v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1) - 128.0
  r = y + 1.402 * v
  g = y - 0.344136 * u - 0.714136 * v
  b = y + 1.772 * u
  return np.clip(np.stack((r, g, b), axis=2), 0.0, 255.0).astype(np.uint8)


def main() -> None:
  parser = argparse.ArgumentParser(description="Save one frame from the live vision lane risk processed stream.")
  parser.add_argument("--output", type=Path, default=Path("/data/media/0/vision_lane_change_risk_stream.png"))
  args = parser.parse_args()

  client = VisionIpcClient(STREAM_SERVER_NAME, STREAM_TYPE, True)
  while not client.connect(False):
    time.sleep(0.1)
  buf = client.recv()
  Image.fromarray(yuv420_to_rgb(buf)).save(args.output)
  print(f"Wrote {args.output}")


if __name__ == "__main__":
  main()
