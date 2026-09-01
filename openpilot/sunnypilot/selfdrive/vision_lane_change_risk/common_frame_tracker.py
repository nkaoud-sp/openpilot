from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass

import numpy as np


GRID_W = 96
GRID_H = 36
MIN_TRACK_AGE = 3
CONFIDENCE_ON = 0.58
CONFIDENCE_OFF = 0.38


@dataclass(frozen=True)
class Region:
  x0: float
  y0: float
  x1: float
  y1: float


LEFT_CONFLICT = Region(0.02, 0.42, 0.30, 0.90)
RIGHT_CONFLICT = Region(0.70, 0.42, 0.98, 0.90)
DEBUG_SCALE = 8


def region_pixels(region: Region, width: int, height: int) -> tuple[int, int, int, int]:
  x0 = max(0, min(width - 1, int(region.x0 * width)))
  x1 = max(x0 + 1, min(width, int(region.x1 * width)))
  y0 = max(0, min(height - 1, int(region.y0 * height)))
  y1 = max(y0 + 1, min(height, int(region.y1 * height)))
  return x0, y0, x1, y1


class SideTracker:
  def __init__(self) -> None:
    self.confidence = 0.0
    self.track_age = 0
    self.risk = False

  def update(self, score: float) -> None:
    self.confidence = float(np.clip(0.70 * self.confidence + 0.30 * score, 0.0, 1.0))
    if self.confidence >= CONFIDENCE_ON:
      self.track_age = min(self.track_age + 1, 65535)
    elif self.confidence <= CONFIDENCE_OFF:
      self.track_age = 0

    self.risk = self.track_age >= MIN_TRACK_AGE


class CommonFrameMotionTracker:
  def __init__(self) -> None:
    self.background: np.ndarray | None = None
    self.left = SideTracker()
    self.right = SideTracker()

  @staticmethod
  def _region_score(diff: np.ndarray, global_motion: float, region: Region) -> float:
    h, w = diff.shape
    x0, y0, x1, y1 = region_pixels(region, w, h)
    roi = diff[y0:y1, x0:x1]

    # Use excess local motion over global brightness/camera motion. This is a
    # cheap first proxy for persistent occupancy in the lane-change side zones.
    excess = np.maximum(roi - global_motion, 0.0)
    motion_fraction = float(np.mean(excess > 5.0))
    motion_strength = float(np.clip(np.mean(excess) / 20.0, 0.0, 1.0))
    return float(np.clip(0.65 * motion_fraction + 0.35 * motion_strength, 0.0, 1.0))

  def update(self, frame: np.ndarray) -> None:
    frame_f = frame.astype(np.float32)
    if self.background is None or self.background.shape != frame.shape:
      self.background = frame_f
      self.left.update(0.0)
      self.right.update(0.0)
      return

    diff = np.abs(frame_f - self.background)
    global_motion = float(np.percentile(diff, 50))
    self.left.update(self._region_score(diff, global_motion, LEFT_CONFLICT))
    self.right.update(self._region_score(diff, global_motion, RIGHT_CONFLICT))
    self.background = 0.95 * self.background + 0.05 * frame_f


def resize_grid(frame: np.ndarray, width: int, height: int = GRID_H) -> np.ndarray:
  ys = np.linspace(0, frame.shape[0] - 1, height).astype(np.int32)
  xs = np.linspace(0, frame.shape[1] - 1, width).astype(np.int32)
  return frame[np.ix_(ys, xs)]


def compose_common_frame(frames: dict[str, np.ndarray]) -> np.ndarray | None:
  if "wide" in frames:
    common = frames["wide"].copy()
  elif "narrow" in frames:
    common = frames["narrow"].copy()
  elif "cabin" in frames:
    common = frames["cabin"].copy()
  else:
    return None

  # Treat the common frame as one tracking canvas:
  # - wide road supplies the full left/right context
  # - narrow road sharpens the forward center
  # - cabin camera is retained as a rear/context band when available
  if "narrow" in frames:
    center_w = GRID_W // 3
    x0 = (GRID_W - center_w) // 2
    common[:, x0:x0 + center_w] = resize_grid(frames["narrow"], center_w)

  if "cabin" in frames:
    band_h = GRID_H // 3
    cabin_band = resize_grid(frames["cabin"], GRID_W, band_h)
    common[-band_h:, :] = ((0.55 * common[-band_h:, :]) + (0.45 * cabin_band)).astype(np.uint8)

  return common


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
  body = chunk_type + data
  return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)


def write_debug_png(
  path: str,
  frame: np.ndarray,
  left_risk: bool,
  right_risk: bool,
  left_confidence: float,
  right_confidence: float,
) -> None:
  image = np.repeat(np.repeat(frame, DEBUG_SCALE, axis=0), DEBUG_SCALE, axis=1)
  rgb = np.repeat(image[:, :, None], 3, axis=2).astype(np.uint8)

  h, w = frame.shape
  for region, risk in ((LEFT_CONFLICT, left_risk), (RIGHT_CONFLICT, right_risk)):
    x0, y0, x1, y1 = region_pixels(region, w, h)
    x0 *= DEBUG_SCALE
    x1 *= DEBUG_SCALE
    y0 *= DEBUG_SCALE
    y1 *= DEBUG_SCALE
    color = np.array([255, 64, 64] if risk else [255, 210, 64], dtype=np.uint8)
    thickness = 2
    rgb[y0:y0 + thickness, x0:x1] = color
    rgb[y1 - thickness:y1, x0:x1] = color
    rgb[y0:y1, x0:x0 + thickness] = color
    rgb[y0:y1, x1 - thickness:x1] = color

  # Tiny confidence bars along the top edge: left on the left, right on the right.
  bar_h = 4
  left_w = int(np.clip(left_confidence, 0.0, 1.0) * rgb.shape[1] * 0.35)
  right_w = int(np.clip(right_confidence, 0.0, 1.0) * rgb.shape[1] * 0.35)
  rgb[:bar_h, :left_w] = np.array([255, 210, 64], dtype=np.uint8)
  if right_w > 0:
    rgb[:bar_h, -right_w:] = np.array([255, 210, 64], dtype=np.uint8)

  raw = b"".join(b"\x00" + row.tobytes() for row in rgb)
  png = (
    b"\x89PNG\r\n\x1a\n" +
    _png_chunk(b"IHDR", struct.pack(">IIBBBBB", rgb.shape[1], rgb.shape[0], 8, 2, 0, 0, 0)) +
    _png_chunk(b"IDAT", zlib.compress(raw, level=1)) +
    _png_chunk(b"IEND", b"")
  )
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "wb") as f:
    f.write(png)
