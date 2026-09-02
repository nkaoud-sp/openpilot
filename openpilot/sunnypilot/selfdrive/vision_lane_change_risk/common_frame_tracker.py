from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass

import numpy as np


RAW_STRIP_PANEL_W = 512
RAW_STRIP_PANEL_H = 512
GRID_W = RAW_STRIP_PANEL_W * 4
GRID_H = RAW_STRIP_PANEL_H
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
DEBUG_SCALE = 1
TUNED_LEFT_PANEL_CENTER = (367, 425)
TUNED_RIGHT_PANEL_CENTER = (1703, 409)
TUNED_LEFT_ROTATION_DEG = -36.0
TUNED_RIGHT_ROTATION_DEG = 36.0


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


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
  ys = np.linspace(0, frame.shape[0] - 1, height).astype(np.int32)
  xs = np.linspace(0, frame.shape[1] - 1, width).astype(np.int32)
  return frame[np.ix_(ys, xs)]


def compose_raw_strip(frames: dict[str, np.ndarray]) -> np.ndarray | None:
  cabin = frames.get("cabin")
  wide = frames.get("wide")
  if cabin is None and wide is None:
    return None

  blank = np.zeros((RAW_STRIP_PANEL_H, RAW_STRIP_PANEL_W), dtype=np.uint8)
  if cabin is not None:
    cabin_panel = resize_frame(cabin, RAW_STRIP_PANEL_W * 2, RAW_STRIP_PANEL_H)
    left_dm = cabin_panel[:, :RAW_STRIP_PANEL_W]
    right_dm = cabin_panel[:, RAW_STRIP_PANEL_W:]
  else:
    left_dm = blank
    right_dm = blank

  front = resize_frame(wide, RAW_STRIP_PANEL_W * 2, RAW_STRIP_PANEL_H) if wide is not None else np.zeros(
    (RAW_STRIP_PANEL_H, RAW_STRIP_PANEL_W * 2), dtype=np.uint8
  )
  return np.hstack((left_dm, front, right_dm)).astype(np.uint8)


def _paste_rotated_panel(canvas: np.ndarray, panel: np.ndarray, center_x: int, center_y: int, angle_deg: float) -> None:
  src_h, src_w = panel.shape[:2]
  dst_h, dst_w = canvas.shape[:2]
  angle = np.deg2rad(angle_deg)
  cos_a = float(np.cos(angle))
  sin_a = float(np.sin(angle))

  half_w = int(np.ceil((abs(src_w * cos_a) + abs(src_h * sin_a)) * 0.5)) + 2
  half_h = int(np.ceil((abs(src_w * sin_a) + abs(src_h * cos_a)) * 0.5)) + 2
  x0 = max(0, center_x - half_w)
  x1 = min(dst_w, center_x + half_w)
  y0 = max(0, center_y - half_h)
  y1 = min(dst_h, center_y + half_h)
  if x0 >= x1 or y0 >= y1:
    return

  yy, xx = np.mgrid[y0:y1, x0:x1]
  dx = xx.astype(np.float32) - center_x
  dy = yy.astype(np.float32) - center_y
  src_x = cos_a * dx + sin_a * dy + (src_w - 1) * 0.5
  src_y = -sin_a * dx + cos_a * dy + (src_h - 1) * 0.5
  valid = (src_x >= 0.0) & (src_x <= src_w - 1) & (src_y >= 0.0) & (src_y <= src_h - 1)
  if not np.any(valid):
    return

  sx0 = np.clip(np.floor(src_x).astype(np.int32), 0, src_w - 1)
  sy0 = np.clip(np.floor(src_y).astype(np.int32), 0, src_h - 1)
  sx1 = np.minimum(sx0 + 1, src_w - 1)
  sy1 = np.minimum(sy0 + 1, src_h - 1)
  if panel.ndim == 3:
    wx = (src_x - sx0)[..., None]
    wy = (src_y - sy0)[..., None]
  else:
    wx = src_x - sx0
    wy = src_y - sy0
  top = (1.0 - wx) * panel[sy0, sx0].astype(np.float32) + wx * panel[sy0, sx1].astype(np.float32)
  bottom = (1.0 - wx) * panel[sy1, sx0].astype(np.float32) + wx * panel[sy1, sx1].astype(np.float32)
  sampled = ((1.0 - wy) * top + wy * bottom).astype(np.uint8)
  canvas_region = canvas[y0:y1, x0:x1]
  canvas_region[valid] = sampled[valid]


def compose_tuned_frame_from_raw(raw_strip: np.ndarray) -> np.ndarray:
  left_dm = raw_strip[:, :RAW_STRIP_PANEL_W]
  front = raw_strip[:, RAW_STRIP_PANEL_W:RAW_STRIP_PANEL_W * 3]
  right_dm = raw_strip[:, RAW_STRIP_PANEL_W * 3:]

  canvas = np.full(raw_strip.shape, 255, dtype=np.uint8)
  _paste_rotated_panel(
    canvas, right_dm, TUNED_LEFT_PANEL_CENTER[0], TUNED_LEFT_PANEL_CENTER[1], TUNED_LEFT_ROTATION_DEG
  )
  _paste_rotated_panel(
    canvas, left_dm, TUNED_RIGHT_PANEL_CENTER[0], TUNED_RIGHT_PANEL_CENTER[1], TUNED_RIGHT_ROTATION_DEG
  )
  canvas[:, RAW_STRIP_PANEL_W:RAW_STRIP_PANEL_W * 3] = front
  return canvas


def compose_tuned_frame(frames: dict[str, np.ndarray]) -> np.ndarray | None:
  raw_strip = compose_raw_strip(frames)
  return compose_tuned_frame_from_raw(raw_strip) if raw_strip is not None else None


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
  body = chunk_type + data
  return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)


def debug_frame_rgb(
  frame: np.ndarray,
  left_risk: bool,
  right_risk: bool,
  left_confidence: float,
  right_confidence: float,
) -> np.ndarray:
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

  return rgb


def write_rgb_png(path: str, rgb: np.ndarray) -> None:
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


def write_debug_png(
  path: str,
  frame: np.ndarray,
  left_risk: bool,
  right_risk: bool,
  left_confidence: float,
  right_confidence: float,
) -> None:
  write_rgb_png(path, debug_frame_rgb(frame, left_risk, right_risk, left_confidence, right_confidence))
