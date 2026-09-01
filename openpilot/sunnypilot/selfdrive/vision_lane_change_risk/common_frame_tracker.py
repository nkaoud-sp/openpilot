from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass

import numpy as np


GRID_W = 1024
GRID_H = 512
MIN_TRACK_AGE = 3
CONFIDENCE_ON = 0.58
CONFIDENCE_OFF = 0.38
PANORAMA_PITCH_TOP = np.deg2rad(55.0)
PANORAMA_PITCH_BOTTOM = np.deg2rad(-55.0)


@dataclass(frozen=True)
class Region:
  x0: float
  y0: float
  x1: float
  y1: float


LEFT_CONFLICT = Region(0.02, 0.42, 0.30, 0.90)
RIGHT_CONFLICT = Region(0.70, 0.42, 0.98, 0.90)
DEBUG_SCALE = 1


@dataclass(frozen=True)
class FisheyeCalibration:
  yaw_deg: float
  pitch_deg: float
  roll_deg: float
  focal: float
  max_theta_deg: float
  flip_x: float
  pan_x: float = 0.0
  pan_y: float = 0.0
  pan_z: float = 0.0
  max_theta_bias_deg: float = 0.0


CAMERA_CALIBRATIONS = {
  "wide": FisheyeCalibration(
    yaw_deg=180.0,
    pitch_deg=6.5,
    roll_deg=0.0,
    focal=0.29,
    max_theta_deg=86.0,
    flip_x=-1.0,
    pan_x=-0.20,
    max_theta_bias_deg=6.0,
  ),
  "cabin": FisheyeCalibration(
    yaw_deg=0.0,
    pitch_deg=14.0,
    roll_deg=0.0,
    focal=0.29,
    max_theta_deg=92.0,
    flip_x=-1.0,
    pan_y=-0.435,
    pan_z=0.03,
  ),
  "narrow": FisheyeCalibration(
    yaw_deg=180.0,
    pitch_deg=4.5,
    roll_deg=0.0,
    focal=1.22,
    max_theta_deg=40.0,
    flip_x=-1.0,
    pan_x=0.015,
  ),
}


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


def _rotation_matrix(cal: FisheyeCalibration) -> np.ndarray:
  # Match the comma-360-viewer shader path:
  # new THREE.Euler(pitch, yaw, roll, 'YXZ') -> Matrix4 -> Matrix3 -> invert().
  x = np.deg2rad(cal.pitch_deg)
  y = np.deg2rad(cal.yaw_deg)
  z = np.deg2rad(cal.roll_deg)

  a, b = np.cos(x), np.sin(x)
  c, d = np.cos(y), np.sin(y)
  e, f = np.cos(z), np.sin(z)
  ce, cf = c * e, c * f
  de, df = d * e, d * f

  mat = np.array((
    (ce + df * b, de * b - cf, a * d),
    (a * f, a * e, -b),
    (cf * b - de, df + ce * b, a * c),
  ), dtype=np.float32)
  return np.linalg.inv(mat).astype(np.float32)


def _panorama_dirs(width: int, height: int) -> np.ndarray:
  yaw = np.linspace(-np.pi, np.pi, width, endpoint=False, dtype=np.float32)
  pitch = np.linspace(PANORAMA_PITCH_TOP, PANORAMA_PITCH_BOTTOM, height, dtype=np.float32)
  yy, pp = np.meshgrid(yaw, pitch)

  cos_pitch = np.cos(pp)
  return np.stack((
    np.sin(yy) * cos_pitch,
    np.sin(pp),
    -np.cos(yy) * cos_pitch,
  ), axis=-1).astype(np.float32)


PANORAMA_DIRS = _panorama_dirs(GRID_W, GRID_H)


def _sample_fisheye(frame: np.ndarray, cal: FisheyeCalibration) -> tuple[np.ndarray, np.ndarray]:
  h, w = frame.shape
  world_pos = PANORAMA_DIRS - np.array((cal.pan_x, cal.pan_y, cal.pan_z), dtype=np.float32)
  cam_pos = world_pos @ _rotation_matrix(cal).T

  r_xy = np.linalg.norm(cam_pos[..., :2], axis=-1)
  theta = np.arctan2(r_xy, cam_pos[..., 2])
  limit = np.deg2rad(cal.max_theta_deg)
  if cal.max_theta_bias_deg:
    bias_dir = np.divide(cam_pos[..., 0], r_xy, out=np.zeros_like(r_xy), where=r_xy > 1e-4)
    limit = limit + np.deg2rad(cal.max_theta_bias_deg) * bias_dir

  uv_dir = np.divide(cam_pos[..., :2], r_xy[..., None], out=np.zeros_like(cam_pos[..., :2]), where=r_xy[..., None] > 1e-4)
  uv_img = uv_dir * theta[..., None]
  u = 0.5 + cal.flip_x * uv_img[..., 0] * cal.focal
  v = 0.5 + uv_img[..., 1] * cal.focal * 1.596

  valid = (theta <= limit) & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
  xs = np.clip((u * (w - 1)).astype(np.int32), 0, w - 1)
  ys = np.clip((v * (h - 1)).astype(np.int32), 0, h - 1)
  return frame[ys, xs], valid


def orient_common_frame(frame: np.ndarray) -> np.ndarray:
  # Keep the tracker frame, debug PNG, and calibration tuning view in one coordinate system.
  # Requested orientation: rotate 180 degrees, then flip horizontally.
  return np.fliplr(np.rot90(frame, 2))


def compose_common_frame(frames: dict[str, np.ndarray]) -> np.ndarray | None:
  if not frames:
    return None

  common = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
  valid_any = np.zeros((GRID_H, GRID_W), dtype=bool)

  # Same priority as the 360 viewer's normal mode: narrow > wide > driver/cabin.
  for name in ("cabin", "wide", "narrow"):
    frame = frames.get(name)
    cal = CAMERA_CALIBRATIONS.get(name)
    if frame is None or cal is None:
      continue

    sampled, valid = _sample_fisheye(frame, cal)
    common[valid] = sampled[valid]
    valid_any |= valid

  if not np.any(valid_any):
    return None
  return orient_common_frame(common)


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
