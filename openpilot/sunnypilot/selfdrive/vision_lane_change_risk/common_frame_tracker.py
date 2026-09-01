from __future__ import annotations

import json
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
EDGE_FEATHER_PX = 18.0
THETA_FEATHER_RAD = np.deg2rad(3.0)


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
CALIBRATION_JSON_PATH = "/data/comma-360-viewer/calibration.json"


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


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
  t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
  return t * t * (3.0 - 2.0 * t)


def _bilinear_sample(frame: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
  h, w = frame.shape
  x = np.clip(u * (w - 1), 0.0, w - 1)
  y = np.clip(v * (h - 1), 0.0, h - 1)

  x0 = np.floor(x).astype(np.int32)
  y0 = np.floor(y).astype(np.int32)
  x1 = np.minimum(x0 + 1, w - 1)
  y1 = np.minimum(y0 + 1, h - 1)

  wx = x - x0
  wy = y - y0
  top = (1.0 - wx) * frame[y0, x0].astype(np.float32) + wx * frame[y0, x1].astype(np.float32)
  bottom = (1.0 - wx) * frame[y1, x0].astype(np.float32) + wx * frame[y1, x1].astype(np.float32)
  return ((1.0 - wy) * top + wy * bottom).astype(np.float32)


def _coverage_weight(u: np.ndarray, v: np.ndarray, theta: np.ndarray, limit: np.ndarray | float, width: int, height: int) -> np.ndarray:
  edge_u = np.minimum(u, 1.0 - u) * width
  edge_v = np.minimum(v, 1.0 - v) * height
  edge_weight = _smoothstep(0.0, EDGE_FEATHER_PX, np.minimum(edge_u, edge_v))
  theta_weight = _smoothstep(0.0, THETA_FEATHER_RAD, limit - theta)
  return (edge_weight * theta_weight).astype(np.float32)


def _sample_fisheye(frame: np.ndarray, cal: FisheyeCalibration) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
  sampled = _bilinear_sample(frame, u, v)
  weight = np.where(valid, _coverage_weight(u, v, theta, limit, w, h), 0.0)
  return sampled, valid, weight


def orient_common_frame(frame: np.ndarray) -> np.ndarray:
  # Keep the tracker frame, debug PNG, and calibration tuning view in one coordinate system.
  # Requested orientation: rotate 180 degrees, then flip horizontally.
  return np.fliplr(np.rot90(frame, 2))


def _json_float(data: dict, key: str, default: float) -> float:
  value = data.get(key, default)
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def load_calibrations_from_json(path: str = CALIBRATION_JSON_PATH) -> dict[str, FisheyeCalibration]:
  calibrations = dict(CAMERA_CALIBRATIONS)
  try:
    with open(path) as f:
      data = json.load(f)
  except (OSError, json.JSONDecodeError):
    return calibrations

  wide = calibrations["wide"]
  cabin = calibrations["cabin"]
  narrow = calibrations["narrow"]
  calibrations["wide"] = FisheyeCalibration(
    yaw_deg=_json_float(data, "frontYaw", wide.yaw_deg),
    pitch_deg=_json_float(data, "frontPitch", wide.pitch_deg),
    roll_deg=_json_float(data, "frontRoll", wide.roll_deg),
    focal=_json_float(data, "frontFocal", wide.focal),
    max_theta_deg=_json_float(data, "frontMaxTheta", wide.max_theta_deg),
    flip_x=_json_float(data, "frontFlipX", wide.flip_x),
    pan_x=_json_float(data, "frontPanX", wide.pan_x * 100.0) / 100.0,
    pan_y=_json_float(data, "frontPanY", wide.pan_y * 100.0) / 100.0,
    pan_z=_json_float(data, "frontPanZ", wide.pan_z * 100.0) / 100.0,
    max_theta_bias_deg=_json_float(data, "frontMaxThetaBias", wide.max_theta_bias_deg),
  )
  calibrations["cabin"] = FisheyeCalibration(
    yaw_deg=_json_float(data, "driverYaw", cabin.yaw_deg),
    pitch_deg=_json_float(data, "driverPitch", cabin.pitch_deg),
    roll_deg=_json_float(data, "driverRoll", cabin.roll_deg),
    focal=_json_float(data, "driverFocal", cabin.focal),
    max_theta_deg=_json_float(data, "driverMaxTheta", cabin.max_theta_deg),
    flip_x=_json_float(data, "driverFlipX", cabin.flip_x),
    pan_x=_json_float(data, "driverPanX", cabin.pan_x * 100.0) / 100.0,
    pan_y=_json_float(data, "driverPanY", cabin.pan_y * 100.0) / 100.0,
    pan_z=_json_float(data, "driverPanZ", cabin.pan_z * 100.0) / 100.0,
  )
  calibrations["narrow"] = FisheyeCalibration(
    yaw_deg=_json_float(data, "narrowYaw", narrow.yaw_deg),
    pitch_deg=_json_float(data, "narrowPitch", narrow.pitch_deg),
    roll_deg=_json_float(data, "narrowRoll", narrow.roll_deg),
    focal=_json_float(data, "narrowFocal", narrow.focal),
    max_theta_deg=_json_float(data, "narrowMaxTheta", narrow.max_theta_deg),
    flip_x=_json_float(data, "narrowFlipX", narrow.flip_x),
    pan_x=_json_float(data, "narrowPanX", narrow.pan_x * 100.0) / 100.0,
    pan_y=_json_float(data, "narrowPanY", narrow.pan_y * 100.0) / 100.0,
    pan_z=_json_float(data, "narrowPanZ", narrow.pan_z * 100.0) / 100.0,
  )
  return calibrations


def compose_common_frame(frames: dict[str, np.ndarray], calibrations: dict[str, FisheyeCalibration] | None = None) -> np.ndarray | None:
  if not frames:
    return None

  if calibrations is None:
    calibrations = CAMERA_CALIBRATIONS

  accum = np.zeros((GRID_H, GRID_W), dtype=np.float32)
  weight_sum = np.zeros((GRID_H, GRID_W), dtype=np.float32)
  valid_any = np.zeros((GRID_H, GRID_W), dtype=bool)

  # Soft equivalent of the 360 viewer's normal mode: narrow > wide > driver/cabin,
  # with feathered overlap so calibration errors are easier to tune from PNGs.
  for name, priority in (("cabin", 1.0), ("wide", 2.0), ("narrow", 4.0)):
    frame = frames.get(name)
    cal = calibrations.get(name)
    if frame is None or cal is None:
      continue

    sampled, valid, weight = _sample_fisheye(frame, cal)
    weight *= priority
    accum += sampled * weight
    weight_sum += weight
    valid_any |= valid

  if not np.any(valid_any):
    return None
  common = np.divide(accum, weight_sum, out=np.zeros_like(accum), where=weight_sum > 1e-6)
  common = np.clip(common, 0.0, 255.0).astype(np.uint8)
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
