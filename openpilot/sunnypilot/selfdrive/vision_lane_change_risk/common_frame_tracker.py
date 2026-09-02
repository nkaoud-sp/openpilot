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


@dataclass(frozen=True)
class TrackedObject:
  track_id: int
  side: str
  x0: int
  y0: int
  x1: int
  y1: int
  confidence: float
  age: int
  vx: float
  vy: float


@dataclass
class MotionTrack:
  track_id: int
  x0: int
  y0: int
  x1: int
  y1: int
  confidence: float
  age: int = 1
  missed: int = 0
  vx: float = 0.0
  vy: float = 0.0

  @property
  def center(self) -> tuple[float, float]:
    return (self.x0 + self.x1) * 0.5, (self.y0 + self.y1) * 0.5

  @property
  def predicted_center(self) -> tuple[float, float]:
    cx, cy = self.center
    return cx + self.vx, cy + self.vy

  @property
  def bbox(self) -> tuple[int, int, int, int]:
    return self.x0, self.y0, self.x1, self.y1


LEFT_CONFLICT = Region(0.02, 0.42, 0.30, 0.90)
RIGHT_CONFLICT = Region(0.70, 0.42, 0.98, 0.90)
DEBUG_SCALE = 1
TUNED_LEFT_PANEL_CENTER = (367, 425)
TUNED_RIGHT_PANEL_CENTER = (1703, 409)
TUNED_LEFT_ROTATION_DEG = -36.0
TUNED_RIGHT_ROTATION_DEG = 36.0
MOTION_THRESHOLD = 8.0
MIN_TRACK_PIXELS = 120
COMPONENT_CELL_SIZE = 8
MIN_COMPONENT_CELLS = 6
MAX_TRACK_MATCH_DISTANCE = 190.0
MAX_TRACK_MISSES = 4
MAX_OBJECT_TRACKS = 12
TRACK_COLOR = np.array([64, 220, 255], dtype=np.uint8)
TRACK_RISK_COLOR = np.array([255, 96, 64], dtype=np.uint8)


def region_pixels(region: Region, width: int, height: int) -> tuple[int, int, int, int]:
  x0 = max(0, min(width - 1, int(region.x0 * width)))
  x1 = max(x0 + 1, min(width, int(region.x1 * width)))
  y0 = max(0, min(height - 1, int(region.y0 * height)))
  y1 = max(y0 + 1, min(height, int(region.y1 * height)))
  return x0, y0, x1, y1


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
  x0, y0, x1, y1 = bbox
  return (x0 + x1) * 0.5, (y0 + y1) * 0.5


def bbox_intersects_region(bbox: tuple[int, int, int, int], region: Region, width: int, height: int) -> bool:
  x0, y0, x1, y1 = bbox
  rx0, ry0, rx1, ry1 = region_pixels(region, width, height)
  return x0 < rx1 and x1 > rx0 and y0 < ry1 and y1 > ry0


def bbox_side(bbox: tuple[int, int, int, int], width: int, height: int) -> str:
  if bbox_intersects_region(bbox, LEFT_CONFLICT, width, height):
    return "left"
  if bbox_intersects_region(bbox, RIGHT_CONFLICT, width, height):
    return "right"
  return "center"


class SideTracker:
  def __init__(self, side: str) -> None:
    self.side = side
    self.confidence = 0.0
    self.track_age = 0
    self.object_age = 0
    self.risk = False
    self.bbox: tuple[int, int, int, int] | None = None
    self.center: tuple[float, float] | None = None
    self.velocity = (0.0, 0.0)

  def update(self, score: float, bbox: tuple[int, int, int, int] | None) -> None:
    self.confidence = float(np.clip(0.70 * self.confidence + 0.30 * score, 0.0, 1.0))
    if self.confidence >= CONFIDENCE_ON:
      self.track_age = min(self.track_age + 1, 65535)
    elif self.confidence <= CONFIDENCE_OFF:
      self.track_age = 0

    self.risk = self.track_age >= MIN_TRACK_AGE
    if bbox is not None:
      self.object_age = min(self.object_age + 1, 65535)
      x0, y0, x1, y1 = bbox
      center = ((x0 + x1) * 0.5, (y0 + y1) * 0.5)
      self.velocity = (
        center[0] - self.center[0],
        center[1] - self.center[1],
      ) if self.center is not None else (0.0, 0.0)
      self.center = center
      self.bbox = bbox
    else:
      self.object_age = 0
      self.bbox = None
      self.center = None
      self.velocity = (0.0, 0.0)

  def tracked_object(self) -> TrackedObject | None:
    if self.bbox is None or self.object_age == 0:
      return None
    return TrackedObject(
      -1, self.side, *self.bbox, self.confidence, self.object_age, self.velocity[0], self.velocity[1]
    )


class CommonFrameMotionTracker:
  def __init__(self) -> None:
    self.background: np.ndarray | None = None
    self.left = SideTracker("left")
    self.right = SideTracker("right")
    self._tracks: list[MotionTrack] = []
    self._next_track_id = 1

  @staticmethod
  def _region_motion(
    diff: np.ndarray,
    global_motion: float,
    region: Region,
  ) -> tuple[float, tuple[int, int, int, int] | None]:
    h, w = diff.shape
    x0, y0, x1, y1 = region_pixels(region, w, h)
    roi = diff[y0:y1, x0:x1]

    # Use excess local motion over global brightness/camera motion. This is a
    # cheap first proxy for persistent occupancy in the lane-change side zones.
    excess = np.maximum(roi - global_motion, 0.0)
    motion_mask = excess > MOTION_THRESHOLD
    motion_fraction = float(np.mean(motion_mask))
    motion_strength = float(np.clip(np.mean(excess) / 20.0, 0.0, 1.0))
    score = float(np.clip(0.65 * motion_fraction + 0.35 * motion_strength, 0.0, 1.0))

    ys, xs = np.where(motion_mask)
    bbox = None
    if xs.size >= MIN_TRACK_PIXELS:
      bbox = (int(x0 + xs.min()), int(y0 + ys.min()), int(x0 + xs.max() + 1), int(y0 + ys.max() + 1))
    return score, bbox

  @staticmethod
  def _motion_components(motion_mask: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
    cell = COMPONENT_CELL_SIZE
    h, w = motion_mask.shape
    cell_h = h // cell
    cell_w = w // cell
    trimmed = motion_mask[:cell_h * cell, :cell_w * cell]
    cell_mask = trimmed.reshape((cell_h, cell, cell_w, cell)).any(axis=(1, 3))
    seen = np.zeros_like(cell_mask, dtype=bool)
    components: list[tuple[tuple[int, int, int, int], float]] = []

    for start_y, start_x in zip(*np.where(cell_mask & ~seen)):
      stack = [(int(start_y), int(start_x))]
      seen[start_y, start_x] = True
      xs: list[int] = []
      ys: list[int] = []

      while stack:
        cy, cx = stack.pop()
        xs.append(cx)
        ys.append(cy)
        for ny in (cy - 1, cy, cy + 1):
          for nx in (cx - 1, cx, cx + 1):
            if ny == cy and nx == cx:
              continue
            if 0 <= ny < cell_h and 0 <= nx < cell_w and cell_mask[ny, nx] and not seen[ny, nx]:
              seen[ny, nx] = True
              stack.append((ny, nx))

      if len(xs) < MIN_COMPONENT_CELLS:
        continue

      x0 = max(0, min(xs) * cell)
      y0 = max(0, min(ys) * cell)
      x1 = min(w, (max(xs) + 1) * cell)
      y1 = min(h, (max(ys) + 1) * cell)
      pixels = int(np.count_nonzero(motion_mask[y0:y1, x0:x1]))
      if pixels < MIN_TRACK_PIXELS:
        continue
      confidence = float(np.clip(pixels / 6000.0, 0.05, 1.0))
      components.append(((x0, y0, x1, y1), confidence))

    return sorted(components, key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]), reverse=True)

  def _update_object_tracks(self, detections: list[tuple[tuple[int, int, int, int], float]]) -> None:
    assigned_tracks: set[int] = set()
    assigned_detections: set[int] = set()
    matches: list[tuple[float, int, int]] = []

    for track_idx, track in enumerate(self._tracks):
      tx, ty = track.predicted_center
      for det_idx, (bbox, _) in enumerate(detections):
        dx = bbox_center(bbox)[0] - tx
        dy = bbox_center(bbox)[1] - ty
        dist = float(np.hypot(dx, dy))
        if dist <= MAX_TRACK_MATCH_DISTANCE:
          matches.append((dist, track_idx, det_idx))

    for _, track_idx, det_idx in sorted(matches):
      if track_idx in assigned_tracks or det_idx in assigned_detections:
        continue
      track = self._tracks[track_idx]
      bbox, confidence = detections[det_idx]
      old_cx, old_cy = track.center
      new_cx, new_cy = bbox_center(bbox)
      track.x0, track.y0, track.x1, track.y1 = bbox
      track.vx = 0.55 * track.vx + 0.45 * (new_cx - old_cx)
      track.vy = 0.55 * track.vy + 0.45 * (new_cy - old_cy)
      track.confidence = float(np.clip(0.65 * track.confidence + 0.35 * confidence, 0.0, 1.0))
      track.age = min(track.age + 1, 65535)
      track.missed = 0
      assigned_tracks.add(track_idx)
      assigned_detections.add(det_idx)

    for track_idx, track in enumerate(self._tracks):
      if track_idx not in assigned_tracks:
        track.missed += 1
        track.confidence *= 0.75

    for det_idx, (bbox, confidence) in enumerate(detections):
      if det_idx in assigned_detections:
        continue
      self._tracks.append(MotionTrack(self._next_track_id, *bbox, confidence))
      self._next_track_id += 1

    self._tracks = [track for track in self._tracks if track.missed <= MAX_TRACK_MISSES]
    self._tracks = sorted(
      self._tracks, key=lambda track: (track.age, track.confidence), reverse=True
    )[:MAX_OBJECT_TRACKS]

  def update(self, frame: np.ndarray) -> None:
    frame_f = frame.astype(np.float32)
    if self.background is None or self.background.shape != frame.shape:
      self.background = frame_f
      self.left.update(0.0, None)
      self.right.update(0.0, None)
      return

    diff = np.abs(frame_f - self.background)
    global_motion = float(np.percentile(diff, 50))
    motion_mask = np.maximum(diff - global_motion, 0.0) > MOTION_THRESHOLD
    left_score, left_bbox = self._region_motion(diff, global_motion, LEFT_CONFLICT)
    right_score, right_bbox = self._region_motion(diff, global_motion, RIGHT_CONFLICT)
    self.left.update(left_score, left_bbox)
    self.right.update(right_score, right_bbox)
    self._update_object_tracks(self._motion_components(motion_mask))
    self.background = 0.95 * self.background + 0.05 * frame_f

  @property
  def tracks(self) -> list[TrackedObject]:
    if self.background is None:
      return []
    return [
      TrackedObject(
        track.track_id,
        bbox_side(track.bbox, self.background.shape[1], self.background.shape[0]),
        track.x0,
        track.y0,
        track.x1,
        track.y1,
        track.confidence,
        track.age,
        track.vx,
        track.vy,
      )
      for track in self._tracks
    ]


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


def _draw_box(rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: np.ndarray, thickness: int = 2) -> None:
  x0 = max(0, min(rgb.shape[1] - 1, x0))
  x1 = max(x0 + 1, min(rgb.shape[1], x1))
  y0 = max(0, min(rgb.shape[0] - 1, y0))
  y1 = max(y0 + 1, min(rgb.shape[0], y1))
  rgb[y0:y0 + thickness, x0:x1] = color
  rgb[y1 - thickness:y1, x0:x1] = color
  rgb[y0:y1, x0:x0 + thickness] = color
  rgb[y0:y1, x1 - thickness:x1] = color


def debug_frame_rgb(
  frame: np.ndarray,
  left_risk: bool,
  right_risk: bool,
  left_confidence: float,
  right_confidence: float,
  tracks: list[TrackedObject] | None = None,
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
    _draw_box(rgb, x0, y0, x1, y1, color)

  for track in tracks or []:
    color = TRACK_RISK_COLOR if track.age >= MIN_TRACK_AGE else TRACK_COLOR
    _draw_box(
      rgb,
      track.x0 * DEBUG_SCALE,
      track.y0 * DEBUG_SCALE,
      track.x1 * DEBUG_SCALE,
      track.y1 * DEBUG_SCALE,
      color,
      thickness=3,
    )

  # Tiny confidence bars along the top edge: left on the left, right on the right.
  bar_h = 4
  left_w = int(np.clip(left_confidence, 0.0, 1.0) * rgb.shape[1] * 0.35)
  right_w = int(np.clip(right_confidence, 0.0, 1.0) * rgb.shape[1] * 0.35)
  rgb[:bar_h, :left_w] = np.array([255, 210, 64], dtype=np.uint8)
  if right_w > 0:
    rgb[:bar_h, -right_w:] = np.array([255, 210, 64], dtype=np.uint8)

  return rgb


def rgb_to_yuv420(rgb: np.ndarray) -> bytes:
  rgb_f = rgb.astype(np.float32)
  r = rgb_f[..., 0]
  g = rgb_f[..., 1]
  b = rgb_f[..., 2]
  y = np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0.0, 255.0).astype(np.uint8)
  u = np.clip(-0.169 * r - 0.331 * g + 0.500 * b + 128.0, 0.0, 255.0)
  v = np.clip(0.500 * r - 0.419 * g - 0.081 * b + 128.0, 0.0, 255.0)

  u420 = u.reshape((rgb.shape[0] // 2, 2, rgb.shape[1] // 2, 2)).mean(axis=(1, 3)).astype(np.uint8)
  v420 = v.reshape((rgb.shape[0] // 2, 2, rgb.shape[1] // 2, 2)).mean(axis=(1, 3)).astype(np.uint8)
  return y.tobytes() + u420.tobytes() + v420.tobytes()


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
  tracks: list[TrackedObject] | None = None,
) -> None:
  write_rgb_png(path, debug_frame_rgb(frame, left_risk, right_risk, left_confidence, right_confidence, tracks))
