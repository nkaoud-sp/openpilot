"""Resample comma 3X camera frames to the comma four resolution before they cross the link to chestnut.

Since the warp moved onto the card, both full NV12 frames are copied over USB every model run. That
transfer is a fixed per-frame cost that depends only on the camera resolution:

  comma four  os04c10 @ 1344x760 (binned)  ~3.2 MB per frame pair
  comma 3X    ar0231/ox03c10 @ 1928x1208   ~7.5 MB per frame pair (~22 ms of the 50 ms budget)

The big model JIT is compiled per camera resolution, so a comma 3X that resamples its frames to the comma
four size on its own GPU runs the exact same JIT as a comma four, with the same amount of the frame
budget left for inference. The resample is a bilinear NV12 -> NV12 kernel run on the device GPU (QCOM),
written straight into the padded layout the comma four JIT expects. The warp matrices are scaled to match.

The kernel is captured at runtime rather than shipped as a pickle: tinygrad hash-conses buffer UOps by slot
number, so a pickled JIT loaded into a process that already holds same-shaped buffers can alias them. It is
two small kernels; tinygrad caches the compiled programs on disk, so only the first ever load pays a compile.
"""
from collections.abc import Iterable

import numpy as np

from openpilot.common.transformations.camera import _os_fisheye
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

# comma four camera resolution: every released big model is sized and validated against it
CHESTNUT_FRAME_SIZE: tuple[int, int] = _os_fisheye.size


def downscale_target(cam_w: int, cam_h: int) -> tuple[int, int]:
  """Frame size a camera of this resolution should send to chestnut."""
  tw, th = CHESTNUT_FRAME_SIZE
  return (tw, th) if cam_w * cam_h > tw * th else (cam_w, cam_h)


def select_frame_size(available: Iterable[tuple[int, int]], cam_w: int, cam_h: int, native: bool = False) -> tuple[int, int]:
  """Pick the resolution the big model JIT runs at, from the resolutions a pkl was compiled for."""
  available_set = set(available)
  native_size, target = (cam_w, cam_h), downscale_target(cam_w, cam_h)
  for size in ((native_size, target) if native else (target, native_size)):
    if size in available_set:
      return size
  raise KeyError(f"no big model JIT for {cam_w}x{cam_h} or {target[0]}x{target[1]}, pkl has {sorted(available_set)}")


def frame_scale_matrix(src_size: tuple[int, int], dst_size: tuple[int, int]) -> np.ndarray:
  """Maps camera pixel coordinates of the source frame to the resampled frame; identity if the sizes match."""
  return np.diag([dst_size[0] / src_size[0], dst_size[1] / src_size[1], 1.0]).astype(np.float32)


def _axis_lut(src_len: int, dst_len: int, out_len: int, channels: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  # Pixel-center aligned bilinear taps along one axis. Output positions past dst_len (stride/height padding)
  # replicate the last real pixel. With channels=2 the axis is an interleaved UV row: output byte j is
  # chroma pixel j//2, channel j%2, and the taps address the matching source byte.
  j = np.arange(out_len)
  c = j % channels
  d = np.minimum(j // channels, dst_len - 1)
  s = np.clip((d + 0.5) * (src_len / dst_len) - 0.5, 0, src_len - 1)
  i0 = np.floor(s).astype(np.int32)
  i1 = np.minimum(i0 + 1, src_len - 1).astype(np.int32)
  f = (s - i0).astype(np.float32)
  return i0 * channels + c, i1 * channels + c, f


def make_downscale(src_size: tuple[int, int], dst_size: tuple[int, int], device: str | None = None):
  """Builds the tinygrad NV12 resample: uint8 source frame -> uint8 frame of nv12_copy_size(dst) bytes."""
  from tinygrad import dtypes
  from tinygrad.device import Device
  from tinygrad.tensor import Tensor
  device = device or Device.DEFAULT

  src_w, src_h = src_size
  dst_w, dst_h = dst_size
  src_stride, src_y_height, _, _ = get_nv12_info(src_w, src_h)
  dst_stride, dst_y_height, dst_uv_height, _ = get_nv12_info(dst_w, dst_h)
  src_uv_offset = src_stride * src_y_height

  planes = [
    # (source plane byte offset, row taps, column taps, output rows)
    (0, _axis_lut(src_h, dst_h, dst_y_height), _axis_lut(src_w, dst_w, dst_stride), dst_y_height),
    (src_uv_offset, _axis_lut(src_h // 2, dst_h // 2, dst_uv_height), _axis_lut(src_w // 2, dst_w // 2, dst_stride, channels=2), dst_uv_height),
  ]
  luts = [(offset, [Tensor(v, device=device).reshape(-1, 1).realize() for v in rows],
           [Tensor(v, device=device).reshape(1, -1).realize() for v in cols], out_rows) for offset, rows, cols, out_rows in planes]

  def resample_plane(src_flat, offset, rows, cols, out_rows):
    y0, y1, fy = rows
    x0, x1, fx = cols

    def tap(yi, xi):
      return src_flat[(yi * src_stride + xi + offset).reshape(-1)].reshape(out_rows, dst_stride).cast(dtypes.float32)
    top = tap(y0, x0) * (1 - fx) + tap(y0, x1) * fx
    bottom = tap(y1, x0) * (1 - fx) + tap(y1, x1) * fx
    return (top * (1 - fy) + bottom * fy + 0.5).cast(dtypes.uint8).reshape(-1)

  def downscale(frame):
    y, uv = (resample_plane(frame, *lut).realize() for lut in luts)
    return y.cat(uv)

  return downscale


class FrameDownscaler:
  """Runs the resample on the device GPU and hands back host frames in the target NV12 layout."""

  def __init__(self, src_size: tuple[int, int], dst_size: tuple[int, int], device: str):
    from tinygrad.engine.jit import TinyJit
    from tinygrad.tensor import Tensor
    self._tensor = Tensor
    self.src_size, self.dst_size, self.device = src_size, dst_size, device
    self.scale = frame_scale_matrix(src_size, dst_size)
    self.src_buf_size = get_nv12_info(*src_size)[3]
    dst_stride, dst_y_height, dst_uv_height, self.dst_buf_size = get_nv12_info(*dst_size)
    self.copy_size = dst_stride * (dst_y_height + dst_uv_height)
    self._blob_cache: dict[int, object] = {}
    # one persistent host frame per model input, so a pointer taken to it stays valid across runs
    self.frames: dict[str, np.ndarray] = {}
    self.jit = TinyJit(make_downscale(src_size, dst_size, device), prune=True)
    for _ in range(3):  # capture the jit on dummy frames so the first camera frame replays it
      self.jit(Tensor.zeros(self.src_buf_size, dtype='uint8', device=device).contiguous().realize()).realize()

  def run(self, key: str, buf) -> np.ndarray:
    ptr = np.frombuffer(buf.data, dtype=np.uint8).ctypes.data
    if ptr not in self._blob_cache:
      self._blob_cache[ptr] = self._tensor.from_blob(ptr, (self.src_buf_size,), dtype='uint8', device=self.device)
    if key not in self.frames:
      self.frames[key] = np.zeros(self.dst_buf_size, dtype=np.uint8)
    frame = self.frames[key]
    frame[:self.copy_size] = self.jit(self._blob_cache[ptr]).numpy()
    return frame
