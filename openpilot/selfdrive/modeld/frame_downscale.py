"""Resample comma 3X camera frames to the comma four resolution before they cross the link to chestnut.

Since the warp moved onto the card, both full NV12 frames are copied over USB every model run. That
transfer is a fixed per-frame cost that depends only on the camera resolution:

  comma four  os04c10 @ 1344x760 (binned)  ~3.2 MB per frame pair
  comma 3X    ar0231/ox03c10 @ 1928x1208   ~7.5 MB per frame pair (~22 ms of the 50 ms budget)

The big model JIT is compiled per camera resolution, so a comma 3X that resamples its frames to the comma
four size on its own GPU runs the exact same JIT as a comma four, with the same amount of the frame
budget left for inference. The resample is a nearest-neighbour NV12 -> NV12 kernel run on the device GPU
(QCOM), written straight into the padded layout the comma four JIT expects. The warp matrices are scaled to match.

Nearest rather than bilinear on purpose: the model's own warp on the card samples nearest-neighbour at ~2 px
spacing, so with native frames the model already saw one nearest source pixel per model pixel. Resampling
nearest first gives it the same thing, while a bilinear kernel measured 5.5 ms per frame on the Adreno, most
of what the smaller transfer saves.

The kernel is captured at runtime rather than shipped as a pickle: tinygrad hash-conses buffer UOps by slot
number, so a pickled JIT loaded into a process that already holds same-shaped buffers can alias them. It is
two small kernels; tinygrad caches the compiled programs on disk, so only the first ever load pays a compile.
"""
from collections.abc import Iterable
import ctypes
import functools

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


def _axis_lut(src_len: int, dst_len: int, out_len: int, channels: int = 1) -> np.ndarray:
  # Pixel-center aligned nearest source index along one axis. Output positions past dst_len (stride/height
  # padding) replicate the last real pixel. With channels=2 the axis is an interleaved UV row: output byte j is
  # chroma pixel j//2, channel j%2, and the index addresses the matching source byte.
  j = np.arange(out_len)
  c = j % channels
  d = np.minimum(j // channels, dst_len - 1)
  i = np.clip(np.round((d + 0.5) * (src_len / dst_len) - 0.5), 0, src_len - 1).astype(np.int32)
  return i * channels + c


def make_downscale(src_size: tuple[int, int], dst_size: tuple[int, int], device: str | None = None):
  """Builds the tinygrad NV12 resample: uint8 source frame -> uint8 frame of nv12_copy_size(dst) bytes."""
  from tinygrad.device import Device
  from tinygrad.tensor import Tensor
  device = device or Device.DEFAULT

  src_w, src_h = src_size
  dst_w, dst_h = dst_size
  src_stride, src_y_height, _, _ = get_nv12_info(src_w, src_h)
  dst_stride, dst_y_height, dst_uv_height, _ = get_nv12_info(dst_w, dst_h)
  src_uv_offset = src_stride * src_y_height

  planes = [
    # (source plane byte offset, row index, column index)
    (0, _axis_lut(src_h, dst_h, dst_y_height), _axis_lut(src_w, dst_w, dst_stride)),
    (src_uv_offset, _axis_lut(src_h // 2, dst_h // 2, dst_uv_height), _axis_lut(src_w // 2, dst_w // 2, dst_stride, channels=2)),
  ]
  # the flat source index of every output byte, one gather per plane
  luts = [Tensor((rows[:, None] * src_stride + cols[None, :] + offset).reshape(-1).astype(np.int32), device=device).realize()
          for offset, rows, cols in planes]

  def downscale(frame):
    y, uv = (frame[idx].realize() for idx in luts)
    return y.cat(uv)

  return downscale


@functools.cache
def _dcache_invalidate():
  # Userspace clean+invalidate of the CPU data cache over a byte range, mirroring tinygrad's dcache_flush (ops_qcom.py).
  # Adreno writes bypass the CPU caches, so lines the CPU read from the output on the previous frame must be dropped
  # before it reads the new frame. dc civac is permitted at EL0 (SCTLR_EL1.UCI), unlike a plain invalidate.
  from tinygrad.device import Device
  from tinygrad.dtype import dtypes, AddrSpace
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  from tinygrad.codegen import to_program
  buf, n = UOp.param(0, dtypes.uint8, 1), UOp.param(1, dtypes.int, shape=(), name="n", addrspace=AddrSpace.ALU)
  i = UOp.range(n, 0, dtype=dtypes.int)
  inv = UOp(Ops.CUSTOM, src=(buf.index(i * 64),), arg=('__asm__ volatile("dc civac, %0" :: "r"({0}) : "memory");', dtypes.void))
  sink = UOp.sink(inv.end(i), UOp(Ops.CUSTOM, arg=('__asm__ volatile("dsb sy" ::: "memory");', dtypes.void)),
                  arg=KernelInfo(name="dcache_invalidate"), tag=1)
  prg = to_program(sink, Device["CPU"].renderer)
  return Device["CPU"].runtime(prg.to_elf())


class FrameDownscaler:
  """Runs the resample on the device GPU and hands back host frames in the target NV12 layout."""

  def __init__(self, src_size: tuple[int, int], dst_size: tuple[int, int], device: str, keys: Iterable[str] = ('img', 'big_img')):
    from tinygrad.device import Device
    from tinygrad.tensor import Tensor
    self._tensor = Tensor
    self._dev = Device[device]
    self.src_size, self.dst_size, self.device = src_size, dst_size, device
    self.scale = frame_scale_matrix(src_size, dst_size)
    self.src_buf_size = get_nv12_info(*src_size)[3]
    dst_stride, dst_y_height, dst_uv_height, self.dst_buf_size = get_nv12_info(*dst_size)
    self.copy_size = dst_stride * (dst_y_height + dst_uv_height)
    self._blob_cache: dict[int, object] = {}
    self._invalidate = _dcache_invalidate() if device.startswith('QCOM') else None

    # One persistent host frame per model input, written by the GPU directly: a pointer taken to it stays
    # valid across runs. QCOM allocations are mapped write-combined, and a CPU read of 1.6 MB of
    # write-combined memory (what .numpy() does) costs more on Snapdragon than the USB transfer this whole
    # thing saves, whereas reading cached memory is ~50x faster; the CPU cache just has to be invalidated over
    # the frame after every GPU write, see _dcache_invalidate.
    self._downscale = make_downscale(src_size, dst_size, device)
    self.frames: dict[str, np.ndarray] = {}
    self.jits: dict[str, object] = {}
    for key in keys:
      self._add_output(key)

  def _add_output(self, key: str) -> None:
    from tinygrad.engine.jit import TinyJit
    Tensor = self._tensor
    frame = np.zeros(self.dst_buf_size, dtype=np.uint8)
    frame[:] = 0  # fault the pages in before mapping, so the map-time cache clean covers them and nothing is dirty later
    out_t = Tensor.from_blob(frame.ctypes.data, (self.copy_size,), dtype='uint8', device=self.device)
    jit = TinyJit(lambda src: out_t.assign(self._downscale(src)).realize(), prune=True)
    for _ in range(3):  # capture the jit on dummy frames so the first camera frame replays it
      jit(Tensor.zeros(self.src_buf_size, dtype='uint8', device=self.device).contiguous().realize())
    self._dev.synchronize()
    self.frames[key], self.jits[key] = frame, jit

  def run(self, key: str, buf) -> np.ndarray:
    ptr = np.frombuffer(buf.data, dtype=np.uint8).ctypes.data
    if ptr not in self._blob_cache:
      self._blob_cache[ptr] = self._tensor.from_blob(ptr, (self.src_buf_size,), dtype='uint8', device=self.device)
    if key not in self.frames:
      self._add_output(key)
    frame = self.frames[key]
    self.jits[key](self._blob_cache[ptr])
    self._dev.synchronize()
    if self._invalidate is not None:
      addr = frame.ctypes.data
      self._invalidate.fxn(ctypes.c_uint64(addr & ~63), -(-(addr + self.copy_size - (addr & ~63)) // 64))
    return frame


if __name__ == "__main__":
  # on-device timing: python3 -m openpilot.selfdrive.modeld.frame_downscale [--device QCOM]
  import argparse
  import os
  import time
  import types
  p = argparse.ArgumentParser()
  # never ask tinygrad for a default device here: with a chestnut attached it probes the AMD card and fetches firmware
  p.add_argument('--device', default='QCOM' if os.path.exists('/dev/kgsl-3d0') else 'CPU')
  p.add_argument('--runs', type=int, default=50)
  args = p.parse_args()
  src, dst = (1928, 1208), CHESTNUT_FRAME_SIZE
  frame = np.random.default_rng(0).integers(0, 256, get_nv12_info(*src)[3], dtype=np.uint8)
  st = time.perf_counter()
  downscaler = FrameDownscaler(src, dst, args.device)
  print(f"init (kernel compile + jit capture): {(time.perf_counter() - st) * 1e3:.1f} ms")
  buf = types.SimpleNamespace(data=frame)
  downscaler.run('img', buf)
  timings = []
  for _ in range(args.runs):
    st = time.perf_counter()
    downscaler.run('img', buf)
    timings.append((time.perf_counter() - st) * 1e3)
  print(f"run (kernel + sync + copy): median {np.median(timings):.2f} ms, max {max(timings):.2f} ms over {args.runs} runs")
  st = time.perf_counter()
  for _ in range(args.runs):
    downscaler.jits['img'](downscaler._blob_cache[frame.ctypes.data])
    downscaler._dev.synchronize()
  print(f"kernel only: {(time.perf_counter() - st) / args.runs * 1e3:.2f} ms")
