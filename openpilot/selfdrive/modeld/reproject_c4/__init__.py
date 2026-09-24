"""Reproject the comma 3X cameras into comma 4 camera geometry, on the GPU, in front of an untouched comma 4 model.

Every comma 4 output pixel is traced through the comma 4 lens to a ray, rotated onto the 3X device axis and looked up in
the 3X wide (fisheye) or, for the narrow output, composited from the 3X narrow inset and the 3X wide surround. None of it
depends on the calibration, so the lookup tables are constants built once per process and the per-frame work is one
gather per output byte (two plus a blend for the composite) - the same cost as the model's own crop warp.

Outputs are NV12 buffers in the comma 4 camerad layout so the model pkl (keyed by its camera size) sees exactly what a
comma 4 would have given it. Lens numbers come from the bench + multi-device work in ~/tizi-to-mici.

geometry: lenses, rotations, sample coordinates. tables: the gather tables. rotation: the fitted rotation across boots.
fit: the rotation fit from frame pairs. meter: the seam exposure match. cameras: camerad's frame pairs. debug: the outlines.
kernel (the GPU stage, tinygrad) is imported by the stage alone."""
import ctypes

import numpy as np

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

from . import geometry as _geometry
from . import tables as _tables
from .geometry import *  # noqa: F403
from .meter import *  # noqa: F403
from .tables import *  # noqa: F403
from .rotation import *  # noqa: F403
from .fit import *  # noqa: F403
from .debug import *  # noqa: F403
from .geometry import C4_CAM, POP_ROTATION, calib_from_rotvec
from .meter import SeamMeter
from .rotation import load_rotation
from .tables import load_tables, table_cache_dir


def _default_calib():
  return calib_from_rotvec(POP_ROTATION)


def sample_coords(out_cam, dst_w, dst_h, scale=1.0, calib=None):
  """Compatibility wrapper for the old single-file API."""
  return _geometry.sample_coords(out_cam, dst_w, dst_h, scale, _default_calib() if calib is None else calib)


def build_tables(src_wh, dst_wh, calib=None):
  """Compatibility wrapper for the old single-file API."""
  return _tables.build_tables(src_wh, dst_wh, _default_calib() if calib is None else calib)


class Reprojector:
  """Chestnut frame stage: 3X wide+narrow NV12 in, comma 4 wide+narrow NV12 out.

  This wraps Amy's rotation-aware reproject_c4 kernel with this branch's frame-stage
  contract, so modeld can continue to choose between native/resample/raymatch modes.
  """

  ROAD_KEY, WIDE_KEY = 'img', 'big_img'
  scale = np.eye(3, dtype=np.float32)
  c4_intrinsics = True

  def __init__(self, src_size=(1928, 1208), dst_size=C4_CAM, device='QCOM', cache_dir=None):
    from tinygrad.device import Device
    from tinygrad.tensor import Tensor
    from openpilot.selfdrive.modeld.frame_downscale import dcache_invalidate
    from .kernel import Reprojector as KernelReprojector

    self._tensor, self._dev = Tensor, Device[device]
    self.src_size, self.dst_size, self.device = src_size, dst_size, device
    self.src_buf_size = get_nv12_info(*src_size)[3]
    self.dst_buf_size = get_nv12_info(*dst_size)[3]

    calib = calib_from_rotvec(load_rotation())
    T = load_tables(src_size, dst_size, cache_dir or table_cache_dir(), calib)
    dst_stride, dst_y_height, dst_uv_height, _ = get_nv12_info(*dst_size)
    self.copy_size = dst_stride * (dst_y_height + dst_uv_height)
    self.uv_offset = dst_stride * dst_y_height

    self._src_cache: dict[int, object] = {}
    self._invalidate = dcache_invalidate() if str(device).startswith('QCOM') else None
    self.frames = {k: np.zeros(self.dst_buf_size, dtype=np.uint8) for k in (self.ROAD_KEY, self.WIDE_KEY)}
    for frame in self.frames.values():
      frame[:] = 0

    self.rp = KernelReprojector(T, dst_size, device)
    self.rp.bind(self.frames[self.WIDE_KEY], self.frames[self.ROAD_KEY])
    self.meter = SeamMeter(T["meter"])

    dummies = [Tensor.zeros(self.src_buf_size, dtype='uint8', device=device).contiguous().realize() for _ in range(2)]
    for _ in range(3):
      self.rp(dummies[0], dummies[1])
    self._dev.synchronize()

  def _src(self, buf):
    data = buf.data if hasattr(buf, 'data') else buf
    ptr = np.frombuffer(data, dtype=np.uint8).ctypes.data
    if ptr not in self._src_cache:
      self._src_cache[ptr] = self._tensor.from_blob(ptr, (self.src_buf_size,), dtype='uint8', device=self.device)
    return self._src_cache[ptr]

  def process(self, bufs: dict, gains: tuple[float, float] = (1.0, 1.0)) -> dict[str, np.ndarray]:
    wide = np.frombuffer(bufs[self.WIDE_KEY].data if hasattr(bufs[self.WIDE_KEY], 'data') else bufs[self.WIDE_KEY],
                         dtype=np.uint8, count=self.src_buf_size)
    narrow = np.frombuffer(bufs[self.ROAD_KEY].data if hasattr(bufs[self.ROAD_KEY], 'data') else bufs[self.ROAD_KEY],
                           dtype=np.uint8, count=self.src_buf_size)
    model_gain = float(gains[0]) if len(gains) else 1.0
    match = self.meter.update(wide, narrow, model_gain)
    self.rp(self._src(bufs[self.WIDE_KEY]), self._src(bufs[self.ROAD_KEY]), match)
    self._dev.synchronize()
    if self._invalidate is not None:
      for frame in self.frames.values():
        addr = frame.ctypes.data
        self._invalidate.fxn(ctypes.c_uint64(addr & ~63), -(-(addr + self.copy_size - (addr & ~63)) // 64))
    return self.frames
