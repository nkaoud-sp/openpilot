"""Reproject the comma 3X cameras into comma 4 camera geometry, on the GPU, in front of an untouched comma 4 model.

Ported from Amy Jeanes' commaai/openpilot commit 4c525a70 ("modeld: reproject comma 3X cameras into comma 4
geometry on the QCOM before the big model"). The geometry, the measured lens constants and the packed gather
tables are hers, unchanged; only the runtime class is adapted to this tree's frame stage protocol. This is the
"C4 Ray Matching" mode of the chestnut frame stage; see frame_downscale.py for the cheaper "C4 Resampling".

Every comma 4 output pixel is traced through the comma 4 lens to a ray, rotated onto the 3X device axis and looked up in
the 3X wide (fisheye) or, for the narrow output, composited from the 3X narrow inset and the 3X wide surround. None of it
depends on the calibration, so the lookup tables are constants built once per process and the per-frame work is one
gather per output byte (two plus a blend for the composite) - the same cost as the model's own crop warp.

Outputs are NV12 buffers in the comma 4 camerad layout so the model pkl (keyed by its camera size) sees exactly what a
comma 4 would have given it. Lens numbers come from the bench + multi-device work in ~/tizi-to-mici (see the memory file).
"""
# The geometry below is kept byte-for-byte as upstream so it can be diffed against 4c525a70; its dense style
# (semicolon statements, dict(), lambdas) is not this repo's, so those rules are off for this file only.
# ruff: noqa: E702, C408, E731
import ctypes
import os

import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

# measured lenses (board-calibrated 3X, self-calibrated comma 4 population); f/cx/cy in px, angles in rad
X3_WIDE = dict(f=596.669, cx=957.566, cy=581.537, k=(-0.014942, -0.0023814, -0.00064424), tc=1.51354)  # fisheye r=f·θ(1+k1θ²+k2θ⁴+k3θ⁶), linear past tc
X3_NARROW = dict(f=2600.85, cx=964.0, cy=604.0, k1=-0.36400)  # pinhole with radial barrel: r=f·x(1+k1·x²)
C4_WIDE = dict(f=442.555, cx=672.380, cy=378.718, k=(0.0089611, 0.029156, -0.015066), tc=1.39626)
R_NARROW_FROM_WIDE = (-0.0167946, 0.0022473, -0.0011422)  # rotvec: the 3X wide points ~1° above the narrow
ZMIN = np.cos(np.radians(88.0))  # rays further off-axis than this have no 3X wide pixel
FEATHER_PX = 24  # composite seam width in comma 4 narrow px
UV_FILL = 128
C4_CAM = (1344, 760)  # comma 4 camerad frame size, what the big model pkl is keyed by


def _theta_d(th, L):
  k, tc = L["k"], L["tc"]
  p = lambda t: t * (1 + k[0] * t**2 + k[1] * t**4 + k[2] * t**6)
  dp = lambda t: 1 + 3 * k[0] * t**2 + 5 * k[1] * t**4 + 7 * k[2] * t**6
  return np.where(th <= tc, p(th), p(tc) + dp(tc) * (th - tc))


def _dtheta_d(th, L):
  k = L["k"]; t = np.minimum(th, L["tc"])
  return 1 + 3 * k[0] * t**2 + 5 * k[1] * t**4 + 7 * k[2] * t**6


def unproject_fisheye(px, L):
  dx, dy = px[..., 0] - L["cx"], px[..., 1] - L["cy"]
  r = np.hypot(dx, dy); td = r / L["f"]; th = td.copy()
  for _ in range(10):
    th = th - (_theta_d(th, L) - td) / _dtheta_d(th, L)
  s = np.sin(th); rr = np.where(r > 0, r, 1)
  return np.stack([s * dx / rr, s * dy / rr, np.cos(th)], -1)


def unproject_pinhole(px, f, cx, cy):
  d = np.stack([(px[..., 0] - cx) / f, (px[..., 1] - cy) / f, np.ones(px.shape[:-1])], -1)
  return d / np.linalg.norm(d, axis=-1, keepdims=True)


def project_fisheye(rays, L):
  rho = np.hypot(rays[..., 0], rays[..., 1]); th = np.arctan2(rho, rays[..., 2])
  r = L["f"] * _theta_d(th, L); rr = np.where(rho > 0, rho, 1)
  return np.stack([L["cx"] + r * rays[..., 0] / rr, L["cy"] + r * rays[..., 1] / rr], -1)


def project_pinhole_k1(rays, L):
  x, y = rays[..., 0] / rays[..., 2], rays[..., 1] / rays[..., 2]
  s = 1 + L["k1"] * (x * x + y * y)
  return np.stack([L["cx"] + L["f"] * x * s, L["cy"] + L["f"] * y * s], -1)


def rotvec_to_matrix(v):
  v = np.asarray(v, dtype=np.float64); a = np.linalg.norm(v)
  if a == 0:
    return np.eye(3)
  k = v / a; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
  return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def sample_coords(out_cam, dst_w, dst_h, scale=1.0):
  """Float 3X wide / narrow coordinates for every comma 4 `out_cam` pixel (scale 0.5 = the half-res chroma plane)."""
  xs, ys = np.meshgrid((np.arange(dst_w) + 0.5) / scale, (np.arange(dst_h) + 0.5) / scale)
  px = np.stack([xs, ys], -1)
  if out_cam == "wide":
    rays = unproject_fisheye(px, C4_WIDE)
  else:
    Kn = DEVICE_CAMERAS[("mici", "os04c10")].narrow_road.intrinsics
    rays = unproject_pinhole(px, Kn[0, 0], Kn[0, 2], Kn[1, 2])
  rays_w = rays @ rotvec_to_matrix(R_NARROW_FROM_WIDE).T
  mw = project_fisheye(rays_w, X3_WIDE) * scale
  mw[rays_w[..., 2] < ZMIN] = -1
  mn = project_pinhole_k1(rays, X3_NARROW) * scale
  mn[rays[..., 2] <= 0] = -1
  return mw, mn


def _nv12_index(xy, src_w, src_h, stride, uv_offset, chroma):
  """Nearest-neighbour byte index into a source NV12 buffer for float sample coords; chroma coords are in the half-res plane.
  Returns (index, valid). For chroma the caller adds +1 for the V byte."""
  xy = np.clip(np.nan_to_num(xy, nan=-1.0), -1e6, 1e6)  # grazing rays project to ±inf
  x = np.round(xy[..., 0] - 0.5).astype(np.int64); y = np.round(xy[..., 1] - 0.5).astype(np.int64)
  w, h = (src_w // 2, src_h // 2) if chroma else (src_w, src_h)
  valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
  x = np.clip(x, 0, w - 1); y = np.clip(y, 0, h - 1)
  idx = (uv_offset + y * stride + 2 * x) if chroma else (y * stride + x)
  return idx, valid


IDX_BITS, ALPHA_SHIFT, INVALID_BIT = 0x3fffff, 22, 1 << 30  # one int32 per output byte: wide index | alpha << 22 | invalid


def build_tables(src_wh, dst_wh):
  """Flat gather tables over the whole destination NV12 buffer (comma 4 layout) for the wide output and the narrow
  composite. Byte b of the output = src[idx[b]] (or the fill value where invalid); for the composite,
  alpha[b]*narrow[pn[b]] + (1-alpha[b])*gain*wide[idx[b]]. The kernel is memory-bound, so the wide index, the blend
  weight and the validity share one int32 (`pw`); only the composite needs a second table (`pn`)."""
  sw, sh = src_wh; dw, dh = dst_wh
  s_stride, s_yh, s_uvh, _ = get_nv12_info(sw, sh); s_uv = s_stride * s_yh
  assert s_stride * (s_yh + s_uvh) <= IDX_BITS + 1
  d_stride, d_yh, d_uvh, d_size = get_nv12_info(dw, dh); d_uv = d_stride * d_yh
  n_body = d_stride * (d_yh + d_uvh)
  out = {}
  for cam in ("wide", "narrow"):
    idx_w = np.zeros(n_body, np.int64); idx_n = np.zeros(n_body, np.int64)
    val_w = np.zeros(n_body, bool); val_n = np.zeros(n_body, bool); dist = np.zeros(n_body, np.float32)
    for chroma in (False, True):
      mw, mn = sample_coords(cam, dw // 2 if chroma else dw, dh // 2 if chroma else dh, 0.5 if chroma else 1.0)
      iw, vw = _nv12_index(mw, sw, sh, s_stride, s_uv, chroma)
      inn, vn = _nv12_index(mn, sw, sh, s_stride, s_uv, chroma)
      # distance to the narrow frame edge, in destination px: the composite feathers over FEATHER_PX of it
      sc = (0.5 if chroma else 1.0) * (X3_NARROW["f"] / DEVICE_CAMERAS[("mici", "os04c10")].narrow_road.intrinsics[0, 0])
      d = np.minimum(np.minimum(mn[..., 0], sw * (0.5 if chroma else 1) - mn[..., 0]),
                     np.minimum(mn[..., 1], sh * (0.5 if chroma else 1) - mn[..., 1])) / sc
      if chroma:
        rows = np.arange(dh // 2)[:, None]; cols = np.arange(dw // 2)[None, :]
        for plane in (0, 1):
          flat = d_uv + rows * d_stride + 2 * cols + plane
          idx_w[flat] = iw + plane; idx_n[flat] = inn + plane; val_w[flat] = vw; val_n[flat] = vn; dist[flat] = d
      else:
        flat = np.arange(dh)[:, None] * d_stride + np.arange(dw)[None, :]
        idx_w[flat] = iw; idx_n[flat] = inn; val_w[flat] = vw; val_n[flat] = vn; dist[flat] = d
    alpha = np.round(np.clip(dist / FEATHER_PX, 0, 1) * val_n * 255).astype(np.int64)
    pw = idx_w | (alpha << ALPHA_SHIFT) | (~val_w * INVALID_BIT)
    out[cam] = dict(pw=pw.astype(np.int32), size=d_size, body=n_body, uv_offset=d_uv)
    if cam == "narrow":
      out[cam]["pn"] = idx_n.astype(np.int32)
  return out


TABLE_VERSION = 2  # bump when the lens numbers or the table layout change


def load_tables(src_wh, dst_wh, cache_dir=None):
  """build_tables takes ~25 s of numpy on the device CPU, so keep a copy on disk (a few MB, compressed)."""
  if cache_dir is None:
    return build_tables(src_wh, dst_wh)
  os.makedirs(cache_dir, exist_ok=True)
  p = os.path.join(cache_dir, f"reproject_c4_v{TABLE_VERSION}_{src_wh[0]}x{src_wh[1]}_{dst_wh[0]}x{dst_wh[1]}.npz")
  if os.path.exists(p):
    z = np.load(p); T = {"wide": {}, "narrow": {}}
    for key in z.files:
      cam, k = key.split("_", 1); T[cam][k] = z[key] if z[key].ndim else int(z[key])
    return T
  T = build_tables(src_wh, dst_wh)
  np.savez_compressed(p + ".tmp.npz", **{f"{cam}_{k}": v for cam, tab in T.items() for k, v in tab.items()})
  os.replace(p + ".tmp.npz", p)
  return T




class Reprojector:
  """Chestnut frame stage: 3X wide+narrow NV12 in, comma 4 wide+narrow NV12 out, in persistent host frames.

  Same output contract as FrameDownscaler in frame_downscale.py, so modeld can hold either: process() returns one
  host array per model input, valid until the next call. Unlike the resampler this synthesises comma 4 camera
  geometry, so the warp matrices are built from comma 4 intrinsics rather than scaled (c4_intrinsics below).
  """

  ROAD_KEY, WIDE_KEY = 'img', 'big_img'
  scale = np.eye(3, dtype=np.float32)  # the frame is comma 4 geometry, so the warp changes intrinsics, not scale
  c4_intrinsics = True

  def __init__(self, src_size=(1928, 1208), dst_size=C4_CAM, device='QCOM', cache_dir=None):
    from tinygrad.device import Device
    from tinygrad.engine.jit import TinyJit
    from tinygrad.tensor import Tensor
    from openpilot.selfdrive.modeld.frame_downscale import dcache_invalidate
    self._tensor, self._dev = Tensor, Device[device]
    self.src_size, self.dst_size, self.device = src_size, dst_size, device
    self.src_buf_size = get_nv12_info(*src_size)[3]
    self.dst_buf_size = get_nv12_info(*dst_size)[3]

    T = load_tables(src_size, dst_size, cache_dir)
    self.copy_size, self.uv_offset = T["wide"]["body"], T["wide"]["uv_offset"]
    self.t = {cam: {k: Tensor(v, device=device).realize() for k, v in tab.items() if isinstance(v, np.ndarray)}
              for cam, tab in T.items()}
    assert self.copy_size == 3 * (self.copy_size - self.uv_offset)  # NV12: the UV plane is the last third of the body
    self.plane = Tensor([0, 0, 1], dtype='uint8', device=device).realize()

    # the jit reads the exposure gains straight out of host memory: uploading them per call costs a scheduled copy
    self.gains_np = np.ones(2, np.float32)
    self.gains = Tensor.from_blob(self.gains_np.ctypes.data, (2,), dtype='float32', device=device)

    self._invalidate = dcache_invalidate() if device.startswith('QCOM') else None
    self._src_cache: dict[int, object] = {}
    # persistent host frames the kernel writes into, so a pointer taken to one stays valid across runs; see the
    # note in frame_downscale.py on why the GPU writes to cached host memory instead of us reading it back
    self.frames = {k: np.zeros(self.dst_buf_size, dtype=np.uint8) for k in (self.ROAD_KEY, self.WIDE_KEY)}
    for frame in self.frames.values():
      frame[:] = 0  # fault the pages in before mapping, so the map-time cache clean covers them
    self.dst = {k: Tensor.from_blob(v.ctypes.data, (self.copy_size,), dtype='uint8', device=device)
                for k, v in self.frames.items()}
    self._run = TinyJit(self._both)
    # capture the jit on dummy frames so the first camera frame replays it; the two must be distinct buffers
    dummies = [Tensor.zeros(self.src_buf_size, dtype='uint8', device=device).contiguous().realize() for _ in range(2)]
    for _ in range(3):
      self._run(*dummies, self.gains)
    self._dev.synchronize()

  def _src(self, buf):
    data = buf.data if hasattr(buf, 'data') else buf
    ptr = np.frombuffer(data, dtype=np.uint8).ctypes.data
    if ptr not in self._src_cache:  # one mapping per VisionIPC ring slot
      self._src_cache[ptr] = self._tensor.from_blob(ptr, (self.src_buf_size,), dtype='uint8', device=self.device)
    return self._src_cache[ptr]

  def _chroma(self):
    return self.plane.reshape(3, 1).expand(3, self.copy_size // 3).reshape(self.copy_size).bool()  # a broadcast: no table read

  def _wide(self, wide):
    pw = self.t["wide"]["pw"]
    return (pw < INVALID_BIT).where(wide[pw & IDX_BITS], self._chroma().cast('uint8') * UV_FILL)

  def _narrow(self, wide, narrow, gain_y, gain_c):
    t = self.t["narrow"]; pw = t["pw"]; chroma = self._chroma()
    # exposure-match the wide surround to the narrow inset: luma gain on Y, chroma gain about the 128 midpoint
    w = wide[pw & IDX_BITS].float()
    w = chroma.where((w - UV_FILL) * gain_c + UV_FILL, w * gain_y).clip(0, 255)
    w = (pw < INVALID_BIT).where(w, chroma.cast('float32') * UV_FILL)
    a = ((pw >> ALPHA_SHIFT) & 0xff).float() * (1 / 255)
    return a * narrow[t["pn"]].float() + (1 - a) * w

  def _both(self, wide, narrow, gains):
    out_w = self.dst[self.WIDE_KEY].assign(self._wide(wide))
    out_n = self.dst[self.ROAD_KEY].assign(self._narrow(wide, narrow, gains[0], gains[1]).round().cast('uint8'))
    return out_w.realize(), out_n.realize()

  def process(self, bufs: dict, gains: tuple[float, float] = (1.0, 1.0)) -> dict[str, np.ndarray]:
    self.gains_np[:] = gains
    self._run(self._src(bufs[self.WIDE_KEY]), self._src(bufs[self.ROAD_KEY]), self.gains)
    self._dev.synchronize()
    if self._invalidate is not None:
      for frame in self.frames.values():
        addr = frame.ctypes.data
        self._invalidate.fxn(ctypes.c_uint64(addr & ~63), -(-(addr + self.copy_size - (addr & ~63)) // 64))
    return self.frames
