"""The gather tables that map every comma 4 output byte to a 3X source byte, built once per rotation and cached on disk."""
import contextlib
import glob
import hashlib
import json
import os

import numpy as np

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from .geometry import C4_NARROW_K, FEATHER_PX, _nv12_index, sample_coords
from .meter import SeamMeter


IDX_BITS, ALPHA_SHIFT, INVALID_BIT = 0x3fffff, 22, 1 << 30  # one int32 per output byte: wide index | alpha << 22 | invalid

def build_tables(src_wh, dst_wh, calib):
  """Flat gather tables over the destination NV12 body (comma 4 layout). Byte b of the wide output = src[pw[b]] (the fill
  value where invalid); of the narrow composite, alpha[b]*narrow[pn[b]] + (1-alpha[b])*gain*wide[pw[b]]. The kernel is
  memory-bound, so the wide index, the blend weight and the validity share one int32 (`pw`)."""
  sw, sh = src_wh; dw, dh = dst_wh
  s_stride, s_yh, s_uvh, _ = get_nv12_info(sw, sh); s_uv = s_stride * s_yh
  assert s_stride * (s_yh + s_uvh) <= IDX_BITS + 1
  d_stride, d_yh, d_uvh, _ = get_nv12_info(dw, dh); d_uv = d_stride * d_yh
  n_body = d_stride * (d_yh + d_uvh)
  sc = calib["narrow"]["f"] / C4_NARROW_K[0, 0]  # 3X narrow px per comma 4 narrow px
  out = {}; planes = {}
  for cam in ("wide", "narrow"):
    idx_w = np.zeros(n_body, np.int64); idx_n = np.zeros(n_body, np.int64)
    val_w = np.zeros(n_body, bool); val_n = np.zeros(n_body, bool); dist = np.zeros(n_body, np.float32)
    for chroma in (False, True):
      s = 0.5 if chroma else 1.0
      mw, mn = sample_coords(cam, dw // 2 if chroma else dw, dh // 2 if chroma else dh, s, calib)
      iw, vw = _nv12_index(mw, sw, sh, s_stride, s_uv, chroma)
      inn, vn = _nv12_index(mn, sw, sh, s_stride, s_uv, chroma)
      # distance to the narrow frame edge, in destination px: the composite feathers over FEATHER_PX of it
      d = np.minimum(np.minimum(mn[..., 0], sw * s - mn[..., 0]), np.minimum(mn[..., 1], sh * s - mn[..., 1])) / (s * sc)
      if cam == "narrow":
        planes[chroma] = (iw, vw, inn, vn, d)
      if chroma:
        rows = np.arange(dh // 2)[:, None]; cols = np.arange(dw // 2)[None, :]
        for plane in (0, 1):
          flat = d_uv + rows * d_stride + 2 * cols + plane
          idx_w[flat] = iw + plane; idx_n[flat] = inn + plane; val_w[flat] = vw; val_n[flat] = vn; dist[flat] = d
      else:
        flat = np.arange(dh)[:, None] * d_stride + np.arange(dw)[None, :]
        idx_w[flat] = iw; idx_n[flat] = inn; val_w[flat] = vw; val_n[flat] = vn; dist[flat] = d
    alpha = np.round(np.clip(dist / FEATHER_PX, 0, 1) * val_n * 255).astype(np.int64)
    out[cam] = dict(pw=(idx_w | (alpha << ALPHA_SHIFT) | (~val_w * INVALID_BIT)).astype(np.int32))
  out["narrow"]["pn"] = idx_n.astype(np.int32)
  out["meter"] = SeamMeter.geometry(dw, dh, planes[False], planes[True])  # ~1 s of numpy on the 3X: not for modeld's swap
  return out

TABLE_VERSION = 6  # bump when the lens numbers, the feather or the table layout change

def table_path(src_wh, dst_wh, cache_dir, calib):
  tag = hashlib.sha1(json.dumps(calib, sort_keys=True, default=float).encode()).hexdigest()[:10]
  return os.path.join(cache_dir, f"reproject_c4_v{TABLE_VERSION}_{src_wh[0]}x{src_wh[1]}_{dst_wh[0]}x{dst_wh[1]}_{tag}.npz")

def load_tables(src_wh, dst_wh, cache_dir, calib):
  """build_tables takes ~25 s of numpy on the device CPU, so keep a copy on disk (19 MB, uncompressed: zlib was half the build on the 3X)."""
  os.makedirs(cache_dir, exist_ok=True)
  p = table_path(src_wh, dst_wh, cache_dir, calib)
  if os.path.exists(p):
    z = np.load(p); T = {"wide": {}, "narrow": {}, "meter": {}}
    for key in z.files:
      cam, k = key.split("_", 1); T[cam][k] = z[key]
    return T
  T = build_tables(src_wh, dst_wh, calib)
  np.savez(p + ".tmp.npz", **{f"{cam}_{k}": v for cam, tab in T.items() for k, v in tab.items()})
  os.replace(p + ".tmp.npz", p)
  for f in sorted(glob.glob(os.path.join(cache_dir, "reproject_c4_*.npz")), key=os.path.getmtime)[:-4]:  # every refit leaves a file
    with contextlib.suppress(OSError):
      os.remove(f)
  return T

def table_cache_dir() -> str:
  """Where the gather tables live: the device's tinygrad cache, a user cache on a PC."""
  from openpilot.common.hardware import PC
  return os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache/openpilot') if PC else '/data/tgcache')
