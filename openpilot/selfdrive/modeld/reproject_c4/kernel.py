"""The GPU stage: one gather per output byte through the tables, the composite blended in the seam band."""

from tinygrad import Tensor, TinyJit
import numpy as np

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from .geometry import UV_FILL
from .meter import SeamMeter
from .tables import ALPHA_SHIFT, IDX_BITS, INVALID_BIT


class Reprojector:
  """Holds the tables `T` (load_tables) on `device` and runs the gathers. Call once per model run with the two 3X NV12 buffers."""

  def __init__(self, T, dst_wh, device):
    self.stride, yh, uvh, _ = get_nv12_info(*dst_wh); self.dw, self.dh = dst_wh
    self.uv_offset, self.body = self.stride * yh, self.stride * (yh + uvh)
    assert self.body == 3 * (self.body - self.uv_offset)  # NV12: the UV plane is the last third of the body
    self.device = device
    self.reload(T)
    # per-frame match parameters as SeamMeter.update() returns them; on QCOM the jit reads them straight from host memory
    # (a per-call upload costs a scheduled copy, ~4 ms of Python)
    self.params_np = SeamMeter.IDENTITY.copy()
    if str(device or "").startswith("QCOM"):
      self.gains = Tensor.from_blob(self.params_np.ctypes.data, self.params_np.shape, dtype="float32", device=device); self.host_params = True
    else:
      self.gains = Tensor(self.params_np, device=device).contiguous().realize(); self.host_params = False
    self.plane = Tensor([0, 0, 1], dtype="uint8", device=device).realize()
    # pixel position for the gain gradient from the byte index (x = i % stride, y = i // stride): a plain sequential read;
    # broadcast row/column vectors cost 4 ms on the Adreno, per-pixel gathers 1.5 ms
    self.idx = Tensor.arange(self.body, dtype="int32").to(device).realize()
    self.dst = None
    self._run = TinyJit(self._both)

  def bind(self, host_wide: np.ndarray, host_narrow: np.ndarray):
    """Write the outputs straight into these host arrays (each `body` bytes, e.g. modeld's packed frame slots) instead of
    returning device tensors: a GPU->host readback costs ~40 ms on the 3X, a direct write ~10 ms."""
    assert host_wide.nbytes == self.body and host_narrow.nbytes == self.body and host_wide.dtype == host_narrow.dtype == np.uint8
    self.dst = (Tensor.from_blob(host_wide.ctypes.data, (self.body,), dtype="uint8", device=self.device),
                Tensor.from_blob(host_narrow.ctypes.data, (self.body,), dtype="uint8", device=self.device))
    self._run = TinyJit(self._both)

  def reload(self, T):
    """Swap in the tables of another rotation between runs: they are jit inputs, so one upload and no re-capture (~1.3 s on the 3X)."""
    assert len(T["wide"]["pw"]) == self.body
    self.t = {cam: {k: Tensor(v, device=self.device).realize() for k, v in T[cam].items()} for cam in ("wide", "narrow")}

  def _chroma(self):
    return self.plane.reshape(3, 1).expand(3, self.body // 3).reshape(self.body).bool()  # a broadcast: no table read

  def _wide(self, wide, pw):
    return (pw < INVALID_BIT).where(wide[pw & IDX_BITS], self._chroma().cast("uint8") * UV_FILL)

  def _narrow(self, wide, narrow, gains, pw, pn):
    chroma = self._chroma()
    # match the wide surround to the narrow: luma times a gain per brightness band (the two ISPs' tone curves; piecewise
    # linear between the band centres, flat beyond) times a gain gradient over the frame (lens shading); chroma gain about
    # the 128 midpoint plus a U/V offset (independent white balance), the offset a parity broadcast: U and V bytes alternate
    off = gains[1:3].reshape(1, 2).expand(self.body // 2, 2).reshape(self.body)
    w = wide[pw & IDX_BITS].float()
    c = [0.5 * (lo + hi) for lo, hi in SeamMeter.BANDS]
    tone = gains[5]
    for i in range(1, len(c)):
      tone = tone + (gains[5 + i] - gains[4 + i]) * ((w - c[i - 1]) * (1 / (c[i] - c[i - 1]))).clip(0, 1)
    x = (self.idx % self.stride).float() * (2.0 / self.dw) - 1; y = (self.idx // self.stride).float() * (2.0 / self.dh) - 1
    w = chroma.where((w - UV_FILL) * gains[0] + UV_FILL + off, w * tone * (1 + gains[3] * x + gains[4] * y)).clip(0, 255)
    w = (pw < INVALID_BIT).where(w, chroma.cast("float32") * UV_FILL)
    a = ((pw >> ALPHA_SHIFT) & 0xff).float() * (1 / 255)
    return a * narrow[pn & IDX_BITS].float() + (1 - a) * w  # the mask looks redundant but bounds the index; without it every gather is guarded (0.8 ms on the 3X)

  def _both(self, wide, narrow, gains, pw_w, pw_n, pn):
    out_w = self._wide(wide, pw_w)
    out_n = self._narrow(wide, narrow, gains, pw_n, pn).round().cast("uint8")
    if self.dst is not None:
      out_w, out_n = self.dst[0].assign(out_w), self.dst[1].assign(out_n)
    return out_w.realize(), out_n.realize()

  def __call__(self, wide, narrow, match=None):
    """wide/narrow: flat uint8 3X NV12 tensors (the same buffers every call, e.g. the VisionIPC ring, so the jit replays).
    match: the seam match vector from SeamMeter.update() (None = no match). Returns (c4_wide, c4_narrow): flat uint8 NV12
    tensors of `body` bytes; with bind() they are views of the bound host arrays."""
    self.params_np[:] = SeamMeter.IDENTITY if match is None else match
    if not self.host_params:
      self.gains.assign(Tensor(self.params_np, device=self.device)).realize()
    return self._run(wide, narrow, self.gains, self.t["wide"]["pw"], self.t["narrow"]["pw"], self.t["narrow"]["pn"])
