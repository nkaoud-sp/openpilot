#!/usr/bin/env python3
"""The 3X's cameras as a comma 4's. Reprojects camerad's narrow and wide frames into the comma 4 geometry on the QCOM GPU
(reproject_c4: the wide as the surround, the narrow as the sharp inset) and serves the pair as VisionIPC streams "reproject"
(narrow = the composite, wide), with camerad's frame ids and timestamps, so modeld runs comma's big model on them as it would
on a mici and the ui can show them (ReprojectionView). On request it also serves "modelinput": those frames warped into the
model's 512x256 inputs the way modeld does it. The seam exposure meter lives here, a rotation reprojectcalibd publishes as
fitted is swapped in here, reprojectState carries each frame's stage time and the applied rotation (HUD, calibrationd,
the log) and reprojectOutlines the debug view's geometry."""
import os
os.environ.setdefault('QCOM_PRIORITY', '1')  # KGSL context priority: the stage preempts the driver-monitoring model's 20 ms
os.environ.setdefault('DEV', 'QCOM')  # tinygrad's default device: never probe the eGPU, modeld owns it (its USB lock)
import threading
import time

import numpy as np
from tinygrad import Tensor, Device

from openpilot.cereal import messaging, custom
from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcClient, VisionIpcServer
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix, MEDMODEL_INPUT_SIZE
from openpilot.selfdrive.modeld import reproject_c4 as RC
from openpilot.selfdrive.modeld.reproject_c4.kernel import Reprojector
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

C4_CAM = RC.C4_CAM
NARROW, WIDE = VisionStreamType.VISION_STREAM_NARROW_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD


class Stage:
  def __init__(self, src_wh: tuple[int, int]):
    self.src_wh, self.cache_dir = src_wh, RC.table_cache_dir()
    self.src_size = get_nv12_info(*src_wh)[3]
    self.rotation = RC.load_rotation(); self.fitted = bool(RC.read_rotation())
    cloudlog.warning(f"reprojectd: rotation {np.degrees(self.rotation).round(3)} deg, "
                     + ('fitted' if self.fitted else 'seeded from CalibrationParams / the fleet median'))
    self.calib = RC.calib_from_rotvec(self.rotation)
    T = RC.load_tables(src_wh, C4_CAM, self.cache_dir, self.calib)
    self.rp = Reprojector(T, C4_CAM, 'QCOM'); self.meter = RC.SeamMeter(T["meter"])
    self.pending = None; self.loader: threading.Thread | None = None
    self._src_tensors: dict[int, Tensor] = {}
    self.out = np.zeros((2, self.rp.body), np.uint8)  # composite, wide: the kernel writes here, the server copies into its ring
    self.rp.bind(self.out[1], self.out[0])
    self.time = 0.0

  def src_tensor(self, buf) -> Tensor:
    ptr = np.frombuffer(buf.data, dtype=np.uint8).ctypes.data
    if ptr not in self._src_tensors:  # one mapping per VisionIPC ring slot
      self._src_tensors[ptr] = Tensor.from_blob(ptr, (self.src_size,), dtype='uint8', device='QCOM')
    return self._src_tensors[ptr]

  def run(self, wide, narrow, match: np.ndarray) -> None:
    t0 = time.perf_counter()
    self.rp(self.src_tensor(wide), self.src_tensor(narrow), match)
    Device['QCOM'].synchronize()
    self.time = time.perf_counter() - t0

  def follow_fit(self, fit) -> None:
    """A rotation reprojectcalibd has published as fitted (its tables already built) is loaded in a thread and swapped in
    between frames (one upload). calibrationd holds meanwhile, so the car cannot be engaged around the swap."""
    if self.pending is not None:
      T, calib, meter, rot = self.pending; self.pending = None; self.loader = None
      t0 = time.perf_counter()
      self.rp.reload(T); self.calib, self.meter, self.rotation, self.fitted = calib, meter, rot, True
      cloudlog.warning(f"reprojectd: fitted rotation {np.degrees(rot).round(3)} deg swapped in ({(time.perf_counter() - t0) * 1e3:.0f} ms)")
      return
    if self.loader is not None or fit is None or fit.status != custom.ReprojectFit.Status.fitted or len(fit.mean) != 3:
      return
    mean = tuple(float(v) for v in fit.mean)
    if np.allclose(mean, self.rotation, atol=1e-6):
      return
    # the tables are cached under the exact rotation the fitter saved; the message's float32 copy would name another file
    saved = RC.read_rotation().get("rotvec")
    if saved is None or not np.allclose(saved, mean, atol=1e-6):
      cloudlog.warning("reprojectd: fitted rotation not saved yet"); return
    rot = tuple(float(v) for v in saved)
    calib = RC.calib_from_rotvec(rot)
    def load():  # one 19 MB read off the frame loop (a build, if the fitter's tables are gone)
      os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))  # inherited SCHED_FIFO otherwise
      T = RC.load_tables(self.src_wh, C4_CAM, self.cache_dir, calib)
      self.pending = (T, calib, RC.SeamMeter(T["meter"]), rot)
    self.loader = threading.Thread(target=load, daemon=True); self.loader.start()


class ModelInputView:
  """What the driving model sees: the comma 4 frames warped into its 512x256 inputs with modeld's own warp matrices (the live
  calibration), bilinear like the model's warp, served as VisionIPC "modelinput" (narrow = the composite's, wide = the wide's)."""
  def __init__(self):
    self.wh = MEDMODEL_INPUT_SIZE
    self.stride, self.y_height, _, self.size = get_nv12_info(*self.wh)
    self.c4_stride, self.c4_y_height, _, _ = get_nv12_info(*C4_CAM)
    self.server = VisionIpcServer("modelinput")
    for tp in (NARROW, WIDE):
      self.server.create_buffers_with_sizes(tp, 4, self.wh[0], self.wh[1], self.size, self.stride, self.stride * self.y_height)
    self.server.start_listener()
    self.frame = np.zeros(self.size, np.uint8)
    self.maps: list[tuple] = []
    self.enabled = False

  def set_calibration(self, device_from_calib_euler: np.ndarray) -> None:
    dc = DEVICE_CAMERAS["mici", "os04c10"]
    self.maps = [self._sample_map(get_warp_matrix(device_from_calib_euler, dc.narrow_road.intrinsics, False)),
                 self._sample_map(get_warp_matrix(device_from_calib_euler, dc.wide_road.intrinsics, True))]

  def _sample_map(self, camera_from_model: np.ndarray) -> tuple:
    w, h = self.wh
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    p = camera_from_model @ np.stack([u.ravel(), v.ravel(), np.ones(u.size, np.float32)])
    x, y = p[0] / p[2], p[1] / p[2]
    x0 = np.clip(np.floor(x), 0, C4_CAM[0] - 2).astype(np.int32); y0 = np.clip(np.floor(y), 0, C4_CAM[1] - 2).astype(np.int32)
    fx = np.clip(x - x0, 0, 1).astype(np.float32); fy = np.clip(y - y0, 0, 1).astype(np.float32)
    xu = np.clip(np.rint(x / 2), 0, C4_CAM[0] // 2 - 1).astype(np.int32); yv = np.clip(np.rint(y / 2), 0, C4_CAM[1] // 2 - 1).astype(np.int32)
    i_uv = (self.c4_stride * self.c4_y_height + yv * self.c4_stride + 2 * xu).reshape(h, w)[::2, ::2]
    return y0 * self.c4_stride + x0, fx, fy, np.ascontiguousarray(i_uv)

  def publish(self, tp: VisionStreamType, src: np.ndarray, frame_id: int, sof: int, eof: int) -> None:
    i00, fx, fy, i_uv = self.maps[0 if tp == NARROW else 1]
    a, b = np.take(src, i00).astype(np.float32), np.take(src, i00 + 1).astype(np.float32)
    c, d = np.take(src, i00 + self.c4_stride).astype(np.float32), np.take(src, i00 + self.c4_stride + 1).astype(np.float32)
    top, bot = a + (b - a) * fx, c + (d - c) * fx
    w, h = self.wh
    self.frame[:self.stride * self.y_height].reshape(self.y_height, self.stride)[:h, :w] = (top + (bot - top) * fy + 0.5).reshape(h, w)
    uv = self.frame[self.stride * self.y_height:].reshape(-1, self.stride)
    uv[:h // 2, 0:w:2] = np.take(src, i_uv); uv[:h // 2, 1:w:2] = np.take(src, i_uv + 1)
    self.server.send(tp, self.frame, frame_id, sof, eof)


class OutlineWriter:
  """reprojectOutlines: the debug view's polygons (RC.debug_outlines) for the applied rotation and the model's calibration,
  recomputed in a plain-priority thread when either moves (~60 ms of numpy that must not sit in the real-time loop) and
  published from the loop."""
  def __init__(self, pm: messaging.PubMaster):
    self.pm = pm
    self.thread: threading.Thread | None = None
    self.written: tuple | None = None
    self.result = None

  def update(self, rotation, rpy: np.ndarray) -> None:
    if self.result is not None:
      self.pm.send('reprojectOutlines', self.result); self.result = None
    key = (tuple(np.round(rotation, 5)), tuple(np.round(rpy, 4)))
    if key == self.written or (self.thread is not None and self.thread.is_alive()):
      return
    self.written = key
    def run():
      os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
      try:
        items = [(frame, it) for frame, its in RC.debug_outlines(RC.calib_from_rotvec(rotation), rpy).items() for it in its]
        msg = messaging.new_message('reprojectOutlines', valid=True)
        for o, (frame, it) in zip(msg.reprojectOutlines.init('items', len(items)), items):
          o.frame = frame; o.name = it['name']; o.colour = it['colour']; o.visible = it['visible']
          o.points = [float(v) for xy in it['points'] for v in xy]
          o.inner = [float(v) for xy in it.get('inner', []) for v in xy]
        self.result = msg
      except Exception:
        cloudlog.exception("reprojectd: debug outlines failed")
    self.thread = threading.Thread(target=run, daemon=True); self.thread.start()


def main():
  sm = messaging.SubMaster(['narrowRoadCameraState', 'wideRoadCameraState', 'extrinsicsCalibration', 'reprojectFit'])
  params = Params(); pm = messaging.PubMaster(['reprojectState', 'reprojectOutlines'])
  narrow = VisionIpcClient("camerad", NARROW, True); wide = VisionIpcClient("camerad", WIDE, False)
  while not narrow.connect(False):
    time.sleep(0.1)
  while not wide.connect(False):
    time.sleep(0.1)
  src_wh = (narrow.width, narrow.height)
  cloudlog.warning(f"reprojectd: cameras {src_wh[0]}x{src_wh[1]} -> comma 4 {C4_CAM[0]}x{C4_CAM[1]}")
  t0 = time.monotonic()
  stage = Stage(src_wh)
  blank = [Tensor.zeros(stage.src_size, dtype='uint8', device='QCOM').contiguous().realize() for _ in range(2)]
  for _ in range(3):  # jit capture before the first real frame
    stage.rp(blank[0], blank[1]); Device['QCOM'].synchronize()
  stride, y_height, _, _ = get_nv12_info(*C4_CAM)
  server = VisionIpcServer("reproject")
  for tp in (NARROW, WIDE):
    server.create_buffers_with_sizes(tp, 4, C4_CAM[0], C4_CAM[1], stage.rp.body, stride, stride * y_height)
  server.start_listener()
  view = ModelInputView(); outlines = OutlineWriter(pm); rpy = np.zeros(3, np.float32)
  cloudlog.warning(f"reprojectd: serving after {time.monotonic() - t0:.1f} s")
  # real-time only from here: the table load and the jit capture above are seconds of CPU, and a real-time task holding a
  # core for a second trips the kernel's throttle, which this kernel turns into a panic (CONFIG_PANIC_ON_RT_THROTTLING)
  config_realtime_process(6, 53)

  n = 0
  while True:
    pair = RC.recv_pair(narrow, wide)
    if pair is None:
      continue
    buf_n, buf_w = pair
    sm.update(0)
    # match the wide surround to the narrow inset: measured in the seam ring of these frames, the sensors' exposure
    # settings as the fallback when the ring is unusable
    ncs, wcs = sm['narrowRoadCameraState'], sm['wideRoadCameraState']
    g = RC.exposure_gain(ncs.gain * ncs.integLines, wcs.gain * wcs.integLines) if sm.seen['narrowRoadCameraState'] and sm.seen['wideRoadCameraState'] else 1.0
    match = stage.meter.update(np.frombuffer(buf_w.data, dtype=np.uint8), np.frombuffer(buf_n.data, dtype=np.uint8), g)
    stage.run(buf_w, buf_n, match)
    server.send(NARROW, stage.out[0], narrow.frame_id, narrow.timestamp_sof, narrow.timestamp_eof)
    server.send(WIDE, stage.out[1], wide.frame_id, wide.timestamp_sof, wide.timestamp_eof)
    n += 1
    msg = messaging.new_message('reprojectState', valid=True); st = msg.reprojectState
    st.frameId = narrow.frame_id; st.stageMs = stage.time * 1e3; st.rotation = [float(v) for v in stage.rotation]
    st.fitted = stage.fitted
    pm.send('reprojectState', msg)
    if sm.updated['extrinsicsCalibration'] and len(sm['extrinsicsCalibration'].rpyCalib) == 3:
      rpy = np.array(sm['extrinsicsCalibration'].rpyCalib, dtype=np.float32)
      view.set_calibration(rpy)
    if view.enabled and view.maps:
      view.publish(NARROW, stage.out[0], narrow.frame_id, narrow.timestamp_sof, narrow.timestamp_eof)
      view.publish(WIDE, stage.out[1], wide.frame_id, wide.timestamp_sof, wide.timestamp_eof)
    if n % 20 == 0:
      stage.follow_fit(sm['reprojectFit'] if sm.seen['reprojectFit'] else None)
      view.enabled = params.get("ReprojectionView", return_default=True) == 1
      if params.get("ShowReprojectionDebug", return_default=True) in (1, 3):  # Visual / All
        outlines.update(stage.rotation, rpy)


if __name__ == "__main__":
  main()
