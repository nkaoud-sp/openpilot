#!/usr/bin/env python3
"""Replay a route with the reprojection views: comma's replay, the reprojection stage and the onroad ui in one prefix, the
ui's params seeded from the route's initData (so the road view and the debug view come up as they were on the drive) with
optional overrides. The route must carry reprojectState / reprojectFit (recorded on c0b234ce0 or later).
  openpilot/tools/reproject_c4/replay.py <route> [-d data_dir] [-s start_s] [--view default|modelinput|reprojected]
      [--debug none|visual|stats|all] [--scale 0.5] [--record out.mp4]
replay keeps the terminal (space pauses, arrows seek, q quits); the stage and the ui follow it and stop with it.

The stage (--stage-only, for a replay and ui started by hand) is what reprojectd and reprojectcalibd put on screen, rebuilt
on this PC from what they logged. It follows replay's camera streams and the logged reprojectState / reprojectFit /
extrinsicsCalibration: reprojects each frame pair into the comma 4 geometry at the logged rotation (served as "reproject",
plus "modelinput" for the Model Input road view) and publishes the debug outlines (reprojectOutlines is not logged).
Everything else comes from replay's republished log."""
import argparse
import glob
import os
os.environ.setdefault('DEV', 'CUDA')  # before reprojectd's import defaults it to QCOM
import subprocess
import sys
import threading
import time
from collections import deque

import numpy as np
from tinygrad import Tensor

from openpilot.cereal import messaging, custom
from msgq.visionipc import VisionIpcClient, VisionIpcServer
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.modeld import reproject_c4 as RC
from openpilot.selfdrive.modeld.reproject_c4.kernel import Reprojector
from openpilot.selfdrive.modeld.reprojectd import ModelInputView, OutlineWriter, NARROW, WIDE
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

DEV = os.environ['DEV']
C4_CAM = RC.C4_CAM; CACHE_DIR = RC.table_cache_dir()
FitStatus = custom.ReprojectFit.Status
VIEWS = {'default': 0, 'modelinput': 1, 'reprojected': 2}
DEBUG = {'none': 0, 'visual': 1, 'stats': 2, 'all': 3}
CAMERAS = {'default': 0, 'narrow': 1, 'wide': 2}  # ReprojectionCamera


class Stage:
  """The kernel on this PC's GPU at a rotation. Tables come from the cache or a build (~10 s of numpy once per rotation);
  a rotation the log says is coming (reprojectFit fitted, not yet applied) is built ahead in a thread."""
  def __init__(self, src_wh, rotation):
    self.src_wh = src_wh; self.src_size = get_nv12_info(*src_wh)[3]
    self.rotation = tuple(float(v) for v in rotation); self.calib = RC.calib_from_rotvec(self.rotation)
    T = RC.load_tables(src_wh, C4_CAM, CACHE_DIR, self.calib)
    self.rp = Reprojector(T, C4_CAM, DEV); self.meter = RC.SeamMeter(T["meter"])
    self.time = 0.0; self.ahead: dict[tuple, threading.Thread] = {}

  def run(self, wide: np.ndarray, narrow: np.ndarray, match: np.ndarray):
    t0 = time.perf_counter()
    out_w, out_n = self.rp(Tensor(wide, device=DEV).realize(), Tensor(narrow, device=DEV).realize(), match)
    comp, wide_out = out_n.numpy(), out_w.numpy()
    self.time = time.perf_counter() - t0
    return comp, wide_out

  def build_ahead(self, rotation) -> None:
    key = tuple(round(float(v), 7) for v in rotation)
    if key == tuple(round(v, 7) for v in self.rotation) or key in self.ahead:
      return
    t = threading.Thread(target=RC.load_tables, args=(self.src_wh, C4_CAM, CACHE_DIR, RC.calib_from_rotvec(key)), daemon=True)
    self.ahead[key] = t; t.start()

  def follow(self, rotation) -> None:
    """Swap to the rotation the log says is applied now."""
    rot = tuple(float(v) for v in rotation)
    if np.allclose(rot, self.rotation, atol=1e-7):
      return
    t = self.ahead.pop(tuple(round(v, 7) for v in rot), None)
    if t is not None:
      t.join()
    t0 = time.perf_counter()
    self.calib = RC.calib_from_rotvec(rot); T = RC.load_tables(self.src_wh, C4_CAM, CACHE_DIR, self.calib)
    self.rp.reload(T); self.meter = RC.SeamMeter(T["meter"]); self.rotation = rot
    print(f"replay stage: rotation {np.degrees(rot).round(3)} deg swapped in ({time.perf_counter() - t0:.1f} s)")


def serve():
  sm = messaging.SubMaster(['reprojectState', 'reprojectFit', 'extrinsicsCalibration', 'narrowRoadCameraState', 'wideRoadCameraState'])
  params = Params(); pm = messaging.PubMaster(['reprojectOutlines'])
  narrow = VisionIpcClient("camerad", NARROW, True); wide = VisionIpcClient("camerad", WIDE, False)
  print("replay stage: waiting for replay's cameras (narrow and wide: replay --wide-road)")
  while not narrow.connect(False):
    time.sleep(0.1)
  while not wide.connect(False):
    time.sleep(0.1)
  src_wh = (narrow.width, narrow.height); src_size = get_nv12_info(*src_wh)[3]
  print(f"replay stage: cameras {src_wh[0]}x{src_wh[1]}; waiting for reprojectState (the route must carry it)")
  while True:
    sm.update(100)
    if sm.seen['reprojectState'] and len(sm['reprojectState'].rotation) == 3:
      break
  stage = Stage(src_wh, sm['reprojectState'].rotation)
  print(f"replay stage: rotation {np.degrees(stage.rotation).round(3)} deg, serving on {DEV}")
  stride, y_height, _, _ = get_nv12_info(*C4_CAM)
  server = VisionIpcServer("reproject")
  for tp in (NARROW, WIDE):
    server.create_buffers_with_sizes(tp, 4, C4_CAM[0], C4_CAM[1], stage.rp.body, stride, stride * y_height)
  server.start_listener()
  view = ModelInputView(); outlines = OutlineWriter(pm); rpy = np.zeros(3, np.float32)
  visual = False
  n = 0; g = 1.0; times: deque[float] = deque(maxlen=200)
  while True:
    pair = RC.recv_pair(narrow, wide)
    if pair is None:
      continue
    buf_n, buf_w = pair
    sm.update(0)
    st = sm['reprojectState']
    if sm.updated['reprojectState']:
      stage.follow(st.rotation)
    fit = sm['reprojectFit'] if sm.seen['reprojectFit'] else None
    if fit is not None and fit.status in (FitStatus.building, FitStatus.fitted) and len(fit.mean) == 3:
      stage.build_ahead(fit.mean)
    ncs, wcs = sm['narrowRoadCameraState'], sm['wideRoadCameraState']
    g = RC.exposure_gain(ncs.gain * ncs.integLines, wcs.gain * wcs.integLines) if sm.seen['narrowRoadCameraState'] and sm.seen['wideRoadCameraState'] else 1.0
    wide_np = np.frombuffer(buf_w.data, dtype=np.uint8, count=src_size); narrow_np = np.frombuffer(buf_n.data, dtype=np.uint8, count=src_size)
    match = stage.meter.update(wide_np, narrow_np, g)
    comp, wide_out = stage.run(wide_np, narrow_np, match)
    server.send(NARROW, comp, narrow.frame_id, narrow.timestamp_sof, narrow.timestamp_eof)
    server.send(WIDE, wide_out, wide.frame_id, wide.timestamp_sof, wide.timestamp_eof)
    times.append(stage.time); n += 1
    if sm.updated['extrinsicsCalibration'] and len(sm['extrinsicsCalibration'].rpyCalib) == 3:
      rpy = np.array(sm['extrinsicsCalibration'].rpyCalib, dtype=np.float32)
      view.set_calibration(rpy)
    if view.enabled and view.maps:
      view.publish(NARROW, comp, narrow.frame_id, narrow.timestamp_sof, narrow.timestamp_eof)
      view.publish(WIDE, wide_out, wide.frame_id, wide.timestamp_sof, wide.timestamp_eof)
    if n % 20 == 0:
      view.enabled = params.get("ReprojectionView", return_default=True) == 1
      visual = params.get("ShowReprojectionDebug", return_default=True) in (1, 3)
      if visual:
        outlines.update(stage.rotation, rpy)
    if n % 200 == 0:
      print(f"replay: frame {narrow.frame_id}: stage median/p95 {np.median(times) * 1e3:.1f}/{np.percentile(times, 95) * 1e3:.1f} ms")


def seed_params(route: str, data_dir: str | None, view: str | None, debug: str | None, camera: str | None) -> None:
  from openpilot.tools.lib.logreader import LogReader
  if data_dir:  # a local route may have no dongle id, which Route() insists on
    name = route.rsplit('/', 1)[-1].rsplit('|', 1)[-1]
    path = sorted(glob.glob(os.path.join(data_dir, f"{name}--*", "rlog.zst")), key=lambda p: int(p.split('--')[-1].split('/')[0]))[0]
  else:
    from openpilot.tools.lib.route import Route
    path = next(p for p in Route(route).log_paths() if p)
  init = next(m.initData for m in LogReader(path) if m.which() == 'initData')
  params = Params()
  for e in init.params.entries:
    try:
      v = params.cpp2python(e.key, e.value)
      if v is not None:
        params.put(e.key, v)
    except (UnknownKeyName, TypeError):
      pass
  if view is not None:
    params.put("ReprojectionView", VIEWS[view])
  if debug is not None:
    params.put("ShowReprojectionDebug", DEBUG[debug])
  if camera is not None:
    params.put("ReprojectionCamera", CAMERAS[camera])
  print(f"replay: params from {os.path.basename(os.path.dirname(path))}; view {params.get('ReprojectionView', return_default=True)}, "
        + f"debug {params.get('ShowReprojectionDebug', return_default=True)}, camera {params.get('ReprojectionCamera', return_default=True)}")


def run(a):
  seed_params(a.route, a.data_dir, a.view, a.debug, a.camera)
  env = dict(os.environ, BIG="1", SCALE=str(1.0 if a.record else a.scale))
  if a.record:
    env.update(RECORD="1", RECORD_OUTPUT=a.record)
  children = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--stage-only"], env=env),
              subprocess.Popen([sys.executable, os.path.join(BASEDIR, "openpilot/selfdrive/ui/ui.py")], env=env)]
  # replay opens a publisher for every service it knows; the stage publishes the (unlogged) outlines itself
  cmd = [os.path.join(BASEDIR, "openpilot/tools/replay/replay"), a.route, "--wide-road", "-b", "reprojectOutlines", "-s", str(a.start)]
  if a.data_dir:
    cmd += ["-d", a.data_dir]
  try:
    subprocess.call(cmd, env=env)
  finally:
    for c in children:
      c.terminate()
    for c in children:
      try:
        c.wait(5)
      except subprocess.TimeoutExpired:
        c.kill()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("route", nargs="?"); ap.add_argument("-d", "--data-dir"); ap.add_argument("-s", "--start", type=int, default=0)
  ap.add_argument("--view", choices=VIEWS); ap.add_argument("--debug", choices=DEBUG)
  ap.add_argument("--camera", choices=CAMERAS)
  ap.add_argument("--scale", type=float, default=0.5, help="ui window scale (the ui is 2160x1080)")
  ap.add_argument("--record", help="record the ui to this mp4 (at scale 1)")
  ap.add_argument("--prefix", default="c4replay")
  ap.add_argument("--stage-only", action="store_true", help="only the stage, for a replay (--wide-road) and ui started by hand")
  a = ap.parse_args()
  if a.stage_only:
    serve()
  elif a.route is None:
    ap.error("a route is required")
  else:
    with OpenpilotPrefix(a.prefix):  # its own params and sockets, cleaned up at the end
      run(a)


if __name__ == "__main__":
  main()
