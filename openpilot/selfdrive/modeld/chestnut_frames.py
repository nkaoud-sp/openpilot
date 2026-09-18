"""Picks the chestnut frame stage: what happens to the comma 3X camera frames before they cross the link.

Three modes, chosen by params (see the Tweaks menu):

  C4 Resampling (default)  frame_downscale.FrameDownscaler  nearest resize to 1344x760, warp scaled to match
  C4 Ray Matching          reproject_c4.Reprojector         3X lenses traced into comma 4 camera geometry
  native                   None                             full 1928x1208 frames, ~12 ms more on the link

Both stages expose the same contract:

  process(bufs, gains) -> {model input name: host NV12 frame}, valid until the next call
  scale                   3x3 applied to the warp matrices
  c4_intrinsics           build the warp from comma 4 intrinsics instead of this device's
  dst_size                the frame size the model pkl must have a JIT for
"""
import os
from typing import TYPE_CHECKING

from openpilot.selfdrive.modeld.frame_downscale import CHESTNUT_FRAME_SIZE, FrameDownscaler, downscale_target

if TYPE_CHECKING:
  from openpilot.common.params import Params

# Ray matching needs both cameras and the tables are built for this exact pair of geometries.
RAY_MATCH_SRC = (1928, 1208)
TABLE_CACHE_DIR = os.environ.get('XDG_CACHE_HOME', '/data/tgcache')


def frame_mode(params: "Params | None" = None) -> str:
  if params is None:
    from openpilot.common.params import Params
    params = Params()
  if params.get_bool("ChestnutNativeFrames"):
    return "native"
  return "raymatch" if params.get_bool("ChestnutRayMatching") else "resample"


def make_frame_stage(cam_size: tuple[int, int], device: str, mode: str, both_cameras: bool = True):
  """Returns the stage for `mode`, or None when the frames go to the card untouched."""
  if mode == "native" or downscale_target(*cam_size) == cam_size:
    return None

  if mode == "raymatch":
    if both_cameras and cam_size == RAY_MATCH_SRC:
      from openpilot.selfdrive.modeld.reproject_c4 import Reprojector
      stage = Reprojector(cam_size, CHESTNUT_FRAME_SIZE, device=device, cache_dir=TABLE_CACHE_DIR)
      stage.dst_size = CHESTNUT_FRAME_SIZE
      return stage
    from openpilot.common.swaglog import cloudlog
    cloudlog.warning(f"chestnut: ray matching needs both {RAY_MATCH_SRC} cameras, got {cam_size} both={both_cameras}; resampling instead")

  stage = FrameDownscaler(cam_size, CHESTNUT_FRAME_SIZE, device)
  stage.dst_size = CHESTNUT_FRAME_SIZE
  return stage


if __name__ == "__main__":
  # on-device timing of both modes: python3 -m openpilot.selfdrive.modeld.chestnut_frames [--device QCOM]
  import argparse
  import time
  import types

  import numpy as np

  from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

  p = argparse.ArgumentParser()
  # never ask tinygrad for a default device here: with a chestnut attached it probes the AMD card and fetches firmware
  p.add_argument('--device', default='QCOM' if os.path.exists('/dev/kgsl-3d0') else 'CPU')
  p.add_argument('--runs', type=int, default=50)
  p.add_argument('--modes', nargs='+', default=['resample', 'raymatch'])
  args = p.parse_args()

  cam = RAY_MATCH_SRC
  rng = np.random.default_rng(0)
  bufs = {k: types.SimpleNamespace(data=rng.integers(0, 256, get_nv12_info(*cam)[3], dtype=np.uint8))
          for k in ('img', 'big_img')}
  for mode in args.modes:
    try:
      st = time.perf_counter()
      stage = make_frame_stage(cam, args.device, mode)
      init_ms = (time.perf_counter() - st) * 1e3
      stage.process(bufs)
      timings = []
      for _ in range(args.runs):
        st = time.perf_counter()
        stage.process(bufs)
        timings.append((time.perf_counter() - st) * 1e3)
      print(f"{mode}: init {init_ms / 1e3:.1f} s, frame pair median {np.median(timings):.2f} ms, max {max(timings):.2f} ms over {args.runs} runs")
    except Exception as e:
      print(f"{mode}: FAILED {type(e).__name__}: {e}")
