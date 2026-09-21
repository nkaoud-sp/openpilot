#!/usr/bin/env python3
import argparse
import pickle
import time

from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor

from openpilot.common.transformations.model import MEDMODEL_INPUT_SIZE
from openpilot.selfdrive.modeld.compile_modeld import NV12Frame, make_frame_prepare, _parse_size
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


def make_driving_warp(nv12: NV12Frame, model_w: int, model_h: int):
  frame_prepare = make_frame_prepare(nv12, model_w, model_h)

  def warp(input_frame, M_inv):
    input_frame = input_frame.to(Device.DEFAULT)
    M_inv = M_inv.to(Device.DEFAULT)
    Tensor.realize(input_frame, M_inv)

    warped_frame = frame_prepare(input_frame[0], M_inv[0]).unsqueeze(0)
    warped_big_frame = frame_prepare(input_frame[1], M_inv[1]).unsqueeze(0)
    return Tensor.cat(warped_frame, warped_big_frame)

  return warp


def compile_driving_warp(nv12: NV12Frame, model_w: int, model_h: int, pkl_path: str, benchmark_runs: int) -> None:
  print(f"Compiling driving warp for {nv12.width}x{nv12.height} -> {model_w}x{model_h}...")
  warp_jit = TinyJit(make_driving_warp(nv12, model_w, model_h), prune=True)

  for i in range(benchmark_runs):
    frame = Tensor.randint((2, nv12.stride * (nv12.y_height + nv12.uv_height)), low=0, high=256, dtype='uint8').realize()
    M_inv = Tensor(Tensor.randn(2, 3, 3).mul(8).realize().numpy(), device='NPY')
    Device.default.synchronize()
    st = time.perf_counter()
    warp_jit(frame, M_inv).realize()
    mt = time.perf_counter()
    Device.default.synchronize()
    et = time.perf_counter()
    print(f"  [{i + 1}/{benchmark_runs}] enqueue {(mt - st) * 1e3:6.2f} ms -- total {(et - st) * 1e3:6.2f} ms")

  with open(pkl_path, "wb") as f:
    pickle.dump({'run': warp_jit}, f)
  print(f"  Saved to {pkl_path}")


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument('--camera-resolution', type=_parse_size, required=True, help='camera resolution WxH')
  p.add_argument('--warp-to', type=_parse_size, default=MEDMODEL_INPUT_SIZE, help='model input WxH')
  p.add_argument('--output', required=True)
  p.add_argument('--benchmark-runs', type=int, default=10)
  args = p.parse_args()

  cam_w, cam_h = args.camera_resolution
  nv12 = NV12Frame(cam_w, cam_h, *get_nv12_info(cam_w, cam_h))
  model_w, model_h = args.warp_to
  compile_driving_warp(nv12, model_w, model_h, args.output, args.benchmark_runs)
