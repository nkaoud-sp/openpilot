"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import types

import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.modeld.frame_downscale import (CHESTNUT_FRAME_SIZE, FrameDownscaler, _axis_lut, downscale_target,
                                                        frame_scale_matrix, make_downscale, select_frame_size)
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

C3X = (1928, 1208)
C4 = (1344, 760)
DEVICE = 'CPU'


def _reference_downscale(frame: np.ndarray, src, dst) -> np.ndarray:
  src_stride, src_y_height, src_uv_height, _ = get_nv12_info(*src)
  dst_stride, dst_y_height, dst_uv_height, _ = get_nv12_info(*dst)
  y = frame[:src_stride * src_y_height].reshape(src_y_height, src_stride)
  uv = frame[src_stride * src_y_height:src_stride * (src_y_height + src_uv_height)].reshape(src_uv_height, src_stride)
  ref_y = y[_axis_lut(src[1], dst[1], dst_y_height)[:, None], _axis_lut(src[0], dst[0], dst_stride)[None, :]]
  ref_uv = uv[_axis_lut(src[1] // 2, dst[1] // 2, dst_uv_height)[:, None], _axis_lut(src[0] // 2, dst[0] // 2, dst_stride, channels=2)[None, :]]
  return np.concatenate([ref_y.reshape(-1), ref_uv.reshape(-1)])


class TestFrameSizeSelection(OpenpilotTestCase):
  def test_comma_four_is_the_target(self):
    assert CHESTNUT_FRAME_SIZE == C4
    assert downscale_target(*C3X) == C4
    assert downscale_target(*C4) == C4

  def test_prefers_comma_four_jit_on_c3x(self):
    assert select_frame_size([C3X, C4], *C3X) == C4
    assert select_frame_size([C4], *C3X) == C4
    assert select_frame_size([C4], *C4) == C4

  def test_falls_back_to_native_when_pkl_lacks_comma_four(self):
    assert select_frame_size([C3X], *C3X) == C3X

  def test_native_override(self):
    assert select_frame_size([C3X, C4], *C3X, native=True) == C3X
    assert select_frame_size([C4], *C3X, native=True) == C4

  def test_raises_without_a_usable_jit(self):
    with self.assertRaises(KeyError):
      select_frame_size([(1164, 874)], *C3X)


class TestFrameScale(OpenpilotTestCase):
  def test_identity_for_same_size(self):
    np.testing.assert_array_equal(frame_scale_matrix(C4, C4), np.eye(3, dtype=np.float32))

  def test_scaled_warp_samples_the_same_scene_point(self):
    # the warp maps model pixels to camera pixels; scaling the camera frame must scale the projected pixel, not the ray
    intrinsics = np.array([[2648.0, 0.0, C3X[0] / 2], [0.0, 2648.0, C3X[1] / 2], [0.0, 0.0, 1.0]])
    warp = get_warp_matrix(np.array([0.01, -0.02, 0.005]), intrinsics, False)
    scaled = frame_scale_matrix(C3X, C4) @ warp
    for model_px in ((0.0, 0.0), (256.0, 128.0), (511.0, 255.0)):
      p = np.array([*model_px, 1.0])
      cam = warp @ p
      cam = cam[:2] / cam[2]
      cam_scaled = scaled @ p
      cam_scaled = cam_scaled[:2] / cam_scaled[2]
      np.testing.assert_allclose(cam_scaled, cam * np.array([C4[0] / C3X[0], C4[1] / C3X[1]]), rtol=1e-5)


class TestAxisLut(OpenpilotTestCase):
  def test_nearest_index_follows_pixel_centres(self):
    idx = _axis_lut(1928, 1344, 1408)
    assert idx[0] == 0 and idx[1343] == 1927
    assert idx[672] == round((672 + 0.5) * 1928 / 1344 - 0.5)
    assert (idx[1344:] == 1927).all()  # stride padding replicates the last column
    uv = _axis_lut(964, 672, 1408, channels=2)
    assert uv[0] == 0 and uv[1] == 1
    assert (uv[0::2] % 2 == 0).all() and (uv[1::2] % 2 == 1).all()


class TestDownscaleKernel(OpenpilotTestCase):
  def setUp(self):
    super().setUp()
    rng = np.random.default_rng(0)
    self.frame = rng.integers(0, 256, get_nv12_info(*C3X)[3], dtype=np.uint8)
    self.ref = _reference_downscale(self.frame, C3X, C4)

  def test_matches_nearest_reference_in_padded_layout(self):
    from tinygrad.tensor import Tensor
    out = make_downscale(C3X, C4, DEVICE)(Tensor(self.frame.copy(), device=DEVICE).realize()).realize().numpy()
    stride, y_height, uv_height, _ = get_nv12_info(*C4)
    assert out.shape == (stride * (y_height + uv_height),)
    np.testing.assert_array_equal(out, self.ref)

    # stride and height padding replicate the last real pixel, so a warp that clips to cam_w/cam_h never sees garbage
    y = out[:stride * y_height].reshape(y_height, stride)
    np.testing.assert_array_equal(y[C4[1]:, :C4[0]], np.repeat(y[C4[1] - 1:C4[1], :C4[0]], y_height - C4[1], axis=0))
    np.testing.assert_array_equal(y[:C4[1], C4[0]:], np.repeat(y[:C4[1], C4[0] - 1:C4[0]], stride - C4[0], axis=1))

  def test_chroma_channels_stay_interleaved(self):
    from tinygrad.tensor import Tensor
    src_stride, src_y_height, src_uv_height, src_size = get_nv12_info(*C3X)
    frame = np.zeros(src_size, dtype=np.uint8)
    uv = frame[src_stride * src_y_height:src_stride * (src_y_height + src_uv_height)].reshape(src_uv_height, src_stride)
    uv[:, 0::2] = 40   # U
    uv[:, 1::2] = 200  # V
    out = make_downscale(C3X, C4, DEVICE)(Tensor(frame, device=DEVICE).realize()).realize().numpy()
    dst_stride, dst_y_height, dst_uv_height, _ = get_nv12_info(*C4)
    out_uv = out[dst_stride * dst_y_height:].reshape(dst_uv_height, dst_stride)
    assert (out_uv[:, 0::2] == 40).all() and (out_uv[:, 1::2] == 200).all()

  def test_frame_downscaler_keeps_one_host_frame_per_input(self):
    downscaler = FrameDownscaler(C3X, C4, DEVICE)
    buf = types.SimpleNamespace(data=self.frame)  # VisionBuf exposes the frame as .data
    first = downscaler.run('img', buf)
    second = downscaler.run('big_img', buf)
    assert first.shape == (get_nv12_info(*C4)[3],)
    np.testing.assert_array_equal(first[:downscaler.copy_size], self.ref)
    assert np.array_equal(first[:downscaler.copy_size], second[:downscaler.copy_size])
    # the same input returns the same array object, so a pointer taken to it stays valid across runs
    assert downscaler.run('img', buf) is first
    np.testing.assert_allclose(np.diag(downscaler.scale), [C4[0] / C3X[0], C4[1] / C3X[1], 1.0])
