"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import types

import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.modeld.chestnut_frames import RAY_MATCH_SRC, frame_mode, make_frame_stage
from openpilot.selfdrive.modeld.frame_downscale import CHESTNUT_FRAME_SIZE, FrameDownscaler
from openpilot.selfdrive.modeld.reproject_c4 import (ALPHA_SHIFT, IDX_BITS, INVALID_BIT, UV_FILL, Reprojector,
                                                     build_tables, sample_coords)
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

C3X, C4 = RAY_MATCH_SRC, CHESTNUT_FRAME_SIZE
DEVICE = 'CPU'


class StubParams:
  def __init__(self, **flags):
    self.flags = flags

  def get_bool(self, key):
    return self.flags.get(key, False)


class TestFrameMode(OpenpilotTestCase):
  def test_resampling_is_the_default(self):
    assert frame_mode(StubParams()) == "resample"

  def test_ray_matching_selected(self):
    assert frame_mode(StubParams(ChestnutRayMatching=True)) == "raymatch"

  def test_native_wins_over_ray_matching(self):
    assert frame_mode(StubParams(ChestnutNativeFrames=True, ChestnutRayMatching=True)) == "native"


class TestStageFactory(OpenpilotTestCase):
  def test_native_builds_no_stage(self):
    assert make_frame_stage(C3X, DEVICE, "native") is None

  def test_comma_four_camera_needs_no_stage(self):
    assert make_frame_stage(C4, DEVICE, "resample") is None
    assert make_frame_stage(C4, DEVICE, "raymatch") is None

  def test_resample_mode_builds_the_downscaler(self):
    stage = make_frame_stage(C3X, DEVICE, "resample")
    assert isinstance(stage, FrameDownscaler) and stage.dst_size == C4
    assert not stage.c4_intrinsics  # keeps this device's intrinsics, scaled

  def test_ray_matching_falls_back_without_both_cameras(self):
    # the narrow output is composited from the wide surround, so one camera cannot produce it
    stage = make_frame_stage(C3X, DEVICE, "raymatch", both_cameras=False)
    assert isinstance(stage, FrameDownscaler)


class TestReprojectorContract(OpenpilotTestCase):
  @classmethod
  def setUpClass(cls):
    cls.rp = Reprojector(C3X, C4, device=DEVICE, cache_dir=None)
    rng = np.random.default_rng(0)
    cls.src = {k: rng.integers(0, 256, get_nv12_info(*C3X)[3], dtype=np.uint8) for k in ('img', 'big_img')}
    cls.bufs = {k: types.SimpleNamespace(data=v) for k, v in cls.src.items()}

  def test_declares_comma_four_geometry(self):
    assert self.rp.c4_intrinsics  # the warp must switch to comma 4 intrinsics, not scale
    np.testing.assert_array_equal(self.rp.scale, np.eye(3, dtype=np.float32))

  def test_emits_one_persistent_frame_per_model_input(self):
    out = self.rp.process(self.bufs)
    assert set(out) == {'img', 'big_img'}
    for frame in out.values():
      assert frame.shape == (get_nv12_info(*C4)[3],)
    again = self.rp.process(self.bufs)
    for key in out:
      assert again[key] is out[key]  # same array, so a pointer taken to it stays valid across runs

  def test_matches_a_numpy_reference_of_the_same_gather(self):
    chroma = np.zeros(self.rp.copy_size, bool)
    chroma[self.rp.uv_offset:] = True
    tables = build_tables(C3X, C4)
    wide, narrow = self.src['big_img'], self.src['img']

    for gain_y, gain_c in ((1.0, 1.0), (1.7, 0.8)):
      out = self.rp.process(self.bufs, (gain_y, gain_c))

      pw = tables["wide"]["pw"].astype(np.int64)
      ref_wide = np.where(pw < INVALID_BIT, wide[pw & IDX_BITS], (chroma * UV_FILL).astype(np.uint8))
      np.testing.assert_array_equal(out['big_img'][:self.rp.copy_size], ref_wide.astype(np.uint8))

      pwn, pn = tables["narrow"]["pw"].astype(np.int64), tables["narrow"]["pn"].astype(np.int64)
      w = wide[pwn & IDX_BITS].astype(np.float64)
      w = np.where(chroma, (w - UV_FILL) * gain_c + UV_FILL, w * gain_y).clip(0, 255)
      w = np.where(pwn < INVALID_BIT, w, chroma * float(UV_FILL))
      alpha = ((pwn >> ALPHA_SHIFT) & 0xff).astype(np.float64) / 255
      ref_narrow = np.round(alpha * narrow[pn].astype(np.float64) + (1 - alpha) * w)
      assert np.abs(out['img'][:self.rp.copy_size].astype(int) - ref_narrow).max() <= 1  # float rounding order


class TestReprojectionGeometry(OpenpilotTestCase):
  @classmethod
  def setUpClass(cls):
    cls.tables = build_tables(C3X, C4)

  def test_comma_four_centre_lands_on_the_3x_narrow_axis(self):
    _, mn = sample_coords("narrow", *C4)
    centre = mn[C4[1] // 2, C4[0] // 2]
    # the 3X narrow principal point, within a few px: straight ahead is straight ahead through both lenses
    np.testing.assert_allclose(centre, (964.0, 604.0), atol=8.0)

  def test_narrow_output_really_is_a_composite(self):
    alpha = (self.tables["narrow"]["pw"].astype(np.int64) >> ALPHA_SHIFT) & 0xff
    # the comma 4 narrow is much wider than the 3X narrow, so a large minority is inset and the rest surround
    assert (alpha == 255).mean() > 0.2, "expected a real 3X narrow inset"
    assert (alpha == 0).mean() > 0.2, "expected a real 3X wide surround"
    assert 0 < ((alpha > 0) & (alpha < 255)).mean() < 0.2, "expected a feathered seam, not a hard edge"

  def test_unsourced_pixels_stay_off_the_model_crop(self):
    # the 3X wide cannot fill every comma 4 wide pixel; what it misses must sit outside where the warp samples
    stride, y_height, _, _ = get_nv12_info(*C4)
    invalid = (self.tables["wide"]["pw"].astype(np.int64) >= INVALID_BIT)[:stride * y_height]
    invalid = invalid.reshape(y_height, stride)[:C4[1], :C4[0]]
    assert invalid.mean() < 0.10
    h, w = C4[1], C4[0]
    central = invalid[int(h * .125):int(h * .875), int(w * .125):int(w * .875)]
    assert not central.any(), "the central 3/4 of the frame must be fully sourced"
