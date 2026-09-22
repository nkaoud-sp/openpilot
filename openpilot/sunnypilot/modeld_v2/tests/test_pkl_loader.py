"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import os
import pickle
import tempfile

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.modeld.helpers import dump_oob
from openpilot.sunnypilot.modeld_v2 import helpers as helpers_module
from openpilot.sunnypilot.modeld_v2.helpers import DynamicTinygradUnpickler, load_driving_pkl


def _write(path: str) -> None:
  with open(path, 'wb') as f:
    dump_oob({'run': 'jit', 'input_specs': {}, 'output_specs': {},
              'weights': pickle.PickleBuffer(bytearray(4096))}, f)


class TestLoadDrivingPkl(OpenpilotTestCase):
  def test_loads_with_the_plain_unpickler(self, monkeypatch):
    used = []
    real = helpers_module._read_oob
    monkeypatch.setattr(helpers_module, '_read_oob',
                        lambda f, cls: (used.append(cls), real(f, cls))[1])

    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'driving_test_tinygrad.pkl')
      _write(path)
      jits = load_driving_pkl(path)

    assert sorted(jits) == ['input_specs', 'output_specs', 'run', 'weights']
    # the compatibility unpickler rewrites tinygrad globals, so it must not run unprompted
    assert used == [pickle.Unpickler], used

  def test_falls_back_to_the_compatibility_unpickler(self, monkeypatch):
    used = []

    def flaky(f, cls):
      used.append(cls)
      if cls is pickle.Unpickler:
        raise AttributeError("'function' object has no attribute 'LINEAR'")
      return {'run': 'jit'}

    monkeypatch.setattr(helpers_module, '_read_oob', flaky)

    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'driving_test_tinygrad.pkl')
      _write(path)
      jits = load_driving_pkl(path)

    assert jits == {'run': 'jit'}
    assert used == [pickle.Unpickler, DynamicTinygradUnpickler], used

  def test_raises_when_both_unpicklers_fail(self, monkeypatch):
    monkeypatch.setattr(helpers_module, '_read_oob',
                        lambda f, cls: (_ for _ in ()).throw(ValueError("broken pkl")))

    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'driving_test_tinygrad.pkl')
      _write(path)
      with self.assertRaisesRegex(ValueError, "broken pkl"):
        load_driving_pkl(path)
