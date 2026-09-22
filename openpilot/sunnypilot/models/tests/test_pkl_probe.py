"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import io
import os
import pickle
import struct
import tempfile

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.modeld.helpers import dump_oob
from openpilot.sunnypilot.models.pkl_probe import (CHESTNUT_DEV, attempt_loads, key_marker, probe, schema_of,
                                                   walk_buffers)


def _write_persistent_id_pkl(path: str, payload: bytes) -> None:
  """The shape tinygrad #18260 writes: opcodes, then a raw arena addressed by persistent ids."""
  opcodes = io.BytesIO()
  pickler = pickle.Pickler(opcodes)
  pickler.persistent_id = lambda obj: (len(payload), 'uchar', 0) if obj is _ARENA else None
  pickler.dump({'run': _ARENA, 'input_specs': {}, 'output_specs': {}})
  with open(path, 'wb') as f:
    f.write(struct.pack('<q', len(blob := opcodes.getvalue())))
    f.write(blob)
    f.write(payload)


class _Arena:
  pass


_ARENA = _Arena()


class TestSchemaOf(OpenpilotTestCase):
  def test_upstream_run_schema(self):
    assert "upstream run-schema" in schema_of(['run', 'input_specs', 'output_specs'])

  def test_sunnypilot_run_model(self):
    assert "run_model" in schema_of(['run_model', 'metadata'])

  def test_sunnypilot_split(self):
    assert "vision/policy" in schema_of(['run_policy', 'metadata'])

  def test_unknown(self):
    assert schema_of(['metadata']) == "unknown"

  def test_run_alone_is_not_upstream(self):
    # a warp pkl is {'run': jit} too, so 'run' on its own must not claim the upstream schema
    assert schema_of(['run']) == "unknown"


class TestKeyMarker(OpenpilotTestCase):
  def test_marker_matches_a_real_pickle(self):
    blob = pickle.dumps({'input_specs': 1}, protocol=5)
    assert key_marker('input_specs') in blob
    assert key_marker('output_specs') not in blob


class TestWalkBuffers(OpenpilotTestCase):
  def test_length_prefixed_records_are_recognized(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'driving_test_tinygrad.pkl')
      with open(path, 'wb') as f:
        dump_oob({'run_model': {}, 'metadata': {}, 'payload': pickle.PickleBuffer(bytearray(4096))}, f)

      lines = probe(path, attempt_load=False)
      assert any("length-prefixed records" in line for line in lines), lines
      assert any("run_model" in line for line in lines), lines
      assert any(f"size: {os.path.getsize(path)} bytes" in line for line in lines), lines

  def test_persistent_id_arena_is_rejected(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'driving_v3_tinygrad.pkl')
      # high bytes set, so reading the arena as an int64 length gives an implausible record
      _write_persistent_id_pkl(path, b'\xff' * 8192)

      lines = probe(path, attempt_load=False)
      assert any("persistent-id arena" in line for line in lines), lines
      assert any("upstream run-schema" in line for line in lines), lines

  def test_truncated_file_is_reported(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'driving_short_tinygrad.pkl')
      with open(path, 'wb') as f:
        dump_oob({'run_model': {}, 'payload': pickle.PickleBuffer(bytearray(4096))}, f)
      with open(path, 'rb+') as f:
        f.truncate(os.path.getsize(path) - 2048)

      lines = probe(path, attempt_load=False)
      assert any("persistent-id arena" in line or "incomplete" in line for line in lines), lines

  def test_empty_buffer_section_is_clean(self):
    assert walk_buffers(io.BytesIO(b'')).startswith("length-prefixed records, 0")

  def test_unreadable_path(self):
    lines = probe('/nonexistent/driving_missing_tinygrad.pkl', attempt_load=False)
    assert any("unreadable" in line for line in lines), lines


class TestAttemptLoads(OpenpilotTestCase):
  def test_plain_load_reports_top_level_keys_and_skips_the_retry(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'driving_ok_tinygrad.pkl')
      with open(path, 'wb') as f:
        dump_oob({'run_model': {}, 'metadata': {}}, f)

      lines = attempt_loads(path)
      # the plain unpickler and modeld_v2's compatibility one are both reported
      assert len(lines) == 2, lines
      assert "plain unpickle: ok, top level metadata, run_model" in lines[0], lines
      assert lines[1].startswith("  - compat unpickle:"), lines

  def test_failed_load_retries_in_the_chestnut_device_context(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'driving_v3_tinygrad.pkl')
      _write_persistent_id_pkl(path, b'\xff' * 8192)

      lines = attempt_loads(path)
      # plain, compat, then whatever the retry under Context(DEV=...) managed
      assert len(lines) == 3, lines
      assert lines[0].startswith("  - plain unpickle: UnpicklingError"), lines
      assert lines[1].startswith("  - compat unpickle:"), lines
      assert CHESTNUT_DEV in lines[2], lines
