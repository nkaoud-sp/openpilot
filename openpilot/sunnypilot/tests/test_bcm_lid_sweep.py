"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pytest

from openpilot.sunnypilot.autolock_commands import LOCK_CMD, UNLOCK_CMD
from openpilot.sunnypilot.bcm_lid_sweep import (
  BCM_SUBADDR,
  GRID_LIDS,
  SVC_IO_CONTROL,
  SVC_READ,
  _decode_frame,
  build_request,
  parse_payload,
)


class TestBuildRequest:
  def test_read_frame(self):
    # 40 02 21 11, padded: sub-address, single frame of 2, ReadDataByLocalIdentifier, identifier
    assert build_request(0x11, SVC_READ) == b"\x40\x02\x21\x11\x00\x00\x00\x00"

  def test_control_probe_frame(self):
    # Same shape as the auto-lock command, with every data bit cleared.
    assert build_request(0x11, SVC_IO_CONTROL) == b"\x40\x05\x30\x11\x00\x00\x00\x00"

  def test_control_probe_matches_the_working_lock_command_shape(self):
    """The BCM answered the dongle's 5-byte probe with 'no such identifier', not 'bad length',
    so the probe has to carry the same DLC as a command known to work."""
    probe = build_request(0x11, SVC_IO_CONTROL)
    for cmd in (LOCK_CMD, UNLOCK_CMD):
      assert probe[0] == cmd[0], "same sub-address"
      assert probe[1] == cmd[1], "same declared length"
      assert probe[2:4] == cmd[2:4], "same service and identifier"

  def test_probe_drives_nothing(self):
    """All data bits clear: the dongle's `30 21 00 00` left the mirror where it was."""
    assert build_request(0x21, SVC_IO_CONTROL)[4:] == b"\x00\x00\x00\x00"

  def test_frames_are_eight_bytes(self):
    for service in (SVC_READ, SVC_IO_CONTROL):
      for lid in (0x00, 0x11, 0xFF):
        assert len(build_request(lid, service)) == 8

  def test_every_frame_carries_the_body_sub_address(self):
    assert all(build_request(lid)[0] == BCM_SUBADDR for lid in GRID_LIDS)

  def test_rejects_out_of_range_identifier(self):
    with pytest.raises(ValueError):
      build_request(0x100)

  def test_rejects_unknown_service(self):
    with pytest.raises(ValueError):
      build_request(0x11, 0x22)


class TestGridLids:
  def test_covers_the_known_good_identifiers(self):
    # windows 0x01, locks 0x11, mirrors 0x21 - the spacing the grid extrapolates from
    for known in (0x01, 0x11, 0x21):
      assert known in GRID_LIDS

  def test_stays_in_range(self):
    assert all(0 <= lid <= 0xFF for lid in GRID_LIDS)


class TestParsePayload:
  def test_positive_read(self):
    # 61 13 d0 c0, as the body ECU answered the hazard dongle
    r = parse_payload(b"\x61\x13\xd0\xc0", SVC_READ)
    assert r.positive and r.data == "d0 c0"
    assert r.verdict == "LIVE"

  def test_positive_control(self):
    r = parse_payload(b"\x70\x21", SVC_IO_CONTROL)
    assert r.positive and r.verdict == "LIVE"

  def test_no_such_identifier(self):
    # what `30 15 00 c0 00` came back with
    r = parse_payload(b"\x7f\x30\x12", SVC_IO_CONTROL)
    assert not r.positive
    assert r.nrc == 0x12 and r.verdict == "not an identifier"

  def test_conditions_not_correct_is_distinct_from_absent(self):
    """The identifier exists and is refusing - the answer the speed gate gives."""
    r = parse_payload(b"\x7f\x30\x22", SVC_IO_CONTROL)
    assert r.nrc == 0x22 and r.verdict == "conditionsNotCorrect"
    assert r.verdict != parse_payload(b"\x7f\x30\x12", SVC_IO_CONTROL).verdict

  def test_response_pending_is_reported_not_swallowed(self):
    r = parse_payload(b"\x7f\x3b\x78", SVC_IO_CONTROL)
    assert r.nrc == 0x78 and r.verdict == "responsePending"

  def test_empty_payload(self):
    assert parse_payload(b"", SVC_READ).verdict == "no response"

  def test_positive_for_the_other_service_is_not_positive(self):
    """A 0x61 read reply must not count as a live I/O control."""
    assert not parse_payload(b"\x61\x13\xd0", SVC_IO_CONTROL).positive


class TestDecodeFrame:
  def test_single_frame(self):
    kind, payload, _ = _decode_frame(b"\x40\x04\x61\x13\xd0\xc0\x00\x00")
    assert kind == "single" and payload == b"\x61\x13\xd0\xc0"

  def test_single_frame_ignores_padding(self):
    kind, payload, _ = _decode_frame(b"\x40\x02\x70\x21\x00\x00\x00\x00")
    assert kind == "single" and payload == b"\x70\x21"

  def test_first_frame_declares_total_length(self):
    kind, chunk, length = _decode_frame(b"\x40\x10\x09\x62\x11\x01\x87\x07")
    assert kind == "first" and length == 9 and chunk == b"\x62\x11\x01\x87\x07"

  def test_consecutive_frame(self):
    kind, chunk, _ = _decode_frame(b"\x40\x21\x00\x00\x00\x00\x00\x00")
    assert kind == "consecutive" and chunk == b"\x00\x00\x00\x00\x00\x00"

  def test_other_sub_address_is_ignored(self):
    """0x750 is shared: mirror and door modules answer on it too, and aren't ours."""
    assert _decode_frame(b"\xa5\x02\x70\x21\x00\x00\x00\x00")[0] == "other"

  def test_runt_frame_is_ignored(self):
    assert _decode_frame(b"\x40")[0] == "other"
