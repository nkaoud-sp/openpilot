"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.autolock_commands import LOCK_CMD, UNLOCK_CMD
from openpilot.sunnypilot.bcm_lid_sweep import (
  BCM_RESP_ADDR,
  BCM_SUBADDR,
  BUS,
  GRID_LIDS,
  SVC_IO_CONTROL,
  SVC_READ,
  TESTER_PRESENT,
  DeviceLink,
  _decode_frame,
  build_request,
  parse_payload,
  preflight,
  probe,
  sweep,
)

SETTLE = 0.02


class FakeSubMaster:
  """A SubMaster whose [] yields a struct reader, exactly as cereal's does.

  It deliberately has no getDeviceState(): that C++ idiom on a Python reader is the mistake
  this stands guard over.
  """

  def __init__(self, alive: bool, started: bool):
    self.alive = {"deviceState": alive}
    self._reader = SimpleNamespace(started=started)

  def update(self, _timeout):
    pass

  def __getitem__(self, _service):
    return self._reader


class FakeLink:
  """Stands in for the device: answers each request from a scripted table."""

  def __init__(self, answers: dict[int, bytes], broadcasts: dict[int, bytes] | None = None,
               awake: bool = True, consumes_scripts: bool = True, idle: list | None = None):
    self.answers = answers          # lid -> response frame, or absent for no reply
    self.broadcasts = broadcasts or {}
    self.awake = awake              # whether the ECU answers TesterPresent
    self.consumes_scripts = consumes_scripts   # whether pandad picks the script up
    self.idle = idle or []          # background bus traffic, replayed on every poll
    self.sent: list[bytes] = []
    self._queue: list[tuple[int, bytes]] = []

  def send(self, frame: bytes):
    self.sent.append(frame)
    if frame[2] == 0x3E:
      if self.awake:
        self._queue.append((BCM_RESP_ADDR, b"\x40\x01\x7e\x00\x00\x00\x00\x00"))
      return
    if frame[2] in (SVC_READ, SVC_IO_CONTROL):
      lid = frame[3]
      if lid in self.answers:
        self._queue.append((BCM_RESP_ADDR, self.answers[lid]))
      for addr, dat in self.broadcasts.get(lid, {}).items():
        self._queue.append((addr, dat))

  def script_pending(self):
    return not self.consumes_scripts

  def poll(self):
    out, self._queue = self._queue + list(self.idle), []
    return out


class TestBuildRequest:
  def test_read_frame(self):
    # 40 02 21 11: sub-address, single frame of 2, ReadDataByLocalIdentifier, identifier
    assert build_request(0x11, SVC_READ) == b"\x40\x02\x21\x11\x00\x00\x00\x00"

  def test_control_probe_frame(self):
    assert build_request(0x11, SVC_IO_CONTROL) == b"\x40\x05\x30\x11\x00\x00\x00\x00"

  def test_control_probe_matches_the_working_lock_command_shape(self):
    """The BCM answered the dongle's 5-byte probe with 'no such identifier', not 'bad length',
    so the probe has to carry the same DLC as a command known to work."""
    probe_frame = build_request(0x11, SVC_IO_CONTROL)
    for cmd in (LOCK_CMD, UNLOCK_CMD):
      assert probe_frame[0] == cmd[0], "same sub-address"
      assert probe_frame[1] == cmd[1], "same declared length"
      assert probe_frame[2:4] == cmd[2:4], "same service and identifier"

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
    assert r.positive and r.data == "d0 c0" and r.verdict == "LIVE"

  def test_positive_control(self):
    assert parse_payload(b"\x70\x21", SVC_IO_CONTROL).verdict == "LIVE"

  def test_no_such_identifier(self):
    # what `30 15 00 c0 00` came back with
    r = parse_payload(b"\x7f\x30\x12", SVC_IO_CONTROL)
    assert not r.positive and r.nrc == 0x12 and r.verdict == "not an identifier"

  def test_conditions_not_correct_is_distinct_from_absent(self):
    """The identifier exists and is refusing - the answer the speed gate gives."""
    r = parse_payload(b"\x7f\x30\x22", SVC_IO_CONTROL)
    assert r.nrc == 0x22 and r.verdict == "conditionsNotCorrect"
    assert r.verdict != parse_payload(b"\x7f\x30\x12", SVC_IO_CONTROL).verdict

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

  def test_other_sub_address_is_ignored(self):
    """0x750 is shared: mirror and door modules answer on it too, and aren't ours."""
    assert _decode_frame(b"\xa5\x02\x70\x21\x00\x00\x00\x00")[0] == "other"

  def test_runt_frame_is_ignored(self):
    assert _decode_frame(b"\x40")[0] == "other"


class TestProbe:
  def test_live_identifier(self):
    link = FakeLink({0x11: b"\x40\x02\x70\x11\x00\x00\x00\x00"})
    resp, changed = probe(link, 0x11, SVC_IO_CONTROL, SETTLE)
    assert resp.positive and not changed

  def test_absent_identifier(self):
    link = FakeLink({0x15: b"\x40\x03\x7f\x30\x12\x00\x00\x00"})
    resp, _ = probe(link, 0x15, SVC_IO_CONTROL, SETTLE)
    assert resp.verdict == "not an identifier"

  def test_silence_is_reported_not_guessed(self):
    resp, _ = probe(FakeLink({}), 0x42, SVC_READ, SETTLE)
    assert resp.verdict == "no response"

  def test_reply_from_another_sub_address_is_not_ours(self):
    """A mirror module answering on 0x758 must not be read as the BCM replying."""
    link = FakeLink({0x21: b"\xa5\x02\x70\x21\x00\x00\x00\x00"})
    resp, _ = probe(link, 0x21, SVC_IO_CONTROL, SETTLE)
    assert resp.verdict == "no response"

  def test_response_pending_waits_for_the_real_answer(self):
    """The body ECU answers 0x78 first and the positive ~85 ms later; the 0x78 isn't the answer."""
    link = FakeLink({0x13: b"\x40\x03\x7f\x30\x78\x00\x00\x00"})
    resp, _ = probe(link, 0x13, SVC_IO_CONTROL, SETTLE)
    assert resp.verdict != "responsePending" or resp.nrc == 0x78

  def test_actuation_is_attributed_to_the_identifier_that_caused_it(self):
    link = FakeLink(
      {0x31: b"\x40\x02\x70\x31\x00\x00\x00\x00"},
      broadcasts={0x31: {0x614: b"\x29\x80\x7e\x38\x00\x00\x0b\x74"}},
    )
    # seed the pre-probe snapshot with a different value so the change is visible
    link._queue.append((0x614, b"\x29\x00\x7e\x30\x00\x00\x0b\x74"))
    _, changed = probe(link, 0x31, SVC_IO_CONTROL, SETTLE)
    assert "0x614" in changed and "38" in changed["0x614"]

  def test_sends_exactly_one_request(self):
    """A KWP negative doesn't echo its identifier, so only one probe may be in flight."""
    link = FakeLink({0x11: b"\x40\x03\x7f\x30\x12\x00\x00\x00"})
    probe(link, 0x11, SVC_IO_CONTROL, SETTLE, wake=False)
    assert len(link.sent) == 1

  def test_wake_sends_tester_present_first(self):
    """The dongle never read an identifier without a TesterPresent in front of it."""
    link = FakeLink({0x11: b"\x40\x02\x70\x11\x00\x00\x00\x00"})
    probe(link, 0x11, SVC_IO_CONTROL, SETTLE, wake=True)
    assert link.sent[0] == TESTER_PRESENT
    assert link.sent[1][2:4] == b"\x30\x11"

  def test_tester_present_reply_is_not_read_as_the_answer(self):
    """0x7E arriving late must not be scored as this identifier's response."""
    link = FakeLink({}, awake=True)
    resp, _ = probe(link, 0x41, SVC_READ, SETTLE, wake=True)
    assert resp.verdict == "no response"


class TestPreflight:
  """A silent sweep has three very different causes; preflight has to tell them apart."""

  BUS_TRAFFIC = [(0x620, b"\x10\x00\x00\x00\xf0\x00\x08\x5a")]

  def test_passes_when_everything_is_working(self):
    link = FakeLink({}, awake=True, consumes_scripts=True, idle=self.BUS_TRAFFIC)
    assert preflight(link, log=lambda *_: None, listen=0.05, timeout=0.05) is True

  def test_fails_on_a_silent_bus(self):
    """Car asleep: no broadcasts at all, so every identifier would look absent."""
    link = FakeLink({}, awake=True, consumes_scripts=True, idle=[])
    assert preflight(link, log=lambda *_: None, listen=0.05, timeout=0.05) is False

  def test_fails_when_pandad_never_takes_the_script(self):
    """Nothing was transmitted, so the sweep would be measuring nothing."""
    link = FakeLink({}, awake=True, consumes_scripts=False, idle=self.BUS_TRAFFIC)
    assert preflight(link, log=lambda *_: None, listen=0.05, timeout=0.05) is False

  def test_fails_when_the_body_ecu_does_not_answer(self):
    link = FakeLink({}, awake=False, consumes_scripts=True, idle=self.BUS_TRAFFIC)
    assert preflight(link, log=lambda *_: None, listen=0.05, timeout=0.05) is False

  def test_reports_each_check_by_name(self):
    lines = []
    preflight(FakeLink({}, awake=False, consumes_scripts=True, idle=self.BUS_TRAFFIC), log=lines.append, listen=0.05, timeout=0.05)
    text = "\n".join(lines)
    for check in ("bus 0 traffic", "pandad picked up", "body ECU awake"):
      assert check in text


class TestSweep:
  def test_reports_every_identifier(self):
    lids = [0x01, 0x11, 0x21]
    link = FakeLink({0x11: b"\x40\x02\x70\x11\x00\x00\x00\x00"})
    out = sweep(link, lids, SVC_IO_CONTROL, SETTLE, log=lambda *_: None)
    assert set(out["results"]) == {"0x01", "0x11", "0x21"}
    assert out["results"]["0x11"]["positive"]
    assert not out["results"]["0x01"]["positive"]

  def test_result_is_json_serialisable(self):
    import json
    link = FakeLink({0x11: b"\x40\x04\x61\x11\xd0\xc0\x00\x00"})
    json.dumps(sweep(link, [0x11], SVC_READ, SETTLE, log=lambda *_: None))


class TestDeviceLink:
  """DeviceLink is the only code that touches cereal, so pin the API it uses."""

  @staticmethod
  def _stub(**attrs):
    link = DeviceLink.__new__(DeviceLink)   # no device, no sockets
    for k, v in attrs.items():
      setattr(link, k, v)
    return link

  def test_onroad_reads_started_off_the_reader(self):
    """SubMaster[...] already returns the struct reader - there is no getDeviceState()."""
    assert self._stub(sm=FakeSubMaster(alive=True, started=True)).onroad() is True

  def test_onroad_is_false_when_the_device_is_parked(self):
    assert self._stub(sm=FakeSubMaster(alive=True, started=False)).onroad() is False

  def test_onroad_is_false_when_deviceState_is_stale(self):
    assert self._stub(sm=FakeSubMaster(alive=False, started=True)).onroad() is False

  def test_send_encodes_a_playable_script_to_the_body_ecu(self):
    from openpilot.sunnypilot.hazard_flash import SCRIPT_RECORD_LEN, ScriptFrame, encode_script

    written = {}
    link = self._stub(
      params=SimpleNamespace(put=lambda k, v: written.__setitem__(k, v)),
      _ScriptFrame=ScriptFrame, _encode_script=encode_script,
    )
    link.send(build_request(0x11, SVC_READ))

    script = written["OffroadCanScript"]
    assert len(script) == SCRIPT_RECORD_LEN, "one frame per probe"
    assert (script[2] << 8 | script[3]) == 0x750
    assert script[4] == BUS
    assert script[6:6 + script[5]] == b"\x40\x02\x21\x11\x00\x00\x00\x00"

  def test_poll_keeps_only_bus_zero_and_returns_bytes(self):
    link = self._stub(
      can_sock=None,
      _messaging=SimpleNamespace(drain_sock_raw=lambda _s: [b""]),
      _can_capnp_to_list=lambda _raw: [(0, [(0x758, b"\x40\x02\x70\x11", BUS),
                                            (0x758, b"\xff", 2)])],
    )
    assert link.poll() == [(0x758, b"\x40\x02\x70\x11")]
