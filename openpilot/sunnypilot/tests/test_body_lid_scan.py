"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pytest

from openpilot.sunnypilot.autolock_commands import LOCK_CMD, MIRR_FOLD_R, UNLOCK_CMD
from openpilot.sunnypilot.body_lid_scan import (
  BLINKERS_STATE_ADDR,
  DIAG_ADDR,
  PROBE_GAP_MS,
  REPLY_ADDR,
  SWEEP_OFF_MS,
  SWEEP_ON_MS,
  NOT_PLAYED_WARNING,
  ScanRecorder,
  build_bit_sweep_frames,
  build_lid_scan_frames,
  build_probe,
  classify_reply,
  combo_controls,
  describe_blinkers_state,
  probe_body,
  probe_lid,
  sweep_controls,
)
from openpilot.sunnypilot.hazard_flash import script_duration_s

BLINKERS_IDLE = bytes.fromhex("29 00 7e 30 00 00 0b 74")
BLINKERS_HAZARD_ON = bytes.fromhex("29 80 7e 38 00 00 0b 74")   # from the rlog: pulse + HAZARD_LIGHT


class TestProbe:
  def test_matches_the_commands_the_fork_already_sends(self):
    # The known-good commands are the format the scan has to reproduce.
    assert build_probe(0x11, b"\x00\x80\x00") == LOCK_CMD
    assert build_probe(0x11, b"\x00\x40\x00") == UNLOCK_CMD
    assert build_probe(0x21, b"\x00\x08", sub_addr=0xA5) == MIRR_FOLD_R

  def test_default_control_is_all_zeros(self):
    assert build_probe(0x5A) == bytes([0x40, 0x05, 0x30, 0x5A, 0x00, 0x00, 0x00, 0x00])

  def test_length_byte_covers_service_lid_and_control(self):
    assert build_probe(0x01, b"\x05\x20\x01")[1] == 5

  def test_rejects_bad_arguments(self):
    with pytest.raises(ValueError):
      build_probe(0x100)
    with pytest.raises(ValueError):
      build_probe(0x01, b"")
    with pytest.raises(ValueError):
      build_probe(0x01, b"\x00" * 5)

  def test_probe_lid_only_recognises_our_frames(self):
    assert probe_lid(build_probe(0x33)) == 0x33
    assert probe_lid(build_probe(0x33, sub_addr=0xA5)) is None
    assert probe_lid(b"\x40\x02\x21\xa3\x00\x00\x00\x00") is None   # a read, not a control


class TestScripts:
  def test_scan_covers_every_lid_once_with_the_gap(self):
    frames = build_lid_scan_frames()
    assert len(frames) == 256
    assert [probe_lid(f.data) for f in frames] == list(range(256))
    assert frames[0].delay_ms == 0
    assert all(f.delay_ms == PROBE_GAP_MS for f in frames[1:])
    assert all(f.addr == DIAG_ADDR and f.bus == 0 for f in frames)
    assert 50 < script_duration_s(frames) < 60

  def test_scan_range_and_sub_address(self):
    frames = build_lid_scan_frames(range(0x10, 0x13), sub_addr=0xA5)
    assert [f.data[0] for f in frames] == [0xA5] * 3
    assert [f.data[3] for f in frames] == [0x10, 0x11, 0x12]

  def test_sweep_sets_one_bit_then_releases_it(self):
    controls = sweep_controls()
    assert len(controls) == 16
    assert all(bin(int.from_bytes(c, "big")).count("1") == 1 for c in controls)
    assert len(set(controls)) == 16
    assert controls[7] == b"\x00\x80\x00", "the lock bit is in the first byte swept"

    frames = build_bit_sweep_frames(0x15)
    assert len(frames) == 32
    for i in range(0, 32, 2):
      assert frames[i].data[4:7] == controls[i // 2]
      assert frames[i + 1].data[4:7] == b"\x00\x00\x00"
      assert frames[i + 1].delay_ms == SWEEP_ON_MS
      assert frames[i].delay_ms == (0 if i == 0 else SWEEP_OFF_MS)
    assert all(probe_lid(f.data) == 0x15 for f in frames)


  def test_combo_sweep_pairs_every_selector_with_every_action_bit(self):
    controls = combo_controls()
    assert len(controls) == 128
    assert len(set(controls)) == 128
    assert controls[:8] == sweep_controls()[:8], "selector 0 is the single-bit sweep's second byte"
    assert bytes([0x05, 0x20, 0x00]) in controls, "the window command's shape is covered"
    frames = build_bit_sweep_frames(0x12, on_ms=500, off_ms=300, controls=controls)
    assert len(frames) == 256
    assert frames[1].delay_ms == 500 and frames[2].delay_ms == 300


class TestClassifyReply:
  def test_positive(self):
    assert classify_reply(bytes.fromhex("40 03 70 11 00 00 00 00"))[0] == "positive"

  def test_unsupported_lid(self):
    kind, detail = classify_reply(bytes.fromhex("40 03 7f 30 12 00 00 00"))
    assert kind == "unsupported"
    assert "0x12" in detail and "subFunctionNotSupported" in detail

  def test_refused_lid_is_not_unsupported(self):
    kind, detail = classify_reply(bytes.fromhex("40 03 7f 30 22 00 00 00"))
    assert kind == "negative"
    assert "conditionsNotCorrect" in detail

  def test_other_sub_address_or_service(self):
    assert classify_reply(bytes.fromhex("e9 04 61 15 50 01 00 00"))[0] == "other"
    assert classify_reply(bytes.fromhex("40 03 7f 21 12 00 00 00"))[0] == "other"


class TestDescribeBlinkersState:
  def test_names_the_dbc_signals(self):
    text = describe_blinkers_state(BLINKERS_IDLE, BLINKERS_HAZARD_ON)
    assert "event pulse on" in text
    assert "HAZARD_LIGHT=1" in text
    assert "TURN_SIGNALS" not in text

  def test_turn_signal(self):
    left = bytes.fromhex("29 80 7e 10 00 00 0b 74")
    assert "TURN_SIGNALS=left" in describe_blinkers_state(BLINKERS_IDLE, left)

  def test_falls_back_to_hex(self):
    other = bytes.fromhex("29 00 7e 30 00 00 0b 75")
    assert "->" in describe_blinkers_state(BLINKERS_IDLE, other)


def _can(t_s: float, addr: int, data: bytes, src: int = 0):
  return (int(t_s * 1e9), [(addr, data, src)])


class TestScanRecorder:
  def test_attributes_replies_and_hits_by_echo(self):
    frames = build_lid_scan_frames(range(0x10, 0x14))
    rec = ScanRecorder(frames)
    rec.update([
      _can(0.0, BLINKERS_STATE_ADDR, BLINKERS_IDLE),
      _can(1.00, DIAG_ADDR, build_probe(0x10), src=128),
      _can(1.01, REPLY_ADDR, bytes.fromhex("40 03 7f 30 12 00 00 00")),
      _can(1.20, DIAG_ADDR, build_probe(0x11), src=128),
      _can(1.21, REPLY_ADDR, bytes.fromhex("40 03 70 11 00 00 00 00")),
      _can(1.40, DIAG_ADDR, build_probe(0x12), src=128),
      _can(1.41, REPLY_ADDR, bytes.fromhex("40 03 70 12 00 00 00 00")),
      _can(1.45, BLINKERS_STATE_ADDR, BLINKERS_HAZARD_ON),
      _can(1.60, DIAG_ADDR, build_probe(0x13), src=128),
      # a reply for another sub-address, and our own probe seen without the echo flag, are ignored
      _can(1.61, REPLY_ADDR, bytes.fromhex("e9 04 61 15 50 01 00 00")),
      _can(1.62, DIAG_ADDR, build_probe(0x13), src=0),
    ])
    rec.place_by_schedule()

    by_lid = {p.lid: p for p in rec.probes}
    assert rec.echoes == 4
    assert rec.frames_seen == 11
    assert rec.blinkers_frames == 2
    assert by_lid[0x10].verdict == "unsupported"
    assert by_lid[0x11].verdict == "positive"
    assert by_lid[0x12].verdict == "positive"
    assert by_lid[0x12].blinker_changes == ["event pulse on, HAZARD_LIGHT=1"]
    assert by_lid[0x13].verdict == "no reply"
    assert by_lid[0x13].sent_at == pytest.approx(1.60)
    assert rec.unmatched_replies == []

    summary = rec.summary()
    assert "positive: 0x11, 0x12" in summary
    assert "0x614 hits: 0x12 00 00 00 -> event pulse on, HAZARD_LIGHT=1" in summary
    report = rec.report("test")
    assert "LID 0x11 ctrl 00 00 00: 40 03 70 11 00 00 00 00  [door locks]" in report
    assert "WARNING" not in report

  def test_places_unechoed_probes_from_the_script_timing(self):
    frames = build_lid_scan_frames(range(0x20, 0x23))
    rec = ScanRecorder(frames)
    rec.update([
      _can(5.0, DIAG_ADDR, build_probe(0x20), src=128),
      _can(5.41, REPLY_ADDR, bytes.fromhex("40 03 70 22 00 00 00 00")),
    ])
    rec.place_by_schedule()
    by_lid = {p.lid: p for p in rec.probes}
    assert by_lid[0x21].sent_at == pytest.approx(5.0 + PROBE_GAP_MS / 1000)
    assert by_lid[0x22].sent_at == pytest.approx(5.0 + 2 * PROBE_GAP_MS / 1000)
    # the reply at 5.41 lands after the third probe's scheduled time, so it belongs to LID 0x22
    assert by_lid[0x22].verdict == "no reply", "attribution only happens as frames arrive"

  def test_no_echoes_at_all_anchors_on_the_first_reply(self):
    frames = build_lid_scan_frames(range(0x30, 0x32))
    rec = ScanRecorder(frames)
    rec.update([_can(2.0, REPLY_ADDR, bytes.fromhex("40 03 7f 30 12 00 00 00"))])
    rec.place_by_schedule()
    assert rec.echoes == 0
    assert rec.probes[0].sent_at == pytest.approx(2.0)
    assert rec.probes[1].sent_at == pytest.approx(2.0 + PROBE_GAP_MS / 1000)
    assert len(rec.unmatched_replies) == 1
    assert "WARNING" in rec.report("test")

  def test_sweep_attributes_by_control_record(self):
    frames = build_bit_sweep_frames(0x15)
    rec = ScanRecorder(frames)
    t = 0.0
    msgs = [_can(t, BLINKERS_STATE_ADDR, BLINKERS_IDLE)]
    for f in frames:
      t += f.delay_ms / 1000
      msgs.append(_can(t, DIAG_ADDR, f.data, src=128))
      if f.data[4:7] == b"\x00\x08\x00":
        msgs.append(_can(t + 0.05, BLINKERS_STATE_ADDR, BLINKERS_HAZARD_ON))
      elif f.data[4:7] == b"\x00\x00\x00":
        msgs.append(_can(t + 0.05, BLINKERS_STATE_ADDR, BLINKERS_IDLE))
    rec.update(msgs)
    hits = [(p.control, p.blinker_changes) for p in rec.probes if p.blinker_changes]
    assert hits[0] == (b"\x00\x08\x00", ["event pulse on, HAZARD_LIGHT=1"])
    assert hits[1][0] == b"\x00\x00\x00"
    assert rec.probes[0].sent_at == pytest.approx(0.0)
    assert rec.probes[1].sent_at == pytest.approx(SWEEP_ON_MS / 1000)
    assert rec.probes[2].sent_at == pytest.approx((SWEEP_ON_MS + SWEEP_OFF_MS) / 1000)

  def test_unplayed_script_is_reported_instead_of_no_replies(self):
    rec = ScanRecorder(build_bit_sweep_frames(0x12))
    rec.update([_can(1.0, BLINKERS_STATE_ADDR, BLINKERS_IDLE)])
    rec.script_played = False
    assert rec.summary() == NOT_PLAYED_WARNING
    report = rec.report("test")
    assert "SCRIPT NOT PLAYED" in report
    assert "probes were placed from the script timing" not in report
    assert "1 CAN frames recorded" in report

  def test_on_sent_fires_once_per_echo_in_script_order(self):
    frames = build_bit_sweep_frames(0x12, controls=sweep_controls()[:2])
    rec = ScanRecorder(frames)
    seen = []
    rec.on_sent = lambda p: seen.append((p.index, p.control))
    rec.update([_can(1.0 + i * 0.5, DIAG_ADDR, f.data, src=128) for i, f in enumerate(frames)])
    assert seen == [(0, b"\x00\x01\x00"), (1, b"\x00\x00\x00"), (2, b"\x00\x02\x00"), (3, b"\x00\x00\x00")]

  def test_report_labels_known_controls(self):
    frames = build_bit_sweep_frames(0x12, controls=[b"\x00\x02\x00", b"\x00\x80\x00"])
    rec = ScanRecorder(frames)
    msgs = []
    for i, f in enumerate(frames):
      msgs.append(_can(1.0 + i, DIAG_ADDR, f.data, src=128))
      msgs.append(_can(1.1 + i, REPLY_ADDR, bytes.fromhex("40 02 70 12 00 00 00 00")))
    rec.update(msgs)
    report = rec.report("test")
    assert "LID 0x12 ctrl 00 02 00: 40 02 70 12 00 00 00 00  [relay under the dash, load not yet identified]" in report
    assert "LID 0x12 ctrl 00 80 00: 40 02 70 12 00 00 00 00  [cabin light on]" in report
    assert "LID 0x12 ctrl 00 00 00: 40 02 70 12 00 00 00 00  [cabin light, dash relay]" in report


class TestDirectAddressing:
  """An ECU with its own address, like the meter at 0x7C0, gets plain ISO-TP frames."""

  def test_probe_is_a_single_frame_without_sub_address(self):
    assert build_probe(0x29, sub_addr=None) == bytes([0x05, 0x30, 0x29, 0x00, 0x00, 0x00, 0x00, 0x00])
    assert probe_lid(build_probe(0x29, sub_addr=None), None) == 0x29
    assert probe_lid(build_probe(0x29, sub_addr=None), 0x40) is None
    assert probe_body(bytes.fromhex("03 7f 30 11 00 00 00 00"), None) == bytes.fromhex("7f 30 11")

  def test_replies_classify_with_service_not_supported_kept_apart(self):
    assert classify_reply(bytes.fromhex("03 7f 30 11 00 00 00 00"), None) == ("noservice", "NRC 0x11 serviceNotSupported")
    assert classify_reply(bytes.fromhex("03 7f 30 12 00 00 00 00"), None)[0] == "unsupported"
    assert classify_reply(bytes.fromhex("02 70 29 00 00 00 00 00"), None)[0] == "positive"
    assert classify_reply(bytes.fromhex("40 03 7f 30 11 00 00 00"), 0x40)[0] == "noservice"

  def test_scan_frames_go_to_the_direct_address(self):
    frames = build_lid_scan_frames(range(3), sub_addr=None, addr=0x7C0)
    assert all(f.addr == 0x7C0 for f in frames)
    assert [f.data[2] for f in frames] == [0, 1, 2]

  def test_recorder_matches_echo_and_reply_on_the_direct_addresses(self):
    frames = build_lid_scan_frames(range(0x10, 0x12), sub_addr=None, addr=0x7C0)
    rec = ScanRecorder(frames, None, tx_addr=0x7C0, rx_addr=0x7C8)
    rec.update([
      _can(1.0, 0x7C0, frames[0].data, src=128),
      _can(1.01, 0x7C8, bytes.fromhex("03 7f 30 11 00 00 00 00")),
      _can(1.02, REPLY_ADDR, bytes.fromhex("40 03 7f 30 12 00 00 00")),   # body ECU noise, wrong address
      _can(1.2, 0x7C0, frames[1].data, src=128),
      _can(1.21, 0x7C8, bytes.fromhex("02 70 11 00 00 00 00 00")),
    ])
    assert rec.echoes == 2
    assert [p.verdict for p in rec.probes] == ["noservice", "positive"]
    assert rec.probes[1].control == b"\x00\x00\x00"
    assert "0x7C0 direct" in rec.summary()
    assert "service 0x30 not supported: 1 probes" in rec.summary()
    assert "does not implement service 0x30" in rec.report("t")
