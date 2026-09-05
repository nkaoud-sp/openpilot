"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Unit tests for the turn-signal discovery probe's candidate encoding and success detector -- the
parts that run without a car. The CAN plumbing in turn_signal_probe.py needs a device and is not
covered here.
"""
from openpilot.sunnypilot.autolock_commands import LOCK_CMD, UNLOCK_CMD, CMD_ADDR, CMD_BUS
from openpilot.sunnypilot.turn_signal_probe_commands import (
  BODY_ECU_SUB_ADDR,
  KNOWN_VALUE_BITS,
  LID_LATTICE_OFFSET,
  LID_LATTICE_STRIDE,
  LOCK_LID,
  MOTION_LIDS,
  Candidate,
  SignalDetector,
  full_sweep,
  shortlist,
  structured_sweep,
)


def _cand(value: int, lid: int = LOCK_LID) -> Candidate:
  return Candidate(BODY_ECU_SUB_ADDR, lid, value, "t")


def test_candidate_payload_matches_known_lock_unlock():
  # The probe must speak the exact byte layout the working auto-lock frames prove.
  assert _cand(0x80).payload == LOCK_CMD
  assert _cand(0x40).payload == UNLOCK_CMD


def test_candidate_record_is_wellformed_offroad_queue_entry():
  rec = _cand(0x10).record
  assert len(rec) == 12
  assert (rec[0] << 8 | rec[1]) == CMD_ADDR   # addr_hi, addr_lo
  assert rec[2] == CMD_BUS                     # bus
  assert rec[3] == 8                           # dlc
  assert rec[4:] == _cand(0x10).payload        # data[8]


def test_shortlist_covers_prime_suspects_first_and_only_lock_lid():
  cands = shortlist()
  # The 0x7C0 layout (left=0x10, right=0x08, hazard=0x18) leads.
  assert [c.value for c in cands[:3]] == [0x10, 0x08, 0x18]
  # Shortlist stays on the known-safe lock register; it never blindly walks LIDs.
  assert all(c.lid == LOCK_LID for c in cands)
  # It doesn't re-send lock/unlock.
  assert all(c.value not in KNOWN_VALUE_BITS for c in cands)


def test_full_sweep_skips_known_bits_and_motion_lids_by_default():
  cands = list(full_sweep())
  assert all(c.lid not in MOTION_LIDS for c in cands)
  # The two known lock/unlock bits are skipped on the lock LID so the sweep can't lock the car.
  lock_lid_vals = {c.value for c in cands if c.lid == LOCK_LID}
  assert lock_lid_vals.isdisjoint(KNOWN_VALUE_BITS)


def test_full_sweep_order_is_stable():
  # A saved sweep resume index refers to a position in this list, so its length and ordering must not
  # drift. If this fails, every user's saved start index now points at a different command.
  cands = list(full_sweep())
  assert len(cands) == 2030
  assert (cands[0].lid, cands[0].value) == (0x00, 0x01)
  assert (cands[-1].lid, cands[-1].value) == (0xFF, 0x80)
  # The observed rear-sunshade LID sits where we decoded it from the reported probe number.
  assert cands[195].lid == 0x19
  assert cands[196].lid == 0x19


def test_structured_sweep_walks_only_the_stride_8_lattice():
  cands = list(structured_sweep())
  assert cands, "lattice must not be empty"
  # Every known actuating LID (windows 0x01, locks 0x11, sunshade 0x19, mirrors 0x21) is 1 mod 8.
  assert all(c.lid % LID_LATTICE_STRIDE == LID_LATTICE_OFFSET for c in cands)
  assert all(c.lid not in MOTION_LIDS for c in cands)
  # The known sunshade LID is on the lattice and therefore covered.
  assert 0x19 in {c.lid for c in cands}
  # Doesn't re-fire lock/unlock.
  assert {c.value for c in cands if c.lid == LOCK_LID}.isdisjoint(KNOWN_VALUE_BITS)


def test_structured_sweep_is_far_cheaper_than_full_sweep():
  # The whole point: same LID coverage of the lattice at roughly an eighth of the probes.
  assert len(list(structured_sweep())) * 6 < len(list(full_sweep()))


def test_structured_sweep_uses_the_body_ecu_frame_shape():
  # Locks and the sunshade both answered on sub 0x40 with len=0x05 / p1=0x00; keep that shape.
  for cand in list(structured_sweep())[:5]:
    payload = cand.payload
    assert payload[0] == BODY_ECU_SUB_ADDR
    assert payload[1] == 0x05
    assert payload[2] == 0x30
    assert payload[4] == 0x00


def test_full_sweep_can_include_motion_lids_when_asked():
  cands = list(full_sweep(skip_motion=False))
  assert any(c.lid in MOTION_LIDS for c in cands)


def test_detector_flags_left_right_and_hazard_from_idle_baseline():
  det = SignalDetector()
  det.set_baseline(turn_signals=3, hazard=0)   # 3 == "none"
  assert det.is_hit(turn_signals=1, hazard=0)  # left
  assert det.is_hit(turn_signals=2, hazard=0)  # right
  assert det.is_hit(turn_signals=3, hazard=1)  # hazard


def test_detector_ignores_steady_idle():
  det = SignalDetector()
  det.set_baseline(turn_signals=3, hazard=0)
  assert not det.is_hit(turn_signals=3, hazard=0)
  assert not det.is_hit(turn_signals=0, hazard=0)


def test_detector_needs_change_from_active_baseline():
  # If the bus was mid-flash when baseline was taken, re-seeing that same state is not a new hit.
  det = SignalDetector()
  det.set_baseline(turn_signals=1, hazard=0)
  assert not det.is_hit(turn_signals=1, hazard=0)
  assert det.is_hit(turn_signals=2, hazard=0)


class _FakeProbe:
  """Minimal stand-in for TurnSignalProbe: the candidate at hit_index reads as a hit."""
  def __init__(self, hit_index: int | None = None):
    self.hit_index = hit_index
    self.sent: list[int] = []

  def probe_one(self, cand) -> bool:
    idx = len(self.sent)
    self.sent.append(idx)
    return idx == self.hit_index


def _run(probe, candidates, **kwargs):
  from openpilot.sunnypilot.turn_signal_probe import run_probe
  statuses: list[dict] = []
  hits = run_probe(probe, candidates, report=statuses.append, **kwargs)
  return hits, statuses


def test_run_probe_start_skips_and_reports_absolute_index():
  cands = shortlist()
  probe = _FakeProbe()
  hits, statuses = _run(probe, cands, start=3)
  # Only candidates from index 3 onward are actually sent.
  assert len(probe.sent) == len(cands) - 3
  # Progress is absolute: first running report is 4/len, final done is len/len.
  running = [s for s in statuses if s["state"] == "running"]
  assert running[0]["index"] == 4
  assert running[0]["total"] == len(cands)
  assert statuses[-1]["state"] == "done"
  assert statuses[-1]["index"] == len(cands)


def test_run_probe_carries_prior_hits_across_sessions():
  cands = shortlist()
  probe = _FakeProbe(hit_index=0)  # first sent candidate (index 2) hits
  hits, statuses = _run(probe, cands, start=2, prior_hits=["earlier-session-hit"])
  assert hits[0] == "earlier-session-hit"      # prior hit is preserved
  assert len(hits) == 2                          # plus the one found this session
  assert statuses[-1]["hits"][0] == "earlier-session-hit"


def test_run_probe_abort_reports_reached_index():
  cands = shortlist()
  probe = _FakeProbe()
  hits, statuses = _run(probe, cands, should_abort=lambda: True)
  # Aborts before sending anything; the abort status carries the resume index (0 here).
  assert probe.sent == []
  assert statuses[-1]["state"] == "aborted"
  assert statuses[-1]["index"] == 0


def test_fisk_hazard_frame_matches_writeup_and_is_elm327_injectable():
  from openpilot.sunnypilot.broadcast_lighting_commands import (
    HAZARD_FLASH_ADDR, HAZARD_FLASH_ONCE, HAZARD_FLASH_TWICE, hazard_record,
  )
  # Exact bytes from Fisk's 2023 RAV4 write-up.
  assert HAZARD_FLASH_ONCE.hex() == "1980000000000020"
  assert HAZARD_FLASH_TWICE.hex() == "1980000000000040"
  # 0x623 is in the 0x600-0x6FF range elm327_tx_hook permits, at length 8.
  assert (HAZARD_FLASH_ADDR & 0x1FFFFF00) == 0x600
  rec = hazard_record()
  assert len(rec) == 12
  assert (rec[0] << 8 | rec[1]) == HAZARD_FLASH_ADDR
  assert rec[2] == 0  # bus 0
  assert rec[3] == 8  # dlc


def test_fisk_blinker_frame_direction_nibble():
  from openpilot.sunnypilot.broadcast_lighting_commands import (
    BLINKER_ADDR, BLINKER_D3_LEFT, BLINKER_D3_RIGHT, blinker_frame, blinker_record,
  )
  # ESORICS-2024 Corolla format: 29 80 00 <dir> 00 00 00 00 (no counter/checksum).
  assert blinker_frame(BLINKER_D3_LEFT).hex() == "2980001000000000"
  assert blinker_frame(BLINKER_D3_RIGHT).hex() == "2980002000000000"
  assert blinker_frame(BLINKER_D3_LEFT)[3] == 0x10
  assert blinker_frame(BLINKER_D3_RIGHT)[3] == 0x20
  assert (BLINKER_ADDR & 0x1FFFFF00) == 0x600  # also ELM327-injectable
  assert (blinker_record(BLINKER_D3_LEFT)[0] << 8 | blinker_record(BLINKER_D3_LEFT)[1]) == BLINKER_ADDR


def test_capture_lines_flags_new_ids_first():
  from openpilot.sunnypilot.turn_signal_probe import _capture_lines
  baseline = {(0, 0x620): {"aa"}, (0, 0x614): {"01", "02"}}
  changes = {
    (0, 0x614): {"09"},          # new value on a known id
    (0, 0x623): {"1980000000000020"},  # brand-new id -- the strongest candidate
  }
  lines = _capture_lines(changes, baseline)
  # Brand-new id sorts first and is tagged NEW-ID.
  assert lines[0] == "b0 0x623 1980000000000020 [NEW-ID]"
  assert any("0x614" in ln and "NEW-VAL" in ln for ln in lines)
