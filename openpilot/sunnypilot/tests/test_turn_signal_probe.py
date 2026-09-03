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
  LOCK_LID,
  MOTION_LIDS,
  Candidate,
  SignalDetector,
  full_sweep,
  shortlist,
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
