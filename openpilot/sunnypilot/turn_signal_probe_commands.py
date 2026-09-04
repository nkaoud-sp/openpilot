"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Candidate body-ECU commands for the turn signal discovery probe, and the detector that decides
whether a candidate actually actuated the exterior signals.

Everything here is diagnostic InputOutputControlByLocalIdentifier (KWP2000 service 0x30) to the
main body ECU at 0x750, exactly like the auto-lock lock/unlock frames in autolock_commands.py.
The lock/unlock frames prove the register layout:

    LOCK   = 40 05 30 11 00 80 00 00   (sub-addr 0x40, len 5, svc 0x30, LID 0x11, param 0x00, val 0x80)
    UNLOCK = 40 05 30 11 00 40 00 00   (same, val 0x40)

Lock and unlock are the same local identifier (0x11) and differ only in a single bit of the value
byte, so that byte is a bitmask of body outputs. Bits 0x80 (lock) and 0x40 (unlock) are known; the
other six are unclaimed outputs on a register the BCM already actuates. The turn-signal lamps are a
plausible tenant. This module enumerates candidates over that space so the probe can find them.
"""
from collections.abc import Iterator
from dataclasses import dataclass

from openpilot.sunnypilot.autolock_commands import CMD_ADDR, CMD_BUS, frame_record

# KWP2000 InputOutputControlByLocalIdentifier
IOCTL_SERVICE = 0x30
BODY_ECU_SUB_ADDR = 0x40  # main body ECU, same sub-address the lock/unlock commands use

# Local identifier that lock/unlock live on. The value byte is a bitmask of body outputs.
LOCK_LID = 0x11

# Known bits on LOCK_LID, so the probe can report/skip them instead of re-locking the car.
KNOWN_VALUE_BITS = {0x80: "lock", 0x40: "unlock"}

# Prime suspects, tried first. The onroad 0x7C0 turn-signal path in opendbc uses left=0x10, right=0x08,
# hazard=0x18 (=0x10|0x08). Toyota reuses bit assignments across interfaces, so the same layout on the
# LOCK_LID value byte is the single most likely hit.
PRIME_SUSPECTS: tuple[tuple[int, str], ...] = (
  (0x10, "left?"),
  (0x08, "right?"),
  (0x18, "hazard?"),
)

# The remaining unclaimed single bits on LOCK_LID, tried after the prime suspects.
REMAINING_BITS: tuple[int, ...] = (0x20, 0x04, 0x02, 0x01)

# LIDs that drive physical motion (windows, mirrors). Excluded from the blind sweep by default so an
# unattended sweep can't roll a window onto something. Overridable on the command line.
MOTION_LIDS = {0x01, 0x21}

# Every local identifier observed to actuate anything on this ECU family is congruent to 1 mod 8:
#   windows 0x01, locks 0x11, rear sunshade 0x19, mirrors 0x21.
# So the IOControl table looks like a stride-8 lattice rather than a dense range. structured_sweep()
# walks only that lattice, which covers the whole 0x00-0xFF LID space in ~240 probes instead of ~2030.
LID_LATTICE_STRIDE = 8
LID_LATTICE_OFFSET = 1

VALUE_BITS: tuple[int, ...] = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80)


@dataclass(frozen=True)
class Candidate:
  """One body-ECU command to try, with a human label for the log."""
  sub_addr: int
  lid: int
  value: int
  label: str
  pre: int = 0x00
  length: int = 0x05

  @property
  def payload(self) -> bytes:
    # [sub_addr, length, service, lid, pre, value, pad, pad] -- matches LOCK/UNLOCK byte-for-byte.
    return bytes([self.sub_addr, self.length, IOCTL_SERVICE, self.lid, self.pre, self.value, 0x00, 0x00])

  @property
  def record(self) -> bytes:
    """12-byte OffroadCanQueue record pandad drains offroad via ELM327."""
    return frame_record(self.payload, addr=CMD_ADDR, bus=CMD_BUS)

  def __str__(self) -> str:
    return f"{self.label:<8} sub=0x{self.sub_addr:02X} lid=0x{self.lid:02X} val=0x{self.value:02X} [{self.payload.hex()}]"


def shortlist() -> list[Candidate]:
  """The safe first pass: only the LOCK_LID value-bit register, prime suspects first."""
  out = [Candidate(BODY_ECU_SUB_ADDR, LOCK_LID, val, label) for val, label in PRIME_SUSPECTS]
  out += [Candidate(BODY_ECU_SUB_ADDR, LOCK_LID, val, "bit?") for val in REMAINING_BITS]
  return out


def full_sweep(sub_addr: int = BODY_ECU_SUB_ADDR, skip_motion: bool = True,
               skip_known_bits: bool = True) -> Iterator[Candidate]:
  """Every LID x single-bit value on one sub-address. Large and blind -- gate behind explicit consent.

  Skips the LOCK_LID bits already covered by the shortlist, the two known lock/unlock bits, and (by
  default) the window/mirror LIDs that move physical parts.
  """
  for lid in range(0x100):
    if skip_motion and lid in MOTION_LIDS:
      continue
    for value in VALUE_BITS:
      if lid == LOCK_LID and skip_known_bits and value in KNOWN_VALUE_BITS:
        continue
      yield Candidate(sub_addr, lid, value, "sweep")


def structured_sweep(sub_addr: int = BODY_ECU_SUB_ADDR, skip_motion: bool = True,
                     skip_known_bits: bool = True) -> Iterator[Candidate]:
  """Only the LIDs on the stride-8 lattice (lid % 8 == 1), where every known function lives.

  Same frame shape as the lock/sunshade commands, which both answer on sub 0x40. Covers the entire
  LID space at ~1/8th the cost of full_sweep, so it is the cheap way to test the upper half.
  """
  for lid in range(LID_LATTICE_OFFSET, 0x100, LID_LATTICE_STRIDE):
    if skip_motion and lid in MOTION_LIDS:
      continue
    for value in VALUE_BITS:
      if lid == LOCK_LID and skip_known_bits and value in KNOWN_VALUE_BITS:
        continue
      yield Candidate(sub_addr, lid, value, "lattice")


class SignalDetector:
  """Decides whether a candidate lit the exterior signals, from BLINKERS_STATE (0x614) samples.

  Baseline is captured with the signals idle. A hit is any later sample where the turn signal reads
  left/right or the hazard bit is set -- i.e. the lamps left their idle state during the observe
  window. Works whether the command latches a lamp steady or kicks off the flasher.
  """
  def __init__(self) -> None:
    self.baseline_turn: int | None = None
    self.baseline_hazard: int | None = None

  def set_baseline(self, turn_signals: int, hazard: int) -> None:
    self.baseline_turn = turn_signals
    self.baseline_hazard = hazard

  def is_hit(self, turn_signals: int, hazard: int) -> bool:
    # TURN_SIGNALS: 1=left, 2=right, 3=none (per BLINKERS_STATE VAL_). Idle is "none" or 0.
    turn_active = turn_signals in (1, 2)
    hazard_active = bool(hazard)
    # Also count a departure from a non-idle baseline, in case the bus was mid-flash when we started.
    changed = (turn_signals != self.baseline_turn) or (hazard != self.baseline_hazard)
    return (turn_active or hazard_active) and changed
