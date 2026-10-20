"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Toyota body-ECU diagnostic commands used by Auto Door Lock, plus the record
encoding consumed by pandad's OffroadCanQueue.
"""

CMD_ADDR = 0x750
CMD_BUS = 0

LOCK_CMD = b"\x40\x05\x30\x11\x00\x80\x00\x00"
UNLOCK_CMD = b"\x40\x05\x30\x11\x00\x40\x00\x00"

MIRR_FOLD_R = b"\xA5\x04\x30\x21\x00\x08\x00\x00"
MIRR_FOLD_L = b"\xA6\x04\x30\x21\x00\x08\x00\x00"

WINDOW_CLOSE_FR = b"\x91\x04\x30\x01\x05\x20\x00\x00"
WINDOW_CLOSE_FL = b"\x90\x04\x30\x01\x05\x20\x00\x00"
WINDOW_CLOSE_RR = b"\x92\x04\x30\x01\x05\x20\x00\x00"
WINDOW_CLOSE_RL = b"\x93\x04\x30\x01\x05\x20\x00\x00"

WINDOW_CLOSE_ALL = (WINDOW_CLOSE_FR, WINDOW_CLOSE_FL, WINDOW_CLOSE_RR, WINDOW_CLOSE_RL)
MIRR_FOLD_ALL = (MIRR_FOLD_L, MIRR_FOLD_R)


def frame_record(data: bytes, addr: int = CMD_ADDR, bus: int = CMD_BUS) -> bytes:
  """Encode one [addr_hi, addr_lo, bus, dlc, data[8]] queue record."""
  return bytes([(addr >> 8) & 0xFF, addr & 0xFF, bus, len(data)]) + data.ljust(8, b"\x00")[:8]


def build_queue(close_windows: bool, fold_mirrors: bool, lock: bool = True) -> bytes:
  """Build an offroad command queue: windows, mirrors, then door lock."""
  queue = b""
  if close_windows:
    for command in WINDOW_CLOSE_ALL:
      queue += frame_record(command)
  if fold_mirrors:
    for command in MIRR_FOLD_ALL:
      queue += frame_record(command)
  if lock:
    queue += frame_record(LOCK_CMD)
  return queue
