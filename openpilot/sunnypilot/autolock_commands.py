"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Body-ECU diagnostic commands used by the auto-lock and hazard-test features, and the encoding
for pandad's OffroadCanQueue param. All commands are 8-byte payloads sent to address 0x750 on
bus 0, in Toyota's sub-addressed KWP format: [subaddr, len, service, ...].

pandad drains OffroadCanQueue one frame at a time with a small gap (see panda_safety.cc), because
the body ECU drops back-to-back diagnostic frames sent as a single burst.
"""

CMD_ADDR = 0x750
CMD_BUS = 0

# Lock / unlock
LOCK_CMD = b"\x40\x05\x30\x11\x00\x80\x00\x00"
UNLOCK_CMD = b"\x40\x05\x30\x11\x00\x40\x00\x00"

# Fold mirrors
MIRR_FOLD_R = b"\xA5\x04\x30\x21\x00\x08\x00\x00"
MIRR_FOLD_L = b"\xA6\x04\x30\x21\x00\x08\x00\x00"

# Close windows
WINDOW_CLOSE_FR = b"\x91\x04\x30\x01\x05\x20\x00\x00"
WINDOW_CLOSE_FL = b"\x90\x04\x30\x01\x05\x20\x00\x00"
WINDOW_CLOSE_RR = b"\x92\x04\x30\x01\x05\x20\x00\x00"
WINDOW_CLOSE_RL = b"\x93\x04\x30\x01\x05\x20\x00\x00"

WINDOW_CLOSE_ALL = (WINDOW_CLOSE_FR, WINDOW_CLOSE_FL, WINDOW_CLOSE_RR, WINDOW_CLOSE_RL)
MIRR_FOLD_ALL = (MIRR_FOLD_L, MIRR_FOLD_R)

# Hazard lamps: WriteDataByLocalId (0x3B) on LID 0x13 of the body ECU (sub-address 0x40).
# 0xC0 sets both turn-lamp bits (= hazard), 0x00 clears them. Reading the same LID with
# `40 02 21 13` returns `40 04 61 13 D0 <state>`, which is how these values were confirmed.
# Note this ECU rejects StartDiagSession on sub-address 0x40 but accepts the write anyway,
# and answers each write with responsePending (0x7F 0x3B 0x78) before the positive 0x7B.
HAZARD_ON = b"\x40\x06\x3B\x13\xF0\xC0\x00\x00"
HAZARD_OFF = b"\x40\x06\x3B\x13\xF0\x00\x00\x00"

# pandad paces the queue at one frame per 200 ms, so repeating each state twice gives a ~400 ms
# half-cycle. That matches the ~380 ms blink observed on a factory hazard capture; sending each
# state once would blink at 2.5 Hz, which is visibly too fast.
HAZARD_FRAMES_PER_STATE = 2


def frame_record(data: bytes, addr: int = CMD_ADDR, bus: int = CMD_BUS) -> bytes:
  """One OffroadCanQueue record: [addr_hi, addr_lo, bus, dlc, data[8]] (12 bytes)."""
  return bytes([(addr >> 8) & 0xFF, addr & 0xFF, bus, len(data)]) + data.ljust(8, b"\x00")[:8]


def build_queue(close_windows: bool, fold_mirrors: bool, lock: bool = True) -> bytes:
  """Build the pandad OffroadCanQueue: windows first, then mirrors, then lock."""
  queue = b""
  if close_windows:
    for cmd in WINDOW_CLOSE_ALL:
      queue += frame_record(cmd)
  if fold_mirrors:
    for cmd in MIRR_FOLD_ALL:
      queue += frame_record(cmd)
  if lock:
    queue += frame_record(LOCK_CMD)
  return queue


def build_hazard_queue(flashes: int = 3) -> bytes:
  """Build an OffroadCanQueue that blinks the hazards `flashes` times, ending with them off.

  The write is a state set, not a toggle, so repeating a frame is idempotent and is only used
  here to stretch each half-cycle to a visible duration.
  """
  queue = b""
  for _ in range(max(0, flashes)):
    for cmd in (HAZARD_ON, HAZARD_OFF):
      queue += frame_record(cmd) * HAZARD_FRAMES_PER_STATE
  return queue
