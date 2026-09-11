"""
Body-ECU diagnostic commands used by the auto-lock feature, plus the encoding
for pandad's OffroadCanQueue param.
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
  """One OffroadCanQueue record: [addr_hi, addr_lo, bus, dlc, data[8]]."""
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
