"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Operational broadcast lighting frames for the RAV4-generation body ECU, from Austin Fisk's
2023 RAV4 CAN reverse-engineering write-up. Unlike the 0x750 diagnostic InputOutputControl the
auto-lock and the turn-signal probe use, these are ordinary broadcast frames on bus 0 -- the same
kind of message the body ECU normally exchanges, so they are not speed-gated.

They are injectable through the existing offroad OffroadCanQueue -> pandad -> ELM327 path because
elm327_tx_hook allows any 11-bit address in 0x600-0x6FF (and 0x700-0x7FF) at length 8, and every
address here is in 0x6xx.

Fisk's decode:
  0x623  "flash hazards"  19 80 00 00 00 00 00 <count>   count 0x20 = 1 flash, 0x40 = 2 flashes
  0x614  BLINKERS_STATE    29 <d1> 62 <d3> 00 01 72 5C    d1 0x80 to initiate; d3 0x30 idle,
                                                          0x10 left, 0x20 right, 0x38 hazard

0x623 is the demonstrated, injectable command. 0x614 is the state the body ECU itself broadcasts,
so injecting it competes with the real transmitter and its last two bytes look like a counter and
checksum -- treat the 0x614 frames here as experimental, the 0x623 hazard flash as known-good.
"""
from openpilot.sunnypilot.autolock_commands import frame_record

CMD_BUS = 0

# --- 0x623: the demonstrated hazard-flash command --------------------------------------------------
HAZARD_FLASH_ADDR = 0x623
HAZARD_FLASH_ONCE = bytes([0x19, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20])
HAZARD_FLASH_TWICE = bytes([0x19, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40])

# --- 0x614: BLINKERS_STATE, direction in data[3] (experimental to inject) --------------------------
BLINKER_ADDR = 0x614
BLINKER_D3_IDLE = 0x30
BLINKER_D3_LEFT = 0x10
BLINKER_D3_RIGHT = 0x20
BLINKER_D3_HAZARD = 0x38


def blinker_frame(data3: int, data1: int = 0x80) -> bytes:
  """A 0x614 frame with Fisk's idle template and the given direction nibble in data[3]."""
  return bytes([0x29, data1, 0x62, data3, 0x00, 0x01, 0x72, 0x5C])


def hazard_record(twice: bool = False) -> bytes:
  data = HAZARD_FLASH_TWICE if twice else HAZARD_FLASH_ONCE
  return frame_record(data, addr=HAZARD_FLASH_ADDR, bus=CMD_BUS)


def blinker_record(data3: int) -> bytes:
  return frame_record(blinker_frame(data3), addr=BLINKER_ADDR, bus=CMD_BUS)
