"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Operational broadcast lighting frames for Toyota/Lexus body ECUs. Unlike the 0x750 diagnostic
InputOutputControl the auto-lock and the turn-signal probe use, these are ordinary broadcast frames
on bus 0 -- the same kind of message the body ECU normally exchanges, so they are not speed-gated.

They are injectable through the existing offroad OffroadCanQueue -> pandad -> ELM327 path because
elm327_tx_hook allows any 11-bit address in 0x600-0x6FF (and 0x700-0x7FF) at length 8, and every
address here is in 0x6xx.

Sources:
  - Live capture from the target 2020 Lexus ES350: 0x614 = 29 80 8a <dir> 00 00 02 ce carries the
    turn-signal/hazard state, and 0x615 = 2a 00 6a 80 00 00 02 ce appears while a turn signal is
    active. These are this car's own bytes and are the primary frames here.
  - Mallojula et al., "Companion Apps or Backdoors?", ESORICS 2024, Table 2: on a 2016 Toyota
    Corolla, injecting 0x614 with data[3]=0x10 turns on the left signal -- corroborates that the
    direction lives in data[3] (0x10 left, 0x20 right, 0x38 hazard).
  - Austin Fisk's 2023 RAV4 write-up: "0x623 19 80 00 00 00 00 00 <count>" flashes the hazards.
    RAV4-specific; kept as a documented fallback constant.

0x614 is also the address the body ECU broadcasts BLINKERS_STATE on, so injection competes with the
real transmitter -- send it as a sustained burst, not a single frame.
"""
from openpilot.sunnypilot.autolock_commands import frame_record

CMD_BUS = 0

# --- 0x623: the demonstrated hazard-flash command --------------------------------------------------
HAZARD_FLASH_ADDR = 0x623
HAZARD_FLASH_ONCE = bytes([0x19, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20])
HAZARD_FLASH_TWICE = bytes([0x19, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40])

# --- 0x614: turn-signal / hazard, direction in data[3], captured live from a 2020 Lexus ES350 -----
# ES350 capture: 0x614 = 29 80 8a <dir> 00 00 02 ce. byte[2]=0x8a and the 02 ce tail are the car's
# own values -- constant across all four directions (so 02 ce is neither a checksum of the payload
# nor a fast counter). The Corolla's 29 80 00 <dir> 00 00 00 00 did not work here; these are this
# car's real bytes. dir: 0x30 idle, 0x10 left, 0x20 right, 0x38 hazard.
BLINKER_ADDR = 0x614
BLINKER_D3_IDLE = 0x30
BLINKER_D3_LEFT = 0x10
BLINKER_D3_RIGHT = 0x20
BLINKER_D3_HAZARD = 0x38

# 0x615 appears only while a turn signal is active (both directions, same payload); not on hazard.
# Sent alongside 0x614 for turn signals since the two ride the bus together.
TURN_ACTIVE_ADDR = 0x615
TURN_ACTIVE_FRAME = bytes([0x2A, 0x00, 0x6A, 0x80, 0x00, 0x00, 0x02, 0xCE])


def blinker_frame(data3: int, data1: int = 0x80) -> bytes:
  """The ES350's 0x614 frame with the given direction nibble in data[3]."""
  return bytes([0x29, data1, 0x8A, data3, 0x00, 0x00, 0x02, 0xCE])


def signal_burst_record(data3: int) -> bytes:
  """One burst unit: the directional 0x614 frame followed by the 0x615 companion, as seen together."""
  return (frame_record(blinker_frame(data3), addr=BLINKER_ADDR, bus=CMD_BUS) +
          frame_record(TURN_ACTIVE_FRAME, addr=TURN_ACTIVE_ADDR, bus=CMD_BUS))


def hazard_record(twice: bool = False) -> bytes:
  data = HAZARD_FLASH_TWICE if twice else HAZARD_FLASH_ONCE
  return frame_record(data, addr=HAZARD_FLASH_ADDR, bus=CMD_BUS)


def blinker_record(data3: int) -> bytes:
  return frame_record(blinker_frame(data3), addr=BLINKER_ADDR, bus=CMD_BUS)
