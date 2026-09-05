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
  - Mallojula et al., "Companion Apps or Backdoors?", ESORICS 2024, Table 2: on a 2016 Toyota
    Corolla, injecting "0x614 29 80 00 10 00 00 00" via an ELM327 OBD dongle turns on the left
    signal. This is the validated turn-signal command (data[3] = 0x10 left, 0x20 right).
  - Austin Fisk's 2023 RAV4 write-up: "0x623 19 80 00 00 00 00 00 <count>" flashes the hazards
    (count 0x20 = 1 flash, 0x40 = 2). RAV4-specific; kept as a fallback candidate.

0x614 is also the address the body ECU broadcasts BLINKERS_STATE on, so injection competes with the
real transmitter -- send it as a sustained burst, not a single frame.
"""
from openpilot.sunnypilot.autolock_commands import frame_record

CMD_BUS = 0

# --- 0x623: the demonstrated hazard-flash command --------------------------------------------------
HAZARD_FLASH_ADDR = 0x623
HAZARD_FLASH_ONCE = bytes([0x19, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20])
HAZARD_FLASH_TWICE = bytes([0x19, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40])

# --- 0x614: turn-signal command, direction in data[3] (ESORICS-2024 validated on a Corolla) -------
BLINKER_ADDR = 0x614
BLINKER_D3_IDLE = 0x30
BLINKER_D3_LEFT = 0x10
BLINKER_D3_RIGHT = 0x20
BLINKER_D3_HAZARD = 0x38


def blinker_frame(data3: int, data1: int = 0x80) -> bytes:
  """A 0x614 turn-signal command, direction in data[3].

  Format validated on a 2016 Toyota Corolla via an ELM327 OBD dongle by Mallojula et al., ESORICS
  2024 (Table 2): "614 29 80 00 10 00 00 00" turns on the left signal. data[2] is 0x00 and the tail
  is all zeros -- no counter/checksum needed -- unlike the body ECU's own status broadcast, which
  carries 0x62 in data[2] and a rolling counter in the last bytes.
  """
  return bytes([0x29, data1, 0x00, data3, 0x00, 0x00, 0x00, 0x00])


def hazard_record(twice: bool = False) -> bytes:
  data = HAZARD_FLASH_TWICE if twice else HAZARD_FLASH_ONCE
  return frame_record(data, addr=HAZARD_FLASH_ADDR, bus=CMD_BUS)


def blinker_record(data3: int) -> bytes:
  return frame_record(blinker_frame(data3), addr=BLINKER_ADDR, bus=CMD_BUS)
