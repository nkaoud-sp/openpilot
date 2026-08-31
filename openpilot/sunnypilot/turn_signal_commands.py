"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Turn-signal diagnostic active test for Toyota/Lexus, encoded for pandad's OffroadCanQueue
(same mechanism autolockd uses for lock/window/mirror). Reconstructed from Techstream captures
of the right- and left-turn active tests on a 2019+ Lexus ES (TSS2):

    request 0x7C0 (Combination Meter), UDS 0x2F InputOutputControlByIdentifier, DID 0x2911

The 12-byte control payload is `2F 29 11 03` + 8 control-state bytes. Comparing the two
captures, the lamp is selected by a single bit that appears in both state byte 3 (the blink
phase, toggled on/off) and state byte 7 (the selection, held for the whole test):

    right  -> 0x08     left -> 0x10     hazard -> 0x18 (both bits; predicted, unverified)

Unlike the body-ECU lock/window/mirror frames (single 8-byte frames to 0x750), this control
is a 12-byte payload, so it is sent as ISO-TP: a first frame + one consecutive frame. pandad
drains these 8-byte frames one at a time (200 ms apart) via ELM327, which permits the
0x700-0x7FF diagnostic range. The ECU sends its own flow-control after the first frame; the
200 ms gap is well within the consecutive-frame timeout, so we don't need to read it back.
"""
from openpilot.sunnypilot.autolock_commands import frame_record

TS_ADDR = 0x7C0  # Combination Meter diagnostic request
TS_BUS = 0

# Lamp-select bits (state byte 3 = blink phase, state byte 7 = selection).
TURN_BITS = {
  "right": 0x08,   # verified (capture)
  "left": 0x10,    # verified (capture)
  "hazard": 0x18,  # predicted: right | left
}

# UDS session / control (single frames)
EXTENDED_SESSION = bytes.fromhex("1003")    # DiagnosticSessionControl -> extended
RETURN_CONTROL = bytes.fromhex("2F291100")  # InputOutputControl -> returnControlToECU


def active_test_payload(bit: int, on: bool) -> bytes:
  """Build a `2F 29 11 03` active-test payload for a lamp bit, in the on or off blink phase."""
  state = bytearray(8)
  state[3] = bit if on else 0x00  # blink phase (message byte 7)
  state[7] = bit                  # lamp selection, held for the whole test (message byte 11)
  return bytes.fromhex("2F291103") + bytes(state)


def isotp_frames(payload: bytes) -> list[bytes]:
  """Split a UDS payload into 8-byte ISO-TP CAN frames (single, or first frame + consecutives)."""
  n = len(payload)
  if n <= 7:
    return [(bytes([n]) + payload).ljust(8, b"\x00")]
  frames = [bytes([0x10 | (n >> 8), n & 0xFF]) + payload[:6]]  # first frame (8 bytes)
  rest, idx = payload[6:], 1
  while rest:
    frames.append((bytes([0x20 | (idx & 0xF)]) + rest[:7]).ljust(8, b"\x00"))  # consecutive frame
    rest, idx = rest[7:], idx + 1
  return frames


def _records(payload: bytes) -> bytes:
  return b"".join(frame_record(f, addr=TS_ADDR, bus=TS_BUS) for f in isotp_frames(payload))


def build_turn_signal_queue(side: str = "right", repeats: int = 5, session: bool = True,
                            payload: bytes | None = None) -> bytes:
  """
  OffroadCanQueue to flash a turn signal. Optionally opens an extended diagnostic session, then
  alternates the active test on/off `repeats` times to blink the lamp the way Techstream does
  (each phase is 2 frames = ~400 ms at pandad's 200 ms spacing). `payload` overrides `side` with
  a raw active-test payload (held, no blink).
  """
  if payload is not None:
    phases = [payload]
  else:
    bit = TURN_BITS[side]
    phases = [active_test_payload(bit, True), active_test_payload(bit, False)]

  queue = _records(EXTENDED_SESSION) if session else b""
  for _ in range(repeats):
    for phase in phases:
      queue += _records(phase)
  queue += _records(RETURN_CONTROL)
  return queue
