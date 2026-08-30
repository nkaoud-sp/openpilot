"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Turn-signal diagnostic active test for Toyota/Lexus, encoded for pandad's OffroadCanQueue
(same mechanism autolockd uses for lock/window/mirror). Reconstructed from a Techstream
capture of a right-turn active test on a 2019+ Lexus ES (TSS2):

    request 0x7C0 (Combination Meter), UDS 0x2F InputOutputControlByIdentifier, DID 0x2911

Unlike the body-ECU lock/window/mirror frames (single 8-byte frames to 0x750), this control
is a 12-byte UDS payload, so it is sent as ISO-TP: a first frame + one consecutive frame.
pandad drains these 8-byte frames one at a time (200 ms apart) via ELM327, which permits the
0x700-0x7FF diagnostic range. The ECU sends its own flow-control after the first frame; the
200 ms gap is well within the consecutive-frame timeout, so we don't need to read it back.
"""
from openpilot.sunnypilot.autolock_commands import frame_record

TS_ADDR = 0x7C0  # Combination Meter diagnostic request
TS_BUS = 0

TURN_BITS = {
  "right": 0x08,
  "left": 0x10,
  "hazard": 0x18,
}

# 12-byte active-test payloads. The right payload is exactly as captured from Techstream.
RIGHT_ON = bytes.fromhex("2F291103" + "0000000800000008")
RIGHT_ALT = bytes.fromhex("2F291103" + "0000000000000008")
SWEEP_BITS = 0x08
SWEEP_BASE_BYTE = 7
SWEEP_BYTES = range(7)
SWEEP_REPEATS_PER_BYTE = 15
SWEEP_CLEAR_DELAY_RECORDS = 5
PULSE_CLEAR_DELAY_RECORDS = 1
SEQUENCE_PATTERN = (
  ("left", 0.8, 0.4, 5),
  ("right", 0.4, 0.8, 5),
  ("hazard", 0.6, 0.6, 5),
)
QUEUE_FRAME_INTERVAL = 0.2

# UDS session / control (single frames)
EXTENDED_SESSION = bytes.fromhex("1003")    # DiagnosticSessionControl -> extended
TESTER_PRESENT = bytes.fromhex("3E00")
RETURN_CONTROL = bytes.fromhex("2F291100")  # InputOutputControl -> returnControlToECU


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


def make_payload(byte_index: int, bits: int = SWEEP_BITS, base_byte: int | None = None) -> bytes:
  """Build a `2F 29 11 03` active test with one control-state byte, plus an optional base byte."""
  state = bytearray(8)
  if base_byte is not None:
    state[base_byte] = bits
  state[byte_index] = bits
  return bytes.fromhex("2F291103") + bytes(state)


def clear_payload() -> bytes:
  """Build a zeroed `2F 29 11 03` active-test state."""
  return bytes.fromhex("2F291103") + bytes(8)


def turn_signal_payload(signal: str) -> bytes:
  """Build a verified turn signal active-test payload by signal name."""
  bits = TURN_BITS[signal]
  return make_payload(3, bits=bits, base_byte=7)


def turn_signal_off_payload(signal: str) -> bytes:
  """Build the captured-style off payload: clear byte 3 while leaving byte 7 selected."""
  bits = TURN_BITS[signal]
  state = bytearray(8)
  state[7] = bits
  return bytes.fromhex("2F291103") + bytes(state)


def build_turn_signal_queue(payload: bytes = RIGHT_ON, repeats: int = 6, session: bool = True) -> bytes:
  """
  OffroadCanQueue to flash a turn signal. Optionally opens an extended diagnostic session,
  then re-sends the active test `repeats` times to hold the lamp (Techstream re-sent ~1/s).
  At 200 ms/frame, repeats=6 spans roughly 2.5 s.
  """
  queue = _records(EXTENDED_SESSION) if session else b""
  for _ in range(repeats):
    queue += _records(payload)
  queue += _records(RETURN_CONTROL)
  return queue


def build_turn_signal_pulse_queue(signal: str, clear_delay_records: int = PULSE_CLEAR_DELAY_RECORDS,
                                  session: bool = True) -> bytes:
  """
  OffroadCanQueue to pulse a verified turn signal briefly. Sends the ON payload once,
  waits about one queue tick, then sends the captured-style OFF payload.
  """
  queue = _records(EXTENDED_SESSION) if session else b""
  queue += _records(turn_signal_payload(signal))
  for _ in range(clear_delay_records):
    queue += _records(TESTER_PRESENT)
  queue += _records(turn_signal_off_payload(signal))
  queue += _records(RETURN_CONTROL)
  return queue


def _delay_records(seconds: float) -> int:
  return max(round(seconds / QUEUE_FRAME_INTERVAL) - 2, 0)


def build_turn_signal_sequence_queue(session: bool = True) -> bytes:
  """
  OffroadCanQueue scripted sequence:
    left 0.8s on / 0.4s off x5
    right 0.4s on / 0.8s off x5
    hazard 0.6s on / 0.6s off x5
  Durations are approximate command-start intervals at pandad's 200 ms queue drain.
  """
  queue = _records(EXTENDED_SESSION) if session else b""
  for signal, on_seconds, off_seconds, repeats in SEQUENCE_PATTERN:
    for _ in range(repeats):
      queue += _records(turn_signal_payload(signal))
      for _ in range(_delay_records(on_seconds)):
        queue += _records(TESTER_PRESENT)
      queue += _records(turn_signal_off_payload(signal))
      for _ in range(_delay_records(off_seconds)):
        queue += _records(TESTER_PRESENT)
  queue += _records(RETURN_CONTROL)
  return queue


def build_turn_signal_sweep_queue(bits: int = SWEEP_BITS, base_byte: int = SWEEP_BASE_BYTE,
                                  repeats_per_byte: int = SWEEP_REPEATS_PER_BYTE,
                                  clear_delay_records: int = SWEEP_CLEAR_DELAY_RECORDS, session: bool = True) -> bytes:
  """
  OffroadCanQueue sweep for discovering which 0x2911 control byte actuates each lamp/output.
  Byte 7 is kept set because the captured right-turn command always includes it.
  Each byte is held for roughly 6 seconds at pandad's 200 ms/frame drain rate.
  A zeroed active-test state is sent about 1 second after each byte stage.
  """
  queue = _records(EXTENDED_SESSION) if session else b""
  for byte_index in SWEEP_BYTES:
    payload = make_payload(byte_index, bits, base_byte=base_byte)
    for _ in range(repeats_per_byte):
      queue += _records(payload)
    for _ in range(clear_delay_records):
      queue += _records(TESTER_PRESENT)
    queue += _records(clear_payload())
    queue += _records(RETURN_CONTROL)
  queue += _records(RETURN_CONTROL)
  return queue
