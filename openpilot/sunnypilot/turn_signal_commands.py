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

# 12-byte active-test payloads, exactly as captured (byte 7 differs; both keep the 0x08 in byte 11).
RIGHT_ON = bytes.fromhex("2F291103" + "0000000800000008")
RIGHT_ALT = bytes.fromhex("2F291103" + "0000000000000008")

# UDS session / control (single frames)
EXTENDED_SESSION = bytes.fromhex("1003")    # DiagnosticSessionControl -> extended
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
