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
TESTER_PRESENT = bytes.fromhex("3E00")      # keep the session alive between refreshes
RETURN_CONTROL = bytes.fromhex("2F291100")  # InputOutputControl -> returnControlToECU

# pandad nominally drains OffroadCanQueue one frame per 200 ms (panda_safety.cc), but the
# measured on-car rate is ~300 ms/frame (a 27-frame queue took ~8.5 s), so timing quantizes to
# ~300 ms. The lamp is energised while any active test with the select bit is being sent, so to
# turn it OFF we return control to the ECU (2F 29 11 00); clearing the "blink phase" byte alone
# does not extinguish it.
FRAME_S = 0.3

# Default test: three refreshed pulses (2 s, 1 s, 0.5 s), then a fourth single-shot pulse that
# sends ONE on message and holds without refreshing to see whether the ECU latches it.
DEFAULT_ON_DURATIONS = (2.0, 1.0, 0.5)
DEFAULT_GAP_S = 1.0        # off time between pulses
DEFAULT_SINGLE_SHOT_S = 0.8  # single-message pulse hold (0 disables)


def active_test_payload(bit: int, on: bool = True) -> bytes:
  """Build a `2F 29 11 03` active-test payload energising a lamp bit (message bytes 7 and 11)."""
  state = bytearray(8)
  state[3] = bit if on else 0x00  # message byte 7 (observed redundant to selection)
  state[7] = bit                  # message byte 11: lamp selection = what energises the lamp
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


def _hold_records(payload: bytes, duration: float) -> bytes:
  """
  Hold one active-test state for ~duration by re-sending it (each ISO-TP message is 2 frames =
  400 ms), padding the sub-400 ms remainder with tester-present frames (200 ms each) for finer
  timing. Refreshing keeps the commanded output alive without relying on the ECU to latch it.
  """
  total_frames = max(1, round(duration / FRAME_S))
  recs, used = b"", 0
  while total_frames - used >= 2:
    recs += _records(payload)
    used += 2
  while used < total_frames:
    recs += _records(TESTER_PRESENT)
    used += 1
  return recs


def _off_records(duration: float) -> bytes:
  """Turn the lamp off (return control to the ECU), then hold off with tester-present."""
  recs = _records(RETURN_CONTROL)  # single frame -> extinguishes the lamp
  for _ in range(max(0, round(duration / FRAME_S) - 1)):
    recs += _records(TESTER_PRESENT)
  return recs


def _single_shot_records(payload: bytes, hold: float) -> bytes:
  """Send exactly one on message, then hold with tester-present (no refresh) for `hold`."""
  recs = _records(payload)  # one FF + CF, delivered once
  for _ in range(max(1, round(hold / FRAME_S))):
    recs += _records(TESTER_PRESENT)
  return recs


def build_turn_signal_pulses(side: str = "right", on_durations=DEFAULT_ON_DURATIONS,
                             gap: float = DEFAULT_GAP_S, session: bool = True,
                             single_shot: float = DEFAULT_SINGLE_SHOT_S) -> bytes:
  """
  OffroadCanQueue that pulses a turn signal: for each of `on_durations`, energise the lamp
  (re-sending to hold it) then return control to turn it off, held for `gap`. If `single_shot` > 0,
  append a final pulse that sends ONE on message and holds it that long WITHOUT refreshing, to test
  whether the ECU latches a single command. Timing quantizes to ~300 ms.
  """
  on = active_test_payload(TURN_BITS[side])
  queue = _records(EXTENDED_SESSION) if session else b""
  for dur in on_durations:
    queue += _hold_records(on, dur)  # lamp on (refreshed)
    queue += _off_records(gap)       # lamp off
  if single_shot > 0:
    queue += _single_shot_records(on, single_shot)  # one message, no refresh
    queue += _off_records(gap)
  return queue


def build_turn_signal_queue(side: str = "right", session: bool = True, payload: bytes | None = None) -> bytes:
  """Simple single-state hold (used for raw --payload); pulse behaviour lives in build_turn_signal_pulses."""
  state = payload if payload is not None else active_test_payload(TURN_BITS[side], True)
  queue = _records(EXTENDED_SESSION) if session else b""
  queue += _hold_records(state, 2.0)
  queue += _records(RETURN_CONTROL)
  return queue
