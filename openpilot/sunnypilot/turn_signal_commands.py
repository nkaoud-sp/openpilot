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
DEFAULT_SESSION = bytes.fromhex("1001")     # DiagnosticSessionControl -> default (releases IO control)
TESTER_PRESENT = bytes.fromhex("3E00")      # keep the session alive between refreshes
RETURN_CONTROL = bytes.fromhex("2F291100")  # InputOutputControl -> returnControlToECU

# pandad nominally drains OffroadCanQueue one frame per 200 ms (panda_safety.cc), but the
# measured on-car rate is ~300 ms/frame (a 27-frame queue took ~8.5 s), so timing quantizes to
# ~300 ms. The lamp stays energised while the active test / extended session is held; on-car,
# returnControlToECU alone did NOT extinguish it, so between pulses we drop to the default session
# (which releases the IO control) and re-enter the extended session before the next pulse.
FRAME_S = 0.3

# Default test: four 1-second flashes with an off gap between each.
DEFAULT_ON_DURATIONS = (1.0, 1.0, 1.0, 1.0)
DEFAULT_GAP_S = 1.5          # off time between pulses
DEFAULT_SINGLE_SHOT_S = 0.0  # extra leading single-message pulse (0 disables)


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
  """Release the active test (return control + drop to default session), then hold off."""
  recs = _records(RETURN_CONTROL) + _records(DEFAULT_SESSION)
  for _ in range(max(0, round(duration / FRAME_S) - 2)):
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
  OffroadCanQueue that pulses a turn signal. If `single_shot` > 0, the first pulse sends ONE on
  message and holds it that long WITHOUT refreshing, to test whether the ECU latches a single
  command. Then, for each of `on_durations`, energise the lamp (re-sending to hold it). Between
  pulses the lamp is turned off by releasing the IO control and dropping to the default session,
  so each pulse re-enters the extended session first. Timing quantizes to ~300 ms.
  """
  on = active_test_payload(TURN_BITS[side])
  holds = ([("single", single_shot)] if single_shot > 0 else []) + [("hold", d) for d in on_durations]

  queue = b""
  for kind, dur in holds:
    if session:
      queue += _records(EXTENDED_SESSION)  # (re)enter extended session before energising
    queue += _single_shot_records(on, dur) if kind == "single" else _hold_records(on, dur)
    queue += _off_records(gap)             # release IO control + default session -> lamp off
  return queue


def pulse_schedule(side: str = "right", on_durations=DEFAULT_ON_DURATIONS, gap: float = DEFAULT_GAP_S,
                   single_shot: float = DEFAULT_SINGLE_SHOT_S, settle: float = 0.4):
  """
  Real-time pulse schedule as a list of (queue_bytes, sleep_s) steps. An executor writes each
  queue_bytes to OffroadCanQueue then sleeps sleep_s, so the on-time is driven by a real clock
  (consistent) rather than a frame count (jittery). One on message latches the lamp, so each pulse
  energises once and holds via the sleep; off is returnControl + default session.
  """
  on = _records(active_test_payload(TURN_BITS[side]))
  enter = _records(EXTENDED_SESSION)
  off = _records(RETURN_CONTROL) + _records(DEFAULT_SESSION)
  durs = ([single_shot] if single_shot > 0 else []) + list(on_durations)
  steps = []
  for dur in durs:
    steps.append((enter, settle))  # enter extended session, let it transmit
    steps.append((on, dur))        # energise (latched) and hold for a real dur
    steps.append((off, gap))       # release + default session -> off, hold the gap
  return steps


def run_pulse_sequence(side: str = "right", on_durations=DEFAULT_ON_DURATIONS, gap: float = DEFAULT_GAP_S,
                       single_shot: float = DEFAULT_SINGLE_SHOT_S) -> None:
  """Execute pulse_schedule in real time, writing each step to OffroadCanQueue (pandad drains it).
  Only touches Params, so it is safe to run alongside pandad (no panda claim)."""
  import time
  from openpilot.common.params import Params
  params = Params()
  for queue, sleep_s in pulse_schedule(side, on_durations, gap, single_shot):
    params.put("OffroadCanQueue", queue)
    time.sleep(sleep_s)


def build_turn_signal_queue(side: str = "right", session: bool = True, payload: bytes | None = None) -> bytes:
  """Simple single-state hold (used for raw --payload); pulse behaviour lives in build_turn_signal_pulses."""
  state = payload if payload is not None else active_test_payload(TURN_BITS[side], True)
  queue = _records(EXTENDED_SESSION) if session else b""
  queue += _hold_records(state, 2.0)
  queue += _records(RETURN_CONTROL)
  return queue
