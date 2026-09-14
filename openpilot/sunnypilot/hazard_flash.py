"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Hazard lamp flash test: a replay of the one thing in an OBD capture of a hazard-flashing dongle
that actually blinks.

The lamps are driven with UDS InputOutputControlByIdentifier (service 0x2F) on data identifier
0x2911, sent to diagnostic address 0x7C0 (the ECU answers on 0x7C8; opendbc's Toyota ECU list
names 0x7C0 the Combination Meter). The request is 12 bytes, so it goes out as an ISO-TP first
frame plus one consecutive frame, exactly as captured:

  7C0  10 0C 2F 29 11 03 00 00     first frame, 12-byte request
  7C8  30 01 00 00 00 00 00 00     flow control from the ECU, ~10 ms later
  7C0  21 00 18 00 00 00 18 00     consecutive frame, ~40 ms after the first frame
  7C8  10 08 6F 29 11 03 00 00     positive response, also multi-frame
  7C0  30 00 00 00 00 00 00 00     our flow control so the ECU can finish its response

Reassembled, the request is:

  2F | 29 11 | 03 | 00 00 00 18 | 00 00 00 18
       DID     |    control      enable mask
               shortTermAdjustment

The enable mask is always 00 00 00 18 (two bits). The control record carries those same two bits
to light the lamps and clears them to put them out - the ECU echoes the control record back in
its 6F response, which is how the two states were told apart. The capture alternates the two
every 380 ms for 17 s, and that alternation is the flashing itself: the ECU does not blink on
its own, the tester toggles it.

Everything else in the capture is a red herring for this feature. The body control module
((0x750, 0x40)) writes around it (3B 13 ...) are one-shot and never toggled, and the 22 10 xx
reads sprinkled through it are a gauge app polling live data.

The sequence is handed to pandad as an OffroadCanScript: 14-byte records carrying a pre-send
delay, so the ISO-TP and blink timing survives the trip. pandad only plays it offroad, on bus 0:
with the car off the panda powers down every CAN transceiver except the main bus one, and OBD
multiplexing only ever moves bus 1, so bus 0 is the only way out. That is also the bus openpilot
runs its own Toyota diagnostic queries on.
"""

from collections.abc import Iterable
from typing import NamedTuple

CMD_ADDR = 0x7C0
CMD_BUS = 0

SCRIPT_RECORD_LEN = 14
MAX_DELAY_MS = 0xFFFF

# DiagnosticSessionControl. 0x2F is only accepted in the extended session, and returning to the
# default session at the end drops the short-term adjustment, so the ECU owns the lamps again
# without waiting out the session timeout.
SESSION_DEFAULT = b"\x02\x10\x01\x00\x00\x00\x00\x00"
SESSION_EXTENDED = b"\x02\x10\x03\x00\x00\x00\x00\x00"

# The two ISO-TP frames of the 0x2F request, split as in the capture.
REQUEST_FIRST_FRAME = b"\x10\x0C\x2F\x29\x11\x03\x00\x00"
REQUEST_CONSECUTIVE_ON = b"\x21\x00\x18\x00\x00\x00\x18\x00"
REQUEST_CONSECUTIVE_OFF = b"\x21\x00\x00\x00\x00\x00\x18\x00"

# Clear-to-send with no block limit, so the ECU can send the rest of its positive response.
FLOW_CONTROL = b"\x30\x00\x00\x00\x00\x00\x00\x00"

# Timing, all from the capture. The ECU's flow control came back ~10 ms after the first frame,
# so holding the consecutive frame for 40 ms leaves room for a slow one; sending it early would
# get it dropped. 380 ms per half cycle is the blink rate the capture ran at.
CONSECUTIVE_FRAME_DELAY_MS = 40
FLOW_CONTROL_DELAY_MS = 40
HALF_CYCLE_MS = 380
SESSION_DELAY_MS = 40

DEFAULT_FLASHES = 3


class ScriptFrame(NamedTuple):
  """One frame of the script: wait `delay_ms` after the previous frame, then send `data`."""
  delay_ms: int
  data: bytes
  addr: int = CMD_ADDR
  bus: int = CMD_BUS


def encode_script(frames: Iterable[ScriptFrame]) -> bytes:
  """Encode frames for pandad's OffroadCanScript param.

  One 14-byte record each: [delay_ms_hi, delay_ms_lo, addr_hi, addr_lo, bus, dlc, data[8]].
  """
  script = b""
  for frame in frames:
    if not 0 <= frame.delay_ms <= MAX_DELAY_MS:
      raise ValueError(f"delay out of range: {frame.delay_ms}")
    data = frame.data.ljust(8, b"\x00")
    if len(data) != 8:
      raise ValueError(f"frame must be at most 8 bytes: {frame.data!r}")
    script += bytes([(frame.delay_ms >> 8) & 0xFF, frame.delay_ms & 0xFF,
                     (frame.addr >> 8) & 0xFF, frame.addr & 0xFF,
                     frame.bus, len(data)]) + data
  return script


def _half_cycle(lamps_on: bool, lead_in_ms: int) -> list[ScriptFrame]:
  """One 380 ms half cycle: set the lamp state, then hold it until the next one."""
  consecutive = REQUEST_CONSECUTIVE_ON if lamps_on else REQUEST_CONSECUTIVE_OFF
  return [
    ScriptFrame(lead_in_ms, REQUEST_FIRST_FRAME),
    ScriptFrame(CONSECUTIVE_FRAME_DELAY_MS, consecutive),
    ScriptFrame(FLOW_CONTROL_DELAY_MS, FLOW_CONTROL),
  ]


def build_hazard_flash_frames(flashes: int = DEFAULT_FLASHES) -> list[ScriptFrame]:
  """Frames that blink the hazards `flashes` times and hand the lamps back to the ECU.

  The lamp state is set, not toggled, so a run that is cut short cannot invert the sequence, and
  the last thing sent before the session reset always puts the lamps out.
  """
  frames = [ScriptFrame(0, SESSION_DEFAULT), ScriptFrame(SESSION_DELAY_MS, SESSION_EXTENDED)]

  # The first frame of a half cycle follows the previous half cycle's flow control, so its lead-in
  # is what is left of the 380 ms after the consecutive frame and that flow control.
  hold_ms = HALF_CYCLE_MS - CONSECUTIVE_FRAME_DELAY_MS - FLOW_CONTROL_DELAY_MS
  for i in range(max(0, flashes)):
    frames += _half_cycle(True, SESSION_DELAY_MS if i == 0 else hold_ms)
    frames += _half_cycle(False, hold_ms)

  frames.append(ScriptFrame(hold_ms, SESSION_DEFAULT))
  return frames


def build_hazard_flash_script(flashes: int = DEFAULT_FLASHES) -> bytes:
  return encode_script(build_hazard_flash_frames(flashes))
