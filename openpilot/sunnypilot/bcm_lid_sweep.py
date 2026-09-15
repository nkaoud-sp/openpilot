#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Body ECU local-identifier sweep: what does (0x750, 0x40) actually expose?

The body ECU speaks KWP, not UDS. It answers 0x21 ReadDataByLocalIdentifier and 0x30
InputOutputControlByLocalIdentifier on a sparse set of single-byte local identifiers, and
answers everything else with 0x7F 30 12 (subFunctionNotSupported). The auto-lock feature
already uses three of them; this finds the rest.

Known-good identifiers, for calibration (see autolock_commands.py):

  30 11 00 80 00   doors lock       (0x750, 0x40)   BCM
  30 11 00 40 00   doors unlock     (0x750, 0x40)   BCM
  30 21 00 08      mirrors fold     (0x750, 0xA5/0xA6)  mirror modules, not the BCM
  30 01 05 20      windows close    (0x750, 0x90-0x93)  door modules, not the BCM

Note the frame shape differs by module: the BCM takes three data bytes (DLC 5), the mirror
and door modules take two (DLC 4). The hazard-dongle capture confirms it - its BCM probe
`40 05 30 15 00 C0 00` came back 0x7F 30 12 (no such identifier) rather than 0x7F 30 13
(incorrect length), so five is the right length and 0x15 simply isn't an identifier.

Two modes:

  read (default)     21 <lid>          Non-actuating. Maps which identifiers exist and what
                                       they hold. Run it twice - once with hazards off, once
                                       with the hazard switch on - then --diff the two to find
                                       the identifier that tracks lamp state.

  control            30 <lid> 00 00 00 Probes whether an identifier accepts I/O control, with
                                       every data bit clear so nothing is driven. The dongle
                                       capture is the evidence this is safe: it sent
                                       `30 21 00 00` to a mirror module and got a positive
                                       response with the mirror not moving, against
                                       `30 21 00 08` on the other side which folded it.

                                       "All bits clear means no action" is inferred from the
                                       lock and mirror encodings, not documented. An identifier
                                       nobody has seen could read 0x00 as a state rather than a
                                       no-op. Park the car before using this mode.

Requires the device to be OFFROAD: pandad only plays OffroadCanScript with deviceState.started
false (panda_safety.cc). To sweep with the ignition on, put the device in Always Offroad first.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict

BCM_REQ_ADDR = 0x750
BCM_RESP_ADDR = 0x758
BCM_SUBADDR = 0x40
BUS = 0

SVC_READ = 0x21           # ReadDataByLocalIdentifier
SVC_IO_CONTROL = 0x30     # InputOutputControlByLocalIdentifier

# The BCM's I/O control payload is `30 <lid> <3 data bytes>`; a probe clears all of them.
IO_CONTROL_PROBE_DATA = b"\x00\x00\x00"

NRC = {
  0x10: "generalReject",
  0x11: "serviceNotSupported",
  0x12: "subFunctionNotSupported",
  0x13: "incorrectLength",
  0x21: "busyRepeatRequest",
  0x22: "conditionsNotCorrect",
  0x31: "requestOutOfRange",
  0x33: "securityAccessDenied",
  0x78: "responsePending",
}

# Identifiers the known-good controls sit on: 0x01, 0x11, 0x21, spaced 0x10 apart. Sweep that
# grid first; --all falls back to the whole single-byte space.
GRID_LIDS = [0x01 + 0x10 * i for i in range(16)]

# Body-ECU broadcasts on bus 0. Watched during each probe so that anything which actuates is
# attributed to the identifier that caused it rather than being noticed later.
WATCH_RANGE = range(0x600, 0x700)

# 0x750 is shared and sub-addressed: 0x40 is the body ECU, 0xA5/0xA6 the mirror modules,
# 0x90-0x93 the door modules, and 0x0F the forward radar. Which of them are reachable depends on
# the bus: openpilot queries 0x750 on bus 0 for the radar only, and opendbc lists the body ECU
# among the ECUs *not* queried there. Probing 0x0F is therefore the control experiment - it is
# known to answer on bus 0, so if it replies and 0x40 does not, the transport is fine and the
# body ECU simply is not on this bus.
RADAR_SUBADDR = 0x0F


def flow_control(subaddr: int = BCM_SUBADDR) -> bytes:
  return bytes([subaddr, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])


def wake_frame(subaddr: int = BCM_SUBADDR) -> bytes:
  return bytes([subaddr, 0x01, 0x3E, 0x00, 0x00, 0x00, 0x00, 0x00])

# The hazard dongle sends TesterPresent before its reads and again every few requests, and the
# body ECU answers 0x7E. A KWP ECU that has gone quiet may need it before it will answer at all,
# so by default every probe is preceded by one.
TESTER_PRESENT_RESPONSE = 0x7E

# The panda re-publishes what it transmits, tagging the bus so the frame's fate is visible:
# 0x80 + bus for a frame that went out, 0xC0 + bus for one the safety model refused to send
# (panda.h). Without this, a frame that never reached the wire and an ECU that stayed silent
# look identical.
RETURNED_SRC = 0x80 + BUS
REJECTED_SRC = 0xC0 + BUS

# pandad polls for a queued script every 100 ms, and its panda-state loop re-asserts NO_OUTPUT
# every 100 ms (pandad.cc). Two timers of the same period can phase-lock, and a safety-model
# change re-inits the CAN cores - which flushes a frame still sitting in the TX FIFO. Giving each
# request a different pre-send delay walks it around that beat instead of landing on it.
SEND_JITTER_MS = (0, 37, 71, 13, 53)


def build_request(lid: int, service: int = SVC_READ, subaddr: int = BCM_SUBADDR) -> bytes:
  """One sub-addressed KWP request frame for `lid`, padded to the 8 bytes ISO 15765-4 wants."""
  if not 0 <= lid <= 0xFF:
    raise ValueError(f"local identifier out of range: {lid}")
  if service == SVC_READ:
    payload = bytes([SVC_READ, lid])
  elif service == SVC_IO_CONTROL:
    payload = bytes([SVC_IO_CONTROL, lid]) + IO_CONTROL_PROBE_DATA
  else:
    raise ValueError(f"unsupported service: {service:#x}")
  return bytes([subaddr, len(payload)]) + payload.ljust(6, b"\x00")


@dataclass
class Response:
  """One decoded answer from the body ECU."""
  raw: str = ""
  positive: bool = False
  nrc: int | None = None
  data: str = ""          # hex of the bytes after the service + echoed identifier
  multiframe: bool = False
  truncated: bool = False

  @property
  def verdict(self) -> str:
    if self.positive:
      return "LIVE"
    if self.nrc is None:
      return "no response"
    if self.nrc == 0x12:
      return "not an identifier"
    return NRC.get(self.nrc, f"NRC {self.nrc:#04x}")


def parse_payload(payload: bytes, service: int) -> Response:
  """Classify a reassembled KWP payload. Kept separate from the socket so it can be tested."""
  resp = Response(raw=payload.hex(" "))
  if not payload:
    return resp
  if payload[0] == 0x7F:
    resp.nrc = payload[2] if len(payload) >= 3 else 0
  elif payload[0] == service + 0x40:
    resp.positive = True
    resp.data = payload[2:].hex(" ")
  return resp


def _decode_frame(frame: bytes, subaddr: int = BCM_SUBADDR) -> tuple[str, bytes, int]:
  """Split a sub-addressed frame into (kind, payload_or_fragment, declared_length).

  kind is "single", "first" (more to come), "consecutive", or "other" for anything not ours.
  """
  if len(frame) < 2 or frame[0] != subaddr:
    return "other", b"", 0
  pci = frame[1] >> 4
  if pci == 0:
    length = frame[1] & 0x0F
    return "single", frame[2:2 + length], length
  if pci == 1:
    length = ((frame[1] & 0x0F) << 8) | frame[2]
    return "first", frame[3:], length
  if pci == 2:
    return "consecutive", frame[2:], 0
  return "other", b"", 0


class DeviceLink:
  """The only part that touches the device: queue a frame, collect bus-0 frames, read onroad.

  Requests go out through OffroadCanScript rather than OffroadCanQueue. A KWP negative response
  does not echo the identifier it refused, so only one request can be in flight if answers are
  to be matched to probes - and the script path is the one with the ELM327 linger that stops a
  frame being flushed out of the TX FIFO by the safety-model change (see panda_safety.cc).
  """

  def __init__(self):
    import openpilot.cereal.messaging as messaging
    from openpilot.common.params import Params
    from openpilot.selfdrive.pandad import can_capnp_to_list
    from openpilot.sunnypilot.hazard_flash import ScriptFrame, encode_script

    self._messaging = messaging
    self._can_capnp_to_list = can_capnp_to_list
    self._ScriptFrame = ScriptFrame
    self._encode_script = encode_script

    self.params = Params()
    self.can_sock = messaging.sub_sock("can", conflate=False, timeout=0)
    self.sm = messaging.SubMaster(["deviceState"])

  def onroad(self) -> bool:
    self.sm.update(1000)
    return bool(self.sm.alive["deviceState"] and self.sm["deviceState"].started)

  def send(self, frame: bytes, delay_ms: int = 0):
    script = self._encode_script([self._ScriptFrame(delay_ms, frame, addr=BCM_REQ_ADDR, bus=BUS)])
    self.params.put("OffroadCanScript", script)

  def script_pending(self) -> bool:
    """True while a queued script is still waiting to be picked up.

    pandad removes the param the moment it takes the script (panda_safety.cc), so this going
    false is proof that pandad saw it - and it staying true means nothing was ever transmitted.
    """
    return bool(self.params.get("OffroadCanScript"))

  def poll(self) -> list[tuple[int, bytes, int]]:
    """Every frame, with its src: bus 0 for received, RETURNED_SRC/REJECTED_SRC for our own."""
    out = []
    raw = self._messaging.drain_sock_raw(self.can_sock)
    for _, frames in self._can_capnp_to_list(raw):
      for addr, dat, src in frames:
        if src in (BUS, RETURNED_SRC, REJECTED_SRC):
          out.append((addr, bytes(dat), src))
    return out


def _collect(link, seconds: float, watched: dict[int, bytes] | None = None) -> list[tuple[int, bytes, int]]:
  """Poll for `seconds`, returning every frame seen."""
  frames = []
  deadline = time.monotonic() + seconds
  while time.monotonic() < deadline:
    for addr, dat, src in link.poll():
      frames.append((addr, dat, src))
      if watched is not None and src == BUS and addr in WATCH_RANGE:
        watched[addr] = dat
    time.sleep(0.01)
  return frames


def preflight(link, log=print, listen: float = 1.5, timeout: float = 1.0,
              subaddr: int = BCM_SUBADDR) -> bool:
  """Check the three things that make every identifier look absent, and name which one failed.

  A silent sweep is ambiguous: an empty identifier space, a sleeping ECU and a panda that never
  transmitted all print the same thing. These separate them before 256 rows of "no response".
  """
  ok = True

  # Informational only. A quiet bus is what the car off looks like, and that is a state worth
  # sweeping in - the auto-lock drives this same ECU, on this same bus, with the car off. The
  # echo check below is the one that actually proves the frame got out, so this must not gate.
  seen: dict[int, int] = {}
  for addr, _, src in _collect(link, listen):
    if src == BUS:
      seen[addr] = seen.get(addr, 0) + 1
  quiet = "  (car appears to be off)" if not seen else ""
  log(f"  bus 0 traffic     {sum(seen.values())} frames from {len(seen)} addresses in {listen:.1f} s{quiet}")

  # Walk the jitter so a phase-locked collision with pandad's NO_OUTPUT beat can't hide the
  # answer on every attempt.
  transmitted = rejected = answered = False
  consumed = False
  for attempt, delay_ms in enumerate(SEND_JITTER_MS):
    link.send(wake_frame(subaddr), delay_ms)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and link.script_pending():
      time.sleep(0.02)
    consumed = consumed or not link.script_pending()

    for addr, dat, src in _collect(link, timeout / 2 + delay_ms / 1000.0):
      if addr == BCM_REQ_ADDR and src == RETURNED_SRC:
        transmitted = True
      if addr == BCM_REQ_ADDR and src == REJECTED_SRC:
        rejected = True
      kind, payload, _ = _decode_frame(dat, subaddr)
      if addr == BCM_RESP_ADDR and src == BUS and kind == "single" and payload[:1] == bytes([TESTER_PRESENT_RESPONSE]):
        answered = True
    if answered:
      log(f"  (answered on attempt {attempt + 1} of {len(SEND_JITTER_MS)}, {delay_ms} ms pre-send delay)")
      break

  log(f"  pandad picked up  {'yes' if consumed else 'NO'}")
  if not consumed:
    log("                    -> pandad never took the script, so nothing was transmitted. It only")
    log("                       plays them offroad: check the device is not onroad, and running.")
    ok = False

  log(f"  frame reached bus {'yes' if transmitted else 'NO'}   (panda echoes what it sends)")
  if consumed and not transmitted and not rejected:
    log("                    -> pandad queued it but the panda never put it on the wire. Its")
    log("                       panda-state loop re-asserts NO_OUTPUT every 100 ms, and a")
    log("                       safety-model change re-inits the CAN cores, flushing the TX FIFO.")
    ok = False
  if rejected:
    log("  safety rejected   YES  -> the panda's safety model refused the frame, not the ECU.")
    ok = False

  log(f"  0x{subaddr:02X} answers      {'yes' if answered else 'NO'}  (TesterPresent -> 0x7E)")
  if transmitted and not answered:
    log(f"                    -> the frame reached the bus and (0x750, 0x{subaddr:02X}) did not answer, so")
    log("                       that module is asleep or not on this bus. A sweep now would report")
    log("                       every identifier absent whether or not it exists.")
    if subaddr != RADAR_SUBADDR:
      log(f"                       Try --subaddr 0x{RADAR_SUBADDR:02X}: the radar is known to answer on bus 0,")
      log("                       so if it replies the transport is fine and this module is elsewhere.")
    ok = False

  return ok


# What a lamp control would show up as, from the labelled rlog: 0x614 byte 3 carries the hazard
# flag and the turn-signal field, and the body ECU broadcasts it on change with byte 1 bit 0x80
# raised for about a second.
HAZARD_MSG = 0x614
HAZARD_BYTE = 3
HAZARD_BIT = 0x08
TURN_FIELD_MASK = 0x30


def describe_hazard(before: bytes, after: bytes) -> str | None:
  """Name the change if 0x614's lamp bits moved, so the one that matters isn't just another row."""
  if len(before) <= HAZARD_BYTE or len(after) <= HAZARD_BYTE:
    return None
  b, a = before[HAZARD_BYTE], after[HAZARD_BYTE]
  if (b ^ a) & (HAZARD_BIT | TURN_FIELD_MASK) == 0:
    return None
  bits = []
  if (b ^ a) & HAZARD_BIT:
    bits.append("HAZARD " + ("on" if a & HAZARD_BIT else "off"))
  if (b ^ a) & TURN_FIELD_MASK:
    bits.append({1: "left", 2: "right", 3: "none"}.get((a & TURN_FIELD_MASK) >> 4, "?"))
  return " + ".join(bits)


def effect_probe(link, lid: int, value: int, settle: float, subaddr: int,
                 log=print) -> dict[str, str]:
  """Drive one identifier and watch what the body ECU says about it afterwards.

  There are no answers to read on this bus - the auto-lock has always worked without one - so a
  live identifier is found by what it changes, not by what it replies. The control is released
  again straight after, so nothing is left latched.
  """
  watched: dict[int, bytes] = {}
  for addr, dat, src in link.poll():
    if src == BUS and addr in WATCH_RANGE:
      watched[addr] = dat

  log(f"  0x{lid:02X}  sending 30 {lid:02X} 00 {value:02X} 00 ...")
  payload = bytes([SVC_IO_CONTROL, lid, 0x00, value, 0x00])
  link.send(bytes([subaddr, len(payload)]) + payload.ljust(6, b"\x00"))

  changed: dict[str, str] = {}
  for addr, dat, src in _collect(link, settle):
    if src != BUS or addr not in WATCH_RANGE:
      continue
    if addr in watched and watched[addr] != dat:
      note = describe_hazard(watched[addr], dat) if addr == HAZARD_MSG else None
      label = f"{watched[addr].hex(' ')} -> {dat.hex(' ')}"
      changed[f"0x{addr:03X}"] = f"{label}   <<< {note}" if note else label
    watched[addr] = dat

  # Hand the output back whatever happened, so a run that is cut short can't leave lamps latched.
  release = bytes([SVC_IO_CONTROL, lid, 0x00, 0x00, 0x00])
  link.send(bytes([subaddr, len(release)]) + release.ljust(6, b"\x00"))
  _collect(link, 0.4)
  return changed


def effect_sweep(link, lids, value: int, settle: float, subaddr: int, log=print) -> dict:
  out: dict[int, dict[str, str]] = {}
  for lid in lids:
    changed = effect_probe(link, lid, value, settle, subaddr, log)
    if changed:
      out[lid] = changed
      for addr, what in changed.items():
        log(f"        {addr}  {what}")
  return {"mode": "effect", "value": f"0x{value:02X}", "subaddr": f"0x{subaddr:02X}",
          "effects": {f"0x{lid:02X}": v for lid, v in out.items()}}


def probe(link, lid: int, service: int, settle: float, wake: bool = True,
          attempts: int = 3, subaddr: int = BCM_SUBADDR) -> tuple[Response, dict[str, str]]:
  """Send one request and collect the answer, plus any broadcast that moved while we waited.

  Retried with a different pre-send delay each time: a frame lost to the NO_OUTPUT beat is
  silence, indistinguishable from an identifier that does not exist, so silence is not trusted
  until it has survived a few different phases.
  """
  changed: dict[str, str] = {}
  resp = Response()
  for attempt in range(attempts):
    resp, moved = _probe_once(link, lid, service, settle, wake,
                              SEND_JITTER_MS[attempt % len(SEND_JITTER_MS)], subaddr)
    changed.update(moved)
    if resp.verdict != "no response":
      break
  return resp, changed


def _probe_once(link, lid: int, service: int, settle: float, wake: bool,
                delay_ms: int, subaddr: int = BCM_SUBADDR) -> tuple[Response, dict[str, str]]:
  watched: dict[int, bytes] = {}
  for addr, dat, src in link.poll():     # drain stale frames, and snapshot the broadcasts
    if src == BUS and addr in WATCH_RANGE:
      watched[addr] = dat

  if wake:
    # Mirrors the dongle, which never read an identifier without a TesterPresent in front of it.
    link.send(wake_frame(subaddr), delay_ms)
    _collect(link, 0.08 + delay_ms / 1000.0, watched)

  link.send(build_request(lid, service, subaddr), delay_ms)

  resp = Response()
  pending: bytearray | None = None
  want = 0
  changed: dict[str, str] = {}
  deadline = time.monotonic() + settle

  while time.monotonic() < deadline:
    for addr, dat, src in link.poll():
      if src != BUS:
        continue          # our own frame echoed back, not an answer
      if addr in WATCH_RANGE:
        if addr in watched and watched[addr] != dat:
          changed[f"0x{addr:03X}"] = f"{watched[addr].hex(' ')} -> {dat.hex(' ')}"
        watched[addr] = dat
      if addr != BCM_RESP_ADDR:
        continue

      kind, chunk, length = _decode_frame(dat, subaddr)
      if kind == "single":
        if chunk[:1] == bytes([TESTER_PRESENT_RESPONSE]):
          continue        # a late answer to the wake frame, not to this probe
        candidate = parse_payload(chunk, service)
        if candidate.nrc == 0x78:
          # responsePending: the ECU is asking for more time, not answering.
          deadline = time.monotonic() + settle
          continue
        resp = candidate
        deadline = min(deadline, time.monotonic() + 0.05)
      elif kind == "first":
        pending, want = bytearray(chunk), length
        resp.multiframe = True
        link.send(flow_control(subaddr))
      elif kind == "consecutive" and pending is not None:
        pending += chunk
        if len(pending) >= want:
          resp = parse_payload(bytes(pending[:want]), service)
          resp.multiframe = True
          pending = None
          deadline = min(deadline, time.monotonic() + 0.05)
    time.sleep(0.01)

  if pending is not None:
    resp.truncated = True
  return resp, changed


def sweep(link, lids, service: int, settle: float, log=print, wake: bool = True,
          subaddr: int = BCM_SUBADDR) -> dict:
  results: dict[int, Response] = {}
  side_effects: dict[int, dict[str, str]] = {}

  for lid in lids:
    resp, changed = probe(link, lid, service, settle, wake, subaddr=subaddr)
    results[lid] = resp
    if changed:
      side_effects[lid] = changed
    flag = "   ** something changed: " + "; ".join(f"{a} {d}" for a, d in changed.items()) if changed else ""
    log(f"  0x{lid:02X}  {resp.verdict:<22} {resp.data or resp.raw}{flag}")

  return {
    "service": f"0x{service:02X}",
    "subaddr": f"0x{subaddr:02X}",
    "results": {f"0x{lid:02X}": asdict(r) for lid, r in results.items()},
    "side_effects": {f"0x{lid:02X}": v for lid, v in side_effects.items()},
  }


def do_diff(path_a: str, path_b: str, log=print):
  with open(path_a) as f:
    a = json.load(f)
  with open(path_b) as f:
    b = json.load(f)
  ra, rb = a["results"], b["results"]
  log(f"identifiers whose value differs between {path_a} and {path_b}:\n")
  found = False
  for lid in sorted(set(ra) & set(rb)):
    va, vb = ra[lid], rb[lid]
    if va.get("positive") and vb.get("positive") and va.get("data") != vb.get("data"):
      found = True
      log(f"  {lid}   {va['data']}   ->   {vb['data']}")
  if not found:
    log("  (none - no live identifier changed between the two runs)")


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--all", action="store_true", help="sweep 0x00-0xFF instead of the 0x10 grid")
  p.add_argument("--lids", help="explicit comma-separated identifiers, e.g. 0x11,0x21")
  p.add_argument("--control", action="store_true",
                 help="probe 0x30 I/O control with all data bits clear instead of reading. Park the car.")
  p.add_argument("--settle", type=float, help="seconds to wait per identifier (default 0.5 read, 1.0 control)")
  p.add_argument("--out", help="write results as JSON, for --diff")
  p.add_argument("--diff", nargs=2, metavar=("A.json", "B.json"), help="compare two runs and report what moved")
  p.add_argument("--force", action="store_true", help="sweep even if the device reports onroad")
  p.add_argument("--yes", action="store_true", help="skip the confirmation prompt for --control")
  p.add_argument("--no-wake", action="store_true", help="don't send TesterPresent before each probe")
  p.add_argument("--no-preflight", action="store_true", help="skip the bus/pandad/ECU checks")
  subaddr_help = f"0x750 sub-address (default {hex(BCM_SUBADDR)} body ECU; {hex(RADAR_SUBADDR)} radar, answers on bus 0)"
  p.add_argument("--subaddr", default=hex(BCM_SUBADDR), help=subaddr_help)
  p.add_argument("--effect", action="store_true",
                 help="drive each identifier and watch what changes, for a bus with no answers. ACTUATES.")
  p.add_argument("--value", default="0x08",
                 help="data byte for --effect (default 0x08, the hazard bit in 0x614 byte 3)")
  args = p.parse_args()

  if args.diff:
    do_diff(*args.diff)
    return

  if args.lids:
    lids = [int(x, 0) for x in args.lids.split(",")]
  elif args.all:
    lids = list(range(0x100))
  else:
    lids = GRID_LIDS

  if args.effect and not args.yes:
    print("--effect DRIVES each identifier: it sends a real, non-zero control and watches what moves.")
    print("Doors, windows, mirrors, lamps and the horn all live in this identifier space. Park the")
    print("car outside, keep hands clear of the windows, and watch it while this runs. Each control")
    print("is released again immediately, and Ctrl-C stops between identifiers.")
    if input("type 'drive it' to continue: ").strip() != "drive it":
      raise SystemExit("aborted")

  if args.control and not args.yes:
    print("--control sends InputOutputControlByLocalIdentifier to identifiers nobody has mapped.")
    print("Every data bit is clear, so a control that follows the lock/mirror encoding does nothing,")
    print("but that is an inference. The car should be parked and you should be able to see it.")
    if input("type 'parked' to continue: ").strip() != "parked":
      raise SystemExit("aborted")

  service = SVC_IO_CONTROL if args.control else SVC_READ
  settle = args.settle if args.settle is not None else (1.0 if args.control else 0.5)
  subaddr = int(args.subaddr, 0)

  link = DeviceLink()
  if link.onroad() and not args.force:
    raise SystemExit("device is onroad; pandad will not play the script. Use Always Offroad, or --force.")

  if not args.no_preflight:
    print("\npreflight:")
    if not preflight(link, subaddr=subaddr) and not args.force:
      raise SystemExit("\npreflight failed; fix the above or pass --force to sweep anyway.")

  if args.effect:
    value = int(args.value, 0)
    print(f"\ndriving {len(lids)} identifiers on (0x{BCM_REQ_ADDR:03X}, 0x{subaddr:02X}) with data 0x{value:02X}\n")
    out = effect_sweep(link, lids, value, args.settle or 1.5, subaddr)
    hits = out["effects"]
    print(f"\n{len(hits)} identifier(s) changed something: {', '.join(hits) if hits else '(none)'}")
    for lid, changed in hits.items():
      if any("HAZARD" in v for v in changed.values()):
        print(f"  {lid} moved the hazard bit in 0x614 - this is the one")
  else:
    print(f"\nsweeping {len(lids)} identifiers on (0x{BCM_REQ_ADDR:03X}, 0x{subaddr:02X}) with service 0x{service:02X}\n")
    out = sweep(link, lids, service, settle, wake=not args.no_wake, subaddr=subaddr)

    live = [k for k, v in out["results"].items() if v["positive"]]
    print(f"\n{len(live)} live identifier(s): {', '.join(live) if live else '(none)'}")
    if out["side_effects"]:
      print(f"identifiers that changed a broadcast: {', '.join(out['side_effects'])}")

  if args.out:
    with open(args.out, "w") as f:
      json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
