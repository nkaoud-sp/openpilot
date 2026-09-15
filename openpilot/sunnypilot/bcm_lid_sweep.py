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

FLOW_CONTROL = bytes([BCM_SUBADDR, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

# The hazard dongle sends TesterPresent before its reads and again every few requests, and the
# body ECU answers 0x7E. A KWP ECU that has gone quiet may need it before it will answer at all,
# so by default every probe is preceded by one.
TESTER_PRESENT = bytes([BCM_SUBADDR, 0x01, 0x3E, 0x00, 0x00, 0x00, 0x00, 0x00])
TESTER_PRESENT_RESPONSE = 0x7E


def build_request(lid: int, service: int = SVC_READ) -> bytes:
  """One sub-addressed KWP request frame for `lid`, padded to the 8 bytes ISO 15765-4 wants."""
  if not 0 <= lid <= 0xFF:
    raise ValueError(f"local identifier out of range: {lid}")
  if service == SVC_READ:
    payload = bytes([SVC_READ, lid])
  elif service == SVC_IO_CONTROL:
    payload = bytes([SVC_IO_CONTROL, lid]) + IO_CONTROL_PROBE_DATA
  else:
    raise ValueError(f"unsupported service: {service:#x}")
  return bytes([BCM_SUBADDR, len(payload)]) + payload.ljust(6, b"\x00")


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


def _decode_frame(frame: bytes) -> tuple[str, bytes, int]:
  """Split a sub-addressed frame into (kind, payload_or_fragment, declared_length).

  kind is "single", "first" (more to come), "consecutive", or "other" for anything not ours.
  """
  if len(frame) < 2 or frame[0] != BCM_SUBADDR:
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

  def send(self, frame: bytes):
    script = self._encode_script([self._ScriptFrame(0, frame, addr=BCM_REQ_ADDR, bus=BUS)])
    self.params.put("OffroadCanScript", script)

  def script_pending(self) -> bool:
    """True while a queued script is still waiting to be picked up.

    pandad removes the param the moment it takes the script (panda_safety.cc), so this going
    false is proof that pandad saw it - and it staying true means nothing was ever transmitted.
    """
    return bool(self.params.get("OffroadCanScript"))

  def poll(self) -> list[tuple[int, bytes]]:
    out = []
    raw = self._messaging.drain_sock_raw(self.can_sock)
    for _, frames in self._can_capnp_to_list(raw):
      for addr, dat, src in frames:
        if src == BUS:
          out.append((addr, bytes(dat)))
    return out


def _collect(link, seconds: float, watched: dict[int, bytes] | None = None) -> list[tuple[int, bytes]]:
  """Poll for `seconds`, returning every frame seen."""
  frames = []
  deadline = time.monotonic() + seconds
  while time.monotonic() < deadline:
    for addr, dat in link.poll():
      frames.append((addr, dat))
      if watched is not None and addr in WATCH_RANGE:
        watched[addr] = dat
    time.sleep(0.01)
  return frames


def preflight(link, log=print, listen: float = 1.5, timeout: float = 1.0) -> bool:
  """Check the three things that make every identifier look absent, and name which one failed.

  A silent sweep is ambiguous: an empty identifier space, a sleeping ECU and a panda that never
  transmitted all print the same thing. These separate them before 256 rows of "no response".
  """
  ok = True

  seen: dict[int, int] = {}
  for addr, _ in _collect(link, listen):
    seen[addr] = seen.get(addr, 0) + 1
  log(f"  bus 0 traffic     {sum(seen.values())} frames from {len(seen)} addresses in {listen:.1f} s")
  if not seen:
    log("                    -> nothing on bus 0. The powertrain bus is asleep or the panda is")
    log("                       not connected. Switch the ignition on (Always Offroad) and retry.")
    ok = False

  link.send(TESTER_PRESENT)
  consumed = False
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if not link.script_pending():
      consumed = True
      break
    time.sleep(0.02)
  log(f"  pandad picked up  {'yes' if consumed else 'NO'}")
  if not consumed:
    log("                    -> pandad never took the script, so nothing was transmitted. It only")
    log("                       plays them offroad: check the device is not onroad, and running.")
    ok = False

  answered = False
  for addr, dat in _collect(link, timeout / 2):
    kind, payload, _ = _decode_frame(dat)
    if addr == BCM_RESP_ADDR and kind == "single" and payload[:1] == bytes([TESTER_PRESENT_RESPONSE]):
      answered = True
  log(f"  body ECU awake    {'yes' if answered else 'NO'}  (TesterPresent -> 0x7E)")
  if consumed and not answered:
    log("                    -> the frame went out but (0x750, 0x40) did not answer. The ECU is")
    log("                       asleep or unreachable; a sweep now would report every identifier")
    log("                       absent whether or not it exists.")
    ok = False

  return ok


def probe(link, lid: int, service: int, settle: float, wake: bool = True) -> tuple[Response, dict[str, str]]:
  """Send one request and collect the answer, plus any broadcast that moved while we waited."""
  watched: dict[int, bytes] = {}
  for addr, dat in link.poll():          # drain stale frames, and snapshot the broadcasts
    if addr in WATCH_RANGE:
      watched[addr] = dat

  if wake:
    # Mirrors the dongle, which never read an identifier without a TesterPresent in front of it.
    link.send(TESTER_PRESENT)
    _collect(link, 0.08, watched)

  link.send(build_request(lid, service))

  resp = Response()
  pending: bytearray | None = None
  want = 0
  changed: dict[str, str] = {}
  deadline = time.monotonic() + settle

  while time.monotonic() < deadline:
    for addr, dat in link.poll():
      if addr in WATCH_RANGE:
        if addr in watched and watched[addr] != dat:
          changed[f"0x{addr:03X}"] = f"{watched[addr].hex(' ')} -> {dat.hex(' ')}"
        watched[addr] = dat
      if addr != BCM_RESP_ADDR:
        continue

      kind, chunk, length = _decode_frame(dat)
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
        link.send(FLOW_CONTROL)
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


def sweep(link, lids, service: int, settle: float, log=print, wake: bool = True) -> dict:
  results: dict[int, Response] = {}
  side_effects: dict[int, dict[str, str]] = {}

  for lid in lids:
    resp, changed = probe(link, lid, service, settle, wake)
    results[lid] = resp
    if changed:
      side_effects[lid] = changed
    flag = "   ** something changed: " + "; ".join(f"{a} {d}" for a, d in changed.items()) if changed else ""
    log(f"  0x{lid:02X}  {resp.verdict:<22} {resp.data or resp.raw}{flag}")

  return {
    "service": f"0x{service:02X}",
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

  if args.control and not args.yes:
    print("--control sends InputOutputControlByLocalIdentifier to identifiers nobody has mapped.")
    print("Every data bit is clear, so a control that follows the lock/mirror encoding does nothing,")
    print("but that is an inference. The car should be parked and you should be able to see it.")
    if input("type 'parked' to continue: ").strip() != "parked":
      raise SystemExit("aborted")

  service = SVC_IO_CONTROL if args.control else SVC_READ
  settle = args.settle if args.settle is not None else (1.0 if args.control else 0.5)

  link = DeviceLink()
  if link.onroad() and not args.force:
    raise SystemExit("device is onroad; pandad will not play the script. Use Always Offroad, or --force.")

  if not args.no_preflight:
    print("\npreflight:")
    if not preflight(link) and not args.force:
      raise SystemExit("\npreflight failed; fix the above or pass --force to sweep anyway.")

  print(f"\nsweeping {len(lids)} identifiers on (0x{BCM_REQ_ADDR:03X}, 0x{BCM_SUBADDR:02X}) with service 0x{service:02X}\n")
  out = sweep(link, lids, service, settle, wake=not args.no_wake)

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
