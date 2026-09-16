#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Body ECU local identifier scan: find which KWP2000 InputOutputControlByLocalIdentifier (service
0x30) commands an ECU behind the 0x750 gateway accepts, and which of them touch the turn lamps.

Why: the hazard flash the OBD dongle used is UDS 0x2F on the combination meter (0x7C0), and the
meter refuses it with NRC 0x22 (conditionsNotCorrect) from about 4 km/h. The auto-lock commands
in this fork go a different way: KWP 0x30 to sub-addressed body ECUs, e.g. the door lock is

  750  40 05 30 11 00 80 00 00     sub-address 0x40 (body ECU), LID 0x11, control 00 80

and those are not known to be speed conditioned. If the body ECU has a LID for the flasher it is
the way to blink the hazards at braking speeds. Nobody has that LID written down, so this scans
for it.

The scan sends `30 LID 00 00 00` for every LID to one sub-address, one probe every PROBE_GAP_MS:
the lock frame with its bits cleared, down to the length byte, since that is the one request the
body ECU is known to take. The control bytes are a bitfield in every command known so far (lock
00 80, unlock 00 40, mirror 00 08), so all zeros is the pattern least likely to actuate anything,
but it is not guaranteed inert for every LID: run it parked, ignition on, engine off, trunk
clear, and watch the car. The ECU answers each probe on 0x758:

  758  40 03 7F 30 12 ...           negative, NRC 0x12: the LID does not exist
  758  40 xx 70 LID ...             positive: the LID exists (and may have done something)

A positive reply with a LID nobody knows is a candidate. For those, the bit sweep sends each of
the 16 control bits on its own, followed by all zeros to release it, and watches BLINKERS_STATE
(0x614) for HAZARD_LIGHT or TURN_SIGNALS changing.

Sending goes through pandad's OffroadCanScript, so this only runs offroad, on bus 0, like the
hazard flash test. Recording reads the `can` stream pandad publishes offroad; probes are matched
to replies by the panda's transmit echo (src >= 128), falling back to the script's own timing.
"""

import argparse
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from openpilot.sunnypilot.hazard_flash import ScriptFrame, encode_script, script_duration_s

DIAG_ADDR = 0x750
REPLY_ADDR = 0x758
BLINKERS_STATE_ADDR = 0x614
CMD_BUS = 0

SUB_ADDR_BODY = 0x40

SERVICE_IO_CONTROL = 0x30
POSITIVE_IO_CONTROL = 0x70
NEGATIVE_RESPONSE = 0x7F

# The body ECU drops back-to-back diagnostic frames (see autolock_commands), and a reply plus any
# lamp reaction has to land before the next probe so it can be attributed to the right LID.
PROBE_GAP_MS = 200
# A set bit may switch something on; give it long enough to show on 0x614 (event pulse is
# immediate, periodic copy is 1 Hz) before the release frame, then settle before the next bit.
SWEEP_ON_MS = 800
SWEEP_OFF_MS = 800

ALL_LIDS = range(0x100)

# Three bytes, like the lock command: the two the known commands use as a bitfield plus a zero.
DEFAULT_CONTROL = b"\x00\x00\x00"

# What the body ECU's 0x30 identifiers are known to drive, from the auto-lock commands and from
# eyes-on bit sweeps of a 2020 Lexus ES 350 (ignition off). Labels in the reports, nothing more.
KNOWN_LIDS = {
  SUB_ADDR_BODY: {
    0x11: "door locks",
    0x12: "cabin light, dash relay",
    0x19: "rear sunshade",
  },
}
KNOWN_CONTROLS = {
  SUB_ADDR_BODY: {
    (0x11, b"\x00\x80"): "door lock",
    (0x11, b"\x00\x40"): "door unlock",
    (0x12, b"\x00\x02"): "relay under the dash, load not yet identified",
    (0x12, b"\x00\x80"): "cabin light on",
    (0x19, b"\x00\x40"): "rear sunshade open",
  },
}

NRC_NAMES = {
  0x10: "generalReject",
  0x11: "serviceNotSupported",
  0x12: "subFunctionNotSupported",
  0x21: "busy",
  0x22: "conditionsNotCorrect",
  0x23: "routineNotComplete",
  0x31: "requestOutOfRange",
  0x33: "securityAccessDenied",
  0x35: "invalidKey",
  0x78: "responsePending",
}
# NRCs that just mean "no such LID"; anything else means the ECU knows the LID.
NRC_UNSUPPORTED = {0x12, 0x31}
# The ECU does not implement service 0x30 at all (a UDS-only ECU answers every LID this way).
NRC_SERVICE_NOT_SUPPORTED = 0x11

DEFAULT_REPORT_DIR = "/data"

# pandad only plays OffroadCanScript offroad (ignition off) and clears it when the car goes onroad.
NOT_PLAYED_WARNING = (
  "SCRIPT NOT PLAYED: pandad never picked it up. It only plays offroad, so the ignition must be off " +
  "(the body ECU and the hazards work without it). The script was removed so it does not run by itself " +
  "at the next ignition off."
)
REPORT_NAME = "body_lid_scan.txt"


def build_probe(lid: int, control: bytes = DEFAULT_CONTROL, sub_addr: int | None = SUB_ADDR_BODY) -> bytes:
  """One 0x30 request, zero padded to 8 bytes.

  Through the 0x750 gateway: [sub_addr, len, 0x30, LID, control...]. To an ECU with its own address
  (sub_addr None, e.g. the meter at 0x7C0): a plain ISO-TP single frame [len, 0x30, LID, control...].
  """
  if not 0 <= lid <= 0xFF:
    raise ValueError(f"LID out of range: {lid}")
  if sub_addr is not None and not 0 <= sub_addr <= 0xFF:
    raise ValueError(f"sub-address out of range: {sub_addr}")
  if not 1 <= len(control) <= 4:
    raise ValueError(f"control record must be 1 to 4 bytes: {control!r}")
  body = bytes([SERVICE_IO_CONTROL, lid]) + control
  if sub_addr is None:
    return bytes([len(body)]) + body.ljust(7, b"\x00")
  return bytes([sub_addr, len(body)]) + body.ljust(6, b"\x00")


def probe_body(data: bytes, sub_addr: int | None = SUB_ADDR_BODY) -> bytes | None:
  """The KWP payload of a frame in the addressing this scan uses, or None if it is not one of ours."""
  if sub_addr is None:
    return data[1:1 + data[0]] if len(data) >= 2 and 1 <= data[0] <= 7 else None
  if len(data) >= 3 and data[0] == sub_addr and 1 <= data[1] <= 6:
    return data[2:2 + data[1]]
  return None


def probe_lid(data: bytes, sub_addr: int | None = SUB_ADDR_BODY) -> int | None:
  """The LID of a probe frame this module built, or None for anything else."""
  body = probe_body(data, sub_addr)
  if body is not None and len(body) >= 2 and body[0] == SERVICE_IO_CONTROL:
    return body[1]
  return None


def build_lid_scan_frames(lids: Iterable[int] = ALL_LIDS, sub_addr: int | None = SUB_ADDR_BODY,
                          gap_ms: int = PROBE_GAP_MS, addr: int = DIAG_ADDR) -> list[ScriptFrame]:
  """Probe every LID once with an all-zero control record."""
  frames = []
  for i, lid in enumerate(lids):
    frames.append(ScriptFrame(0 if i == 0 else gap_ms, build_probe(lid, sub_addr=sub_addr), addr=addr, bus=CMD_BUS))
  return frames


def sweep_controls() -> list[bytes]:
  """The 16 single-bit control records, second byte first since that is where the known commands' bits are."""
  return [bytes([0, 1 << b, 0]) for b in range(8)] + [bytes([1 << b, 0, 0]) for b in range(8)]


def combo_controls() -> list[bytes]:
  """Selector plus action, the shape of the window command (30 01 05 20: window 5, action 0x20).

  A LID that addresses several outputs may need a non-zero first byte before any bit in the second
  does anything, so this pairs every first byte 0x00..0x0F with each single bit of the second.
  The single-bit sweep is the selector-0 row of this table.
  """
  return [bytes([sel, 1 << b, 0]) for sel in range(0x10) for b in range(8)]


def build_bit_sweep_frames(lid: int, sub_addr: int | None = SUB_ADDR_BODY,
                           on_ms: int = SWEEP_ON_MS, off_ms: int = SWEEP_OFF_MS,
                           controls: Sequence[bytes] | None = None, addr: int = DIAG_ADDR) -> list[ScriptFrame]:
  """Set each control record on its own, releasing it with all zeros before the next one."""
  frames = []
  for i, control in enumerate(sweep_controls() if controls is None else controls):
    frames.append(ScriptFrame(0 if i == 0 else off_ms, build_probe(lid, control, sub_addr), addr=addr, bus=CMD_BUS))
    frames.append(ScriptFrame(on_ms, build_probe(lid, DEFAULT_CONTROL, sub_addr), addr=addr, bus=CMD_BUS))
  return frames


def classify_reply(data: bytes, sub_addr: int | None = SUB_ADDR_BODY) -> tuple[str, str]:
  """Sort a reply frame into ('positive' | 'unsupported' | 'noservice' | 'negative' | 'other', detail).

  'unsupported' is an NRC that means the LID does not exist; 'noservice' means the ECU does not
  speak service 0x30 at all; any other NRC is 'negative', an LID the ECU knows but refused.
  """
  body = probe_body(data, sub_addr)
  if body is None or len(body) < 1:
    return "other", data.hex(" ")
  if body[0] == POSITIVE_IO_CONTROL:
    return "positive", data.hex(" ")
  if body[0] == NEGATIVE_RESPONSE and len(body) >= 3 and body[1] == SERVICE_IO_CONTROL:
    nrc = body[2]
    name = NRC_NAMES.get(nrc, "unknown")
    kind = "unsupported" if nrc in NRC_UNSUPPORTED else ("noservice" if nrc == NRC_SERVICE_NOT_SUPPORTED else "negative")
    return kind, f"NRC 0x{nrc:02X} {name}"
  return "other", data.hex(" ")


def describe_blinkers_state(old: bytes, new: bytes) -> str:
  """What changed in BLINKERS_STATE, by the signals the Toyota DBC names."""
  parts = []
  if len(old) >= 4 and len(new) >= 4:
    if (old[1] ^ new[1]) & 0x80:
      parts.append("event pulse " + ("on" if new[1] & 0x80 else "off"))
    if (old[3] ^ new[3]) & 0x08:
      parts.append("HAZARD_LIGHT=" + str((new[3] >> 3) & 1))
    if (old[3] ^ new[3]) & 0x30:
      turn = {0: "0", 1: "left", 2: "right", 3: "none"}[(new[3] >> 4) & 3]
      parts.append("TURN_SIGNALS=" + turn)
  return ", ".join(parts) if parts else f"{old.hex(' ')} -> {new.hex(' ')}"


@dataclass
class ProbeResult:
  index: int
  lid: int
  control: bytes
  sent_at: float | None = None       # seconds, None until the echo (or the schedule) places it
  replies: list[tuple[str, str]] = field(default_factory=list)
  blinker_changes: list[str] = field(default_factory=list)

  @property
  def verdict(self) -> str:
    if any(kind == "positive" for kind, _ in self.replies):
      return "positive"
    kinds = {kind for kind, _ in self.replies}
    if "negative" in kinds:
      return "negative"
    if "noservice" in kinds:
      return "noservice"
    if "unsupported" in kinds:
      return "unsupported"
    return "no reply" if not self.replies else "other"


class ScanRecorder:
  """Attribute 0x758 replies and 0x614 changes to the probe that caused them.

  Feed it the `can` stream as pandad publishes it, as (nanos, [(addr, data, src), ...]) tuples.
  Probes are placed by their transmit echo; a probe the echo never shows is placed from the script
  timing relative to the first echo, or the first reply if there were no echoes at all.
  """

  def __init__(self, frames: Sequence[ScriptFrame], sub_addr: int | None = SUB_ADDR_BODY,
               tx_addr: int = DIAG_ADDR, rx_addr: int = REPLY_ADDR):
    self.sub_addr = sub_addr
    self.tx_addr = tx_addr
    self.rx_addr = rx_addr
    self.frames = list(frames)
    self.probes: list[ProbeResult] = []
    for i, f in enumerate(self.frames):
      lid = probe_lid(f.data, sub_addr)
      if lid is not None:
        self.probes.append(ProbeResult(i, lid, probe_body(f.data, sub_addr)[2:]))
    self._next_echo = 0
    self.echoes = 0
    self.frames_seen = 0
    self.blinkers_frames = 0
    # Set by the runner: False when pandad never picked the script up, so nothing here means anything.
    self.script_played = True
    # Called with the ProbeResult when its transmit echo arrives: the moment the control is live.
    self.on_sent = None
    self.first_reply_at: float | None = None
    self._last_blinkers: bytes | None = None
    self.blinker_changes: list[tuple[float, str]] = []
    self.unmatched_replies: list[tuple[float, str]] = []

  def _current_probe(self, t: float) -> ProbeResult | None:
    placed = [p for p in self.probes if p.sent_at is not None and p.sent_at <= t]
    return placed[-1] if placed else None

  def update(self, can_msgs: Iterable[tuple[int, Iterable[tuple[int, bytes, int]]]]) -> None:
    for nanos, frames in can_msgs:
      t = nanos / 1e9
      for addr, data, src in frames:
        self.frames_seen += 1
        data = bytes(data)
        if addr == self.tx_addr and src >= 128 and probe_lid(data, self.sub_addr) is not None:
          # The panda echoes what it sent with the bus offset by 128. Probes go out in script
          # order, so the next echo is the next unplaced probe.
          if self._next_echo < len(self.probes):
            self.probes[self._next_echo].sent_at = t
            if self.on_sent is not None:
              self.on_sent(self.probes[self._next_echo])
            self._next_echo += 1
            self.echoes += 1
        elif addr == self.rx_addr and src < 128 and probe_body(data, self.sub_addr) is not None:
          if self.first_reply_at is None:
            self.first_reply_at = t
          probe = self._current_probe(t)
          if probe is None:
            self.unmatched_replies.append((t, data.hex(" ")))
          else:
            probe.replies.append(classify_reply(data, self.sub_addr))
        elif addr == BLINKERS_STATE_ADDR and src < 128:
          self.blinkers_frames += 1
          if self._last_blinkers is not None and self._last_blinkers != data:
            what = describe_blinkers_state(self._last_blinkers, data)
            self.blinker_changes.append((t, what))
            probe = self._current_probe(t)
            if probe is not None:
              probe.blinker_changes.append(what)
          self._last_blinkers = data

  def place_by_schedule(self) -> None:
    """Give every probe still without an echo a time from the script's delays."""
    anchor_index = next((i for i, p in enumerate(self.probes) if p.sent_at is not None), None)
    if anchor_index is None:
      if self.first_reply_at is None:
        return
      anchor_index, anchor_t = 0, self.first_reply_at
    else:
      anchor_t = self.probes[anchor_index].sent_at
    offsets = []
    total = 0
    for f in self.frames:
      total += f.delay_ms
      offsets.append(total / 1000.0)
    base = offsets[self.probes[anchor_index].index]
    for p in self.probes:
      if p.sent_at is None:
        p.sent_at = anchor_t + offsets[p.index] - base

  @property
  def target(self) -> str:
    if self.sub_addr is None:
      return f"0x{self.tx_addr:03X} direct (replies on 0x{self.rx_addr:03X})"
    return f"(0x{self.tx_addr:03X}, 0x{self.sub_addr:02X})"

  def report(self, title: str) -> str:
    known_lids = KNOWN_LIDS.get(self.sub_addr, {})
    known_controls = KNOWN_CONTROLS.get(self.sub_addr, {})

    def label(p: ProbeResult) -> str:
      what = known_controls.get((p.lid, bytes(p.control[:2]))) or known_lids.get(p.lid)
      return f"  [{what}]" if what else ""
    header = f"target {self.target}, {len(self.probes)} probes, {self.echoes} transmit echoes seen, "
    header += f"{self.frames_seen} CAN frames recorded, {self.blinkers_frames} of them BLINKERS_STATE (0x614)"
    lines = [title, header, ""]
    if not self.script_played:
      lines.append(NOT_PLAYED_WARNING)
      lines.append("")
    elif self.echoes == 0:
      lines.append("WARNING: no transmit echoes; probes were placed from the script timing, so attribution is approximate")
      lines.append("")
    positives = [p for p in self.probes if p.verdict == "positive"]
    negatives = [p for p in self.probes if p.verdict == "negative"]
    silent = [p for p in self.probes if p.verdict == "no reply"]
    hits = [p for p in self.probes if p.blinker_changes]
    noservice = [p for p in self.probes if p.verdict == "noservice"]
    if noservice:
      lines.append(f"{len(noservice)} probes answered NRC 0x11 serviceNotSupported: this ECU does not implement service 0x30"
                   + (" at all" if len(noservice) == len(self.probes) else ""))
      lines.append("")
    lines.append("LIDs answering positive (exist, may have actuated):")
    lines += [f"  LID 0x{p.lid:02X} ctrl {p.control.hex(' ')}: {', '.join(d for _, d in p.replies)}"
              + label(p) for p in positives] or ["  none"]
    lines.append("LIDs answering with another negative code (exist, refused):")
    lines += [f"  LID 0x{p.lid:02X} ctrl {p.control.hex(' ')}: {', '.join(d for _, d in p.replies)}" for p in negatives] or ["  none"]
    lines.append("BLINKERS_STATE (0x614) changes during a probe:")
    lines += [f"  LID 0x{p.lid:02X} ctrl {p.control.hex(' ')}: {'; '.join(p.blinker_changes)}" for p in hits] or ["  none"]
    silent_lids = ", ".join(f"0x{p.lid:02X}" for p in silent[:24]) + ("..." if len(silent) > 24 else "")
    lines.append(f"probes with no reply: {len(silent)}" + (f" ({silent_lids})" if silent else ""))
    if self.unmatched_replies:
      lines.append(f"replies before the first placed probe: {len(self.unmatched_replies)}")
    lines.append("")
    lines.append("all probes:")
    for p in self.probes:
      when = f"{p.sent_at:.2f}" if p.sent_at is not None else "?"
      lines.append(f"  #{p.index:3d} t={when:>8} LID 0x{p.lid:02X} ctrl {p.control.hex(' ')}: {p.verdict}"
                   + (f" ({', '.join(d for _, d in p.replies)})" if p.replies else "")
                   + (f" 0x614: {'; '.join(p.blinker_changes)}" if p.blinker_changes else ""))
    return "\n".join(lines)

  def summary(self) -> str:
    """The few lines worth putting on the device screen."""
    positives = [p for p in self.probes if p.verdict == "positive"]
    negatives = [p for p in self.probes if p.verdict == "negative"]
    hits = [p for p in self.probes if p.blinker_changes]
    if not self.script_played:
      return NOT_PLAYED_WARNING
    unanswered = sum(1 for p in self.probes if p.verdict == "no reply")
    noservice = sum(1 for p in self.probes if p.verdict == "noservice")
    lines = [f"{len(self.probes)} probes to {self.target}, {self.echoes} echoes, {unanswered} unanswered"]
    if noservice:
      lines.append(f"service 0x30 not supported: {noservice} probes" + (" (all of them)" if noservice == len(self.probes) else ""))
    lines.append("positive: " + (", ".join(f"0x{p.lid:02X}" for p in positives) if positives else "none"))
    lines.append("refused: " + (", ".join(f"0x{p.lid:02X}" for p in negatives) if negatives else "none"))
    lines.append("0x614 hits: " + ("; ".join(f"0x{p.lid:02X} {p.control.hex(' ')} -> {' / '.join(p.blinker_changes)}" for p in hits) if hits else "none"))
    return "\n".join(lines)


def announce(probe: ProbeResult) -> None:
  """Live line per probe, so whoever is watching the car knows which control is active right now."""
  if probe.control == DEFAULT_CONTROL:
    print(f"  released  (probe #{probe.index})", flush=True)
  else:
    print(f"LID 0x{probe.lid:02X} ctrl {probe.control.hex(' ')}  <- live now  (probe #{probe.index})", flush=True)


def run_script_and_record(frames: Sequence[ScriptFrame], sub_addr: int | None, settle_s: float = 3.0,
                          live: bool = False, tx_addr: int = DIAG_ADDR, rx_addr: int = REPLY_ADDR) -> ScanRecorder:
  """Queue the script for pandad and record the bus until it has had time to finish."""
  import openpilot.cereal.messaging as messaging
  from openpilot.common.params import Params
  from openpilot.selfdrive.pandad import can_capnp_to_list

  recorder = ScanRecorder(frames, sub_addr, tx_addr, rx_addr)
  if live:
    recorder.on_sent = announce
  params = Params()
  # pandad decides offroad from deviceState.started, so ask the same source. If it does not answer
  # in time, go ahead: a script pandad never takes is caught after the run below.
  sm = messaging.SubMaster(["deviceState"])
  sm.update(1000)
  if sm.updated["deviceState"] and sm["deviceState"].started:
    # pandad would drop the script on the spot; say so instead of recording a minute of nothing.
    recorder.script_played = False
    return recorder

  can_sock = messaging.sub_sock("can", conflate=False, timeout=0)
  # Drain what is already queued so the first probe is not matched against stale frames.
  messaging.drain_sock_raw(can_sock)

  params.put("OffroadCanScript", encode_script(frames))
  # pandad polls the param every 100 ms; the rest is the script's own timing.
  deadline = time.monotonic() + 0.5 + script_duration_s(frames) + settle_s
  while time.monotonic() < deadline:
    raw = messaging.drain_sock_raw(can_sock)
    if raw:
      recorder.update(can_capnp_to_list(raw))
    else:
      time.sleep(0.01)
  if params.get("OffroadCanScript"):
    # Still queued after the whole run: pandad never took it (went onroad, or is not running).
    params.remove("OffroadCanScript")
    recorder.script_played = False
    return recorder
  recorder.place_by_schedule()
  return recorder


def report_path(out: str | None) -> str:
  if out:
    return out
  return os.path.join(DEFAULT_REPORT_DIR if os.path.isdir(DEFAULT_REPORT_DIR) else os.getcwd(), REPORT_NAME)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="scan a body ECU for 0x30 local identifiers, or sweep one LID's control bits")
  parser.add_argument("--sub-addr", type=lambda s: int(s, 0), default=SUB_ADDR_BODY, help="0x750 sub-address (default 0x40, main body ECU)")
  parser.add_argument("--direct", type=lambda s: int(s, 0), default=None,
                      help="address an ECU directly instead of through 0x750, e.g. 0x7C0 for the combination meter (replies on address + 8)")
  parser.add_argument("--lid", type=lambda s: int(s, 0), default=None, help="sweep this LID's 16 control bits instead of scanning")
  parser.add_argument("--first", type=lambda s: int(s, 0), default=0, help="first LID to scan")
  parser.add_argument("--last", type=lambda s: int(s, 0), default=0xFF, help="last LID to scan")
  parser.add_argument("--gap-ms", type=int, default=PROBE_GAP_MS, help="ms between scan probes")
  parser.add_argument("--combo", action="store_true", help="sweep selector byte 0x00..0x0F x each action bit instead of single bits")
  parser.add_argument("--low-byte-only", action="store_true", help="sweep only the 8 bits of the second control byte")
  parser.add_argument("--quiet", action="store_true", help="do not print each control as it goes live")
  parser.add_argument("--control", nargs="+", default=None,
                      help="with --lid: send just this control record (hex bytes, e.g. 00 20), hold it --on-ms, then release it")
  parser.add_argument("--repeat", type=int, default=1, help="with --control: how many times to set and release it")
  parser.add_argument("--on-ms", type=int, default=SWEEP_ON_MS, help="ms a sweep control is held before it is released")
  parser.add_argument("--off-ms", type=int, default=SWEEP_OFF_MS, help="ms between a release and the next sweep control")
  parser.add_argument("--out", default=None, help=f"report file (default {DEFAULT_REPORT_DIR}/{REPORT_NAME})")
  args = parser.parse_args(argv)

  if args.direct is not None:
    sub_addr, tx_addr, rx_addr = None, args.direct, args.direct + 8
    target = f"0x{tx_addr:03X} direct"
  else:
    sub_addr, tx_addr, rx_addr = args.sub_addr, DIAG_ADDR, REPLY_ADDR
    target = f"(0x{tx_addr:03X}, 0x{sub_addr:02X})"

  if args.lid is not None and args.control is not None:
    control = bytes(int(b, 16) for b in args.control).ljust(len(DEFAULT_CONTROL), b"\x00")
    frames = build_bit_sweep_frames(args.lid, sub_addr, args.on_ms, args.off_ms, [control] * max(1, args.repeat), tx_addr)
    title = f"control {control.hex(' ')} x{max(1, args.repeat)} on LID 0x{args.lid:02X}"
  elif args.lid is not None:
    controls = combo_controls() if args.combo else sweep_controls()
    if args.low_byte_only and not args.combo:
      controls = controls[:8]
    frames = build_bit_sweep_frames(args.lid, sub_addr, args.on_ms, args.off_ms, controls, tx_addr)
    title = f"{'combo' if args.combo else 'bit'} sweep of LID 0x{args.lid:02X}"
  else:
    frames = build_lid_scan_frames(range(args.first, args.last + 1), sub_addr, args.gap_ms, tx_addr)
    title = f"LID scan 0x{args.first:02X}..0x{args.last:02X}"
  title += f" on {target} at {time.strftime('%Y-%m-%d %H:%M:%S')}"

  print(f"{title}: {len(frames)} frames, about {script_duration_s(frames):.0f} s. Ignition off, watch the car.", flush=True)
  recorder = run_script_and_record(frames, sub_addr, live=not args.quiet, tx_addr=tx_addr, rx_addr=rx_addr)

  path = report_path(args.out)
  with open(path, "w") as f:
    f.write(recorder.report(title) + "\n")
  print(recorder.summary())
  print(f"report: {path}")
  return 0 if recorder.script_played else 1


if __name__ == "__main__":
  raise SystemExit(main())
