#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Turn signal discovery probe (offroad).

Finds a non-diagnostic-looking body-ECU command that actuates the exterior turn signals, so the
turn-signal feature can eventually drive the lamps through a path that is not the speed-gated UDS
active test on 0x7C0.

How it works, reusing the auto-lock machinery already on this branch:
  - send: each candidate body-ECU command is queued one at a time through the OffroadCanQueue param,
    which pandad drains offroad via ELM327 (autolock_commands.py / panda_safety.cc). This only works
    OFFROAD -- pandad ignores the queue onroad, where the Toyota safety mode is active.
  - detect: BLINKERS_STATE (0x614) is decoded off the raw 'can' stream. If the lamps leave their idle
    state during the observe window after a command, that candidate is a hit.

Operating conditions:
  - Car OFFROAD (ignition off), gear in P, parked. This is a discovery tool, not the final feature.
  - The body CAN bus must be awake enough to broadcast 0x614 -- open the driver door first, or run
    right after switching off. If no 0x614 is seen at all, the probe says so instead of guessing.
  - For a FULL sweep: empty car, all windows fully DOWN, nobody near the mirrors. The sweep is blind
    and can hit body outputs other than the lamps.

Usage (on device, offroad):
  python -m openpilot.sunnypilot.turn_signal_probe                 # safe shortlist
  python -m openpilot.sunnypilot.turn_signal_probe --full --yes    # blind sweep (empty car!)
  python -m openpilot.sunnypilot.turn_signal_probe --dry-run       # print candidates, send nothing
"""
import argparse
import sys
import time
from collections.abc import Callable

import openpilot.cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.pandad import can_capnp_to_list
from openpilot.sunnypilot.turn_signal_probe_commands import (
  Candidate,
  SignalDetector,
  BODY_ECU_SUB_ADDR,
  full_sweep,
  shortlist,
)

BLINKERS_MSG = "BLINKERS_STATE"       # 0x614 / 1556
BLINKERS_BUS = 0
DEFAULT_DBC = "toyota_nodsu_pt_generated"  # same default autolockd uses; over‑ridable with --dbc

SEND_DRAIN_S = 1.0        # time to let pandad clock the queued frame out (gap is 200ms)
OBSERVE_S = 2.5           # watch BLINKERS_STATE this long after each command
BASELINE_S = 3.0          # capture the idle baseline over this long before probing (0x614 is bursty)
RESULTS_PATH = "/data/turn_signal_probe_results.txt"


def resolve_dbc(params: Params, override: str | None) -> str:
  if override:
    return override
  # Best effort: derive the powertrain DBC from the live CarParams so secoc/tnga cars work too.
  try:
    from openpilot.cereal import car
    from opendbc.car import Bus
    from opendbc.car.toyota.values import DBC
    raw = params.get("CarParams")
    if raw:
      cp = car.CarParams.from_bytes(raw)
      dbc = DBC.get(cp.carFingerprint, {}).get(Bus.pt)
      if dbc:
        return dbc
  except Exception:
    cloudlog.exception("turn_signal_probe: could not derive DBC from CarParams, using default")
  return DEFAULT_DBC


class TurnSignalProbe:
  def __init__(self, dbc: str):
    self.params = Params()
    self.can_sock = messaging.sub_sock("can", conflate=False, timeout=0)

    from opendbc.can import CANParser
    self.parser = CANParser(dbc, [(BLINKERS_MSG, 0)], BLINKERS_BUS)
    self.detector = SignalDetector()
    self.saw_blinkers = False

  # --- CAN plumbing ----------------------------------------------------------
  def _pump(self) -> None:
    raw = messaging.drain_sock_raw(self.can_sock)
    if raw and self.parser.update(can_capnp_to_list(raw)):
      self.saw_blinkers = True

  def drain_frames(self) -> list[tuple[int, int, bytes]]:
    """Raw (bus, addr, data) frames since the last call, for the diff capture."""
    out: list[tuple[int, int, bytes]] = []
    raw = messaging.drain_sock_raw(self.can_sock)
    for _, frames in can_capnp_to_list(raw):
      for addr, dat, src in frames:
        out.append((int(src), int(addr), bytes(dat)))
    return out

  def _read_signals(self) -> tuple[int, int]:
    vl = self.parser.vl[BLINKERS_MSG]
    return int(vl["TURN_SIGNALS"]), int(vl["HAZARD_LIGHT"])

  @property
  def _offroad(self) -> bool:
    # IsOffroad is the authoritative offroad flag the manager writes on every onroad/offroad
    # transition. Reading it is immediate and needs no SubMaster warm-up (a fresh SubMaster hasn't
    # received pandaStates yet, so checking ignition here would spuriously read "onroad"). The
    # OffroadCanQueue path is a no-op onroad regardless.
    return self.params.get_bool("IsOffroad")

  # --- probe steps -----------------------------------------------------------
  def _observe(self, duration: float) -> tuple[bool, int, int]:
    """Pump CAN for `duration`, return (hit, last_turn, last_hazard)."""
    end = time.monotonic() + duration
    hit = False
    turn = hazard = 0
    while time.monotonic() < end:
      self._pump()
      turn, hazard = self._read_signals()
      if self.detector.is_hit(turn, hazard):
        hit = True
      time.sleep(0.01)
    return hit, turn, hazard

  def capture_baseline(self) -> bool:
    """Watch the idle bus, set the baseline. False if BLINKERS_STATE never showed up."""
    end = time.monotonic() + BASELINE_S
    turn = hazard = 0
    while time.monotonic() < end:
      self._pump()
      turn, hazard = self._read_signals()
      time.sleep(0.01)
    if not self.saw_blinkers:
      return False
    self.detector.set_baseline(turn, hazard)
    if turn in (1, 2) or hazard:
      cloudlog.warning(f"turn_signal_probe: baseline is NOT idle (turn={turn} hazard={hazard}); " +
                       "turn the signals off before probing")
    return True

  def send(self, cand: Candidate) -> None:
    self.params.put("OffroadCanQueue", cand.record)

  def probe_one(self, cand: Candidate) -> bool:
    self.send(cand)
    # Let pandad clock it out, then look for the lamps to move.
    drain_end = time.monotonic() + SEND_DRAIN_S
    while time.monotonic() < drain_end:
      self._pump()
      time.sleep(0.01)
    hit, turn, hazard = self._observe(OBSERVE_S)
    return hit


# Probe states, shared with the daemon and reported through TurnSignalProbeStatus.
STATE_BASELINE = "baseline"
STATE_RUNNING = "running"
STATE_ACTIVE = "active"      # capture: recording while the user operates the controls
STATE_DONE = "done"
STATE_ABORTED = "aborted"
STATE_ERROR = "error"

# Diff-capture range: all standard 11-bit addresses (0x000-0x7FF), so it covers powertrain, body,
# AND the diagnostic 0x7xx range (e.g. 0x750). High-churn powertrain IDs are dropped by the noise
# filter below, leaving event-driven body/lighting frames. Widened from 0x600-0x6FF, which missed
# anything on 0x7xx.
CAPTURE_ADDR_LO = 0x000
CAPTURE_ADDR_HI = 0x7FF
CAPTURE_NOISE_THRESHOLD = 4  # an addr with more distinct idle payloads than this is a counter; skip
CAPTURE_MAX_PAYLOADS_PER_ADDR = 4  # cap how many distinct new payloads we list per address
CAPTURE_MAX_LINES = 150          # cap total reported lines (the result dialog scrolls to show them)


def make_status(state: str, index: int, total: int, hits: list,
                message: str = "", last_candidate: str = "") -> dict:
  """The JSON status the UI reads: coarse state, progress, the current candidate, and any hits."""
  return {
    "state": state,
    "index": index,
    "total": total,
    "message": message,
    "lastCandidate": last_candidate,
    "hits": [str(h) for h in hits],
  }


def run_probe(probe: "TurnSignalProbe", candidates: list[Candidate],
              report: Callable[[dict], None] | None = None,
              should_abort: Callable[[], bool] | None = None,
              start: int = 0, prior_hits: list[str] | None = None) -> list[str]:
  """Send each candidate and watch for a signal actuation. Shared by the CLI and the daemon.

  `report` receives a status dict per candidate (and on finish/abort); `should_abort` is polled
  between candidates so a caller can stop the run (car went onroad, user pressed Stop). `start`
  skips the first N candidates (resume a long sweep across sessions) while still reporting absolute
  index/total; `prior_hits` seeds the hit list with hits carried over from an earlier session.
  """
  hits: list[str] = list(prior_hits) if prior_hits else []
  total = len(candidates)
  start = max(0, min(start, total))
  for i in range(start, total):
    cand = candidates[i]
    if should_abort is not None and should_abort():
      if report is not None:
        report(make_status(STATE_ABORTED, i, total, hits, "aborted", str(cand)))
      return hits
    hit = probe.probe_one(cand)
    if hit:
      hits.append(str(cand))
    if report is not None:
      report(make_status(STATE_RUNNING, i + 1, total, hits, "hit" if hit else "", str(cand)))
  if report is not None:
    report(make_status(STATE_DONE, total, total, hits, "done"))
  return hits


def run_capture(probe: "TurnSignalProbe", report: Callable[[dict], None],
                should_abort: Callable[[], bool] | None = None,
                baseline_s: float = 25.0, active_s: float = 30.0) -> list[str]:
  """Diff-capture the bus: learn the idle bus, then flag frames that change.

  Records every distinct payload per (bus, addr) in the capture range while idle, then during an
  active window flags addresses that take a payload not seen at idle. A longer baseline sees more of
  each cyclic message's natural variation, so fewer normal frames get mis-flagged as changes.
  Addresses that were already cycling through many payloads at idle (counters) are dropped as noise;
  a lamp command shows up either as a brand-new address or a new value on an otherwise-stable one.
  """
  baseline: dict[tuple[int, int], set[str]] = {}
  t_end = time.monotonic() + baseline_s
  report(make_status(STATE_BASELINE, 0, 0, [], "Hold still - recording the idle bus..."))
  while time.monotonic() < t_end:
    if should_abort is not None and should_abort():
      report(make_status(STATE_ABORTED, 0, 0, [], "aborted"))
      return []
    for bus, addr, dat in probe.drain_frames():
      if CAPTURE_ADDR_LO <= addr <= CAPTURE_ADDR_HI:
        baseline.setdefault((bus, addr), set()).add(dat.hex())
    time.sleep(0.005)

  # Counters/cyclic messages churn through many idle payloads; ignore them.
  noisy = {k for k, v in baseline.items() if len(v) > CAPTURE_NOISE_THRESHOLD}

  changes: dict[tuple[int, int], set[str]] = {}
  t_end = time.monotonic() + active_s
  report(make_status(STATE_ACTIVE, 0, 0, [], "Now operate hazards / turn stalk / fob lock..."))
  while time.monotonic() < t_end:
    if should_abort is not None and should_abort():
      break
    for bus, addr, dat in probe.drain_frames():
      key = (bus, addr)
      if not (CAPTURE_ADDR_LO <= addr <= CAPTURE_ADDR_HI) or key in noisy:
        continue
      h = dat.hex()
      seen = baseline.get(key)
      if seen is None or h not in seen:
        changes.setdefault(key, set()).add(h)
    report(make_status(STATE_ACTIVE, len(changes), 0, _capture_lines(changes, baseline),
                       "recording"))
    time.sleep(0.02)

  lines = _capture_lines(changes, baseline)
  report(make_status(STATE_DONE, len(changes), 0, lines, "done"))
  return lines


def _capture_lines(changes: dict[tuple[int, int], set[str]],
                   baseline: dict[tuple[int, int], set[str]]) -> list[str]:
  """Format the diff, brand-new addresses first (the strongest command candidates)."""
  def sort_key(item):
    (bus, addr), _ = item
    return (0 if (bus, addr) not in baseline else 1, bus, addr)

  lines: list[str] = []
  for (bus, addr), payloads in sorted(changes.items(), key=sort_key):
    tag = "NEW-ID" if (bus, addr) not in baseline else "NEW-VAL"
    for h in sorted(payloads)[:CAPTURE_MAX_PAYLOADS_PER_ADDR]:
      lines.append(f"b{bus} 0x{addr:X} {h} [{tag}]")
  # NEW-IDs already sort first; cap so a result dialog whose OK button overlaps can't bury a hit.
  if len(lines) > CAPTURE_MAX_LINES:
    lines = lines[:CAPTURE_MAX_LINES] + [f"... (+{len(lines) - CAPTURE_MAX_LINES} more)"]
  return lines


def run(candidates: list[Candidate], dbc: str, dry_run: bool) -> int:
  if dry_run:
    print(f"# dry run: {len(candidates)} candidate(s), DBC={dbc}, nothing sent\n")
    for c in candidates:
      print(c)
    return 0

  probe = TurnSignalProbe(dbc)

  if not probe._offroad:
    print("REFUSING: car is not offroad. The OffroadCanQueue path is a no-op onroad " +
          "(pandad clears it and the Toyota safety mode blocks 0x750). Switch the car off and retry.")
    return 2

  print(f"turn_signal_probe: DBC={dbc}, {len(candidates)} candidate(s)")
  print("capturing idle baseline from BLINKERS_STATE (0x614)...")
  if not probe.capture_baseline():
    print("REFUSING: no BLINKERS_STATE (0x614) seen on bus 0. The body bus is asleep -- open the " +
          "driver door (or probe right after switch-off) so the BCM broadcasts, then retry.")
    return 2

  with open(RESULTS_PATH, "w") as f:
    f.write(f"# turn_signal_probe results  dbc={dbc}  candidates={len(candidates)}\n")
    f.flush()

    def report(status: dict) -> None:
      if status["state"] != STATE_RUNNING:
        return
      cand, hit = status["lastCandidate"], status["message"] == "hit"
      line = f"[{status['index']:>4}/{status['total']}] {cand}{'  <<< HIT' if hit else ''}"
      print(line)
      f.write(line + "\n")
      f.flush()

    # Bail out if the car comes back onroad mid-run; we must not keep poking the bus while driving.
    hits = run_probe(probe, candidates, report=report, should_abort=lambda: not probe._offroad)

  print("\n" + "=" * 60)
  if hits:
    print(f"{len(hits)} candidate(s) actuated the signals:")
    for c in hits:
      print(f"  {c}")
    print(f"\nfull log: {RESULTS_PATH}")
  else:
    print("no candidate moved BLINKERS_STATE.")
    print("next: try --full --yes (empty car, windows down), or confirm the bus was awake.")
  return 0 if hits else 1


def main() -> int:
  p = argparse.ArgumentParser(description="Offroad turn-signal command discovery probe.")
  p.add_argument("--full", action="store_true", help="blind sweep of every LID x bit (empty car!)")
  p.add_argument("--yes", action="store_true", help="confirm you understand the --full risks")
  p.add_argument("--sub-addr", type=lambda x: int(x, 0), default=BODY_ECU_SUB_ADDR,
                 help="ECU sub-address for --full (default 0x40, main body ECU)")
  p.add_argument("--include-motion", action="store_true",
                 help="don't skip window/mirror LIDs in --full (moves physical parts!)")
  p.add_argument("--dbc", default=None, help="override the BLINKERS_STATE DBC (default: from CarParams)")
  p.add_argument("--dry-run", action="store_true", help="print candidates and exit, send nothing")
  args = p.parse_args()

  if args.full and not args.yes:
    print("--full is a blind sweep that can actuate body outputs other than the lamps " +
          "(horn, alarm, trunk, and -- with --include-motion -- windows/mirrors).\n" +
          "Park empty, put every window fully DOWN, keep clear, then re-run with --yes.")
    return 2

  if args.full:
    candidates = list(full_sweep(sub_addr=args.sub_addr, skip_motion=not args.include_motion))
  else:
    candidates = shortlist()

  dbc = resolve_dbc(Params(), args.dbc)
  return run(candidates, dbc, args.dry_run)


if __name__ == "__main__":
  sys.exit(main())
