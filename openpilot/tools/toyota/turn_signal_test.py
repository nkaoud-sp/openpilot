#!/usr/bin/env python3
"""
Toyota/Lexus turn-signal active-test tool (bench / reverse-engineering).

This replays the UDS "Active Test" that Techstream uses to actuate the turn
signals on a 2019+ Lexus ES / TSS2 Toyota. It was reconstructed from a
Techstream CAN capture of a right-turn active test:

    request  0x7C0  (Combination Meter diagnostic address)
    response 0x7C8
    service  0x2F   InputOutputControlByIdentifier
    DID      0x2911 controlOption 0x03 (shortTermAdjustment)

The two 12-byte payloads seen alternating in that capture were:

    2F 29 11 03 00 00 00 00 00 00 00 08     # state A
    2F 29 11 03 00 00 00 08 00 00 00 08     # state B (extra bit in byte 7)

Sent as a 12-byte ISO-TP message this reproduces the capture's framing exactly
(first frame `10 0C ...` + one consecutive frame `21 ...`).

IMPORTANT
  * This talks to the panda DIRECTLY and puts it in elm327 (diagnostic) safety
    mode, which permits transmit to the 0x700-0x7FF diagnostic range. It does
    NOT go through the running openpilot stack, whose Toyota safety would block
    it. Because the panda can only be claimed by one process, openpilot must be
    STOPPED (or the device offroad with the manager not running) before use,
    otherwise the panda will report as busy.
  * Bench use only. Do the car ON, in park, stationary. Sending diagnostic
    actuator commands to the wrong ECU/DID on the wrong platform can set DTCs.
    Only verified on 2019+ Lexus ES (TSS2). Treat every option other than
    --right as unverified until you confirm it on your own car.
"""
import argparse
import sys
import time

REQUEST_ADDR = 0x7C0   # Combination Meter diagnostic request
RESPONSE_ADDR = 0x7C8  # its response
BUS = 0

# Verified right-turn active-test payloads (exact bytes from the Techstream capture)
RIGHT_A = bytes.fromhex("2F291103" + "0000000000000008")
RIGHT_B = bytes.fromhex("2F291103" + "0000000800000008")
TURN_BITS = {
  "right": 0x08,
  "left": 0x10,
  "hazard": 0x18,
}

# Session / control helpers
EXTENDED_SESSION = bytes.fromhex("1003")   # DiagnosticSessionControl -> extended
DEFAULT_SESSION = bytes.fromhex("1001")    # DiagnosticSessionControl -> default
TESTER_PRESENT = bytes.fromhex("3E00")
RETURN_CONTROL = bytes.fromhex("2F291100")  # InputOutputControl -> returnControlToECU


def _pad(frame: bytes) -> bytes:
  return frame.ljust(8, b"\x00")


def _wait_flow_control(panda, rx: int, timeout: float = 0.3) -> None:
  """Best-effort wait for an ISO-TP flow-control frame (0x3x). Never blocks long."""
  end = time.monotonic() + timeout
  while time.monotonic() < end:
    for addr, dat, bus in panda.can_recv():
      if addr == rx and bus == BUS and len(dat) and (dat[0] & 0xF0) == 0x30:
        return
    time.sleep(0.002)


def isotp_send(panda, payload: bytes, tx: int = REQUEST_ADDR) -> None:
  """Minimal, hang-free ISO-TP transmit (single frame or first-frame + CFs)."""
  n = len(payload)
  if n <= 7:
    panda.can_send(tx, _pad(bytes([n]) + payload), BUS)
    return

  # First frame
  panda.can_send(tx, _pad(bytes([0x10 | (n >> 8), n & 0xFF]) + payload[:6]), BUS)
  rest = payload[6:]
  _wait_flow_control(panda, tx + 8)  # honor FC if the ECU sends it; continue regardless

  idx = 1
  while rest:
    panda.can_send(tx, _pad(bytes([0x20 | (idx & 0xF)]) + rest[:7]), BUS)
    rest = rest[7:]
    idx += 1
    time.sleep(0.005)


def isotp_recv(panda, rx: int = RESPONSE_ADDR, timeout: float = 0.6) -> bytes | None:
  """Bounded ISO-TP receive. Returns the reassembled payload or None on timeout."""
  end = time.monotonic() + timeout
  dat = b""
  need = None
  idx = 1
  while time.monotonic() < end:
    for addr, d, bus in panda.can_recv():
      if addr != rx or bus != BUS or not len(d):
        continue
      pci = d[0] & 0xF0
      if pci == 0x00:  # single frame
        return d[1:1 + (d[0] & 0x0F)]
      if pci == 0x10:  # first frame
        need = ((d[0] & 0x0F) << 8) | d[1]
        dat = d[2:]
        panda.can_send(rx - 8, _pad(b"\x30\x00\x00"), BUS)  # flow control: clear to send
        idx = 1
      elif pci == 0x20 and need is not None:  # consecutive frame
        dat += d[1:]
        idx += 1
        if len(dat) >= need:
          return dat[:need]
    time.sleep(0.002)
  return dat if dat else None


def _describe(resp: bytes | None) -> str:
  if resp is None:
    return "no response"
  if resp and resp[0] == 0x7F:
    return f"negative response (NRC 0x{resp[2]:02X})" if len(resp) >= 3 else "negative response"
  return "OK: " + resp.hex(" ")


def open_panda(elm327: bool = True):
  from panda import Panda
  from opendbc.car.structs import CarParams
  try:
    panda = Panda()
  except Exception as e:  # pragma: no cover - hardware dependent
    print(f"\nERROR: could not open the panda ({e}).", file=sys.stderr)
    print("Is openpilot still running? Stop it first (the panda can only be", file=sys.stderr)
    print("claimed by one process): `tmux kill-session -t comma` / stop the manager.", file=sys.stderr)
    raise SystemExit(1) from e
  if elm327:
    panda.set_safety_mode(CarParams.SafetyModel.elm327)
  panda.can_clear(0xFFFF)
  return panda


def begin_session(panda) -> None:
  isotp_send(panda, EXTENDED_SESSION)
  print(f"  extended session:  {_describe(isotp_recv(panda))}")


def end_session(panda) -> None:
  isotp_send(panda, RETURN_CONTROL)
  isotp_recv(panda)
  isotp_send(panda, DEFAULT_SESSION)
  isotp_recv(panda)
  print("  returned control to ECU, default session restored")


def hold(panda, frames: list[bytes], duration: float, rate: float = 1.0) -> None:
  """Keep the active test alive by re-sending, alternating through `frames`."""
  print(f"  holding for {duration:.0f}s (Ctrl-C to stop early)...")
  end = time.monotonic() + duration
  i = 0
  try:
    while time.monotonic() < end:
      payload = frames[i % len(frames)]
      isotp_send(panda, payload)
      resp = isotp_recv(panda)
      print(f"    -> {payload.hex(' ')}   {_describe(resp)}")
      isotp_send(panda, TESTER_PRESENT)
      isotp_recv(panda)
      i += 1
      time.sleep(rate)
  except KeyboardInterrupt:
    print("\n  interrupted")


def make_payload(byte_index: int, bits: int) -> bytes:
  """Build a `2F 29 11 03` active test with one control-state byte set."""
  state = bytearray(8)
  state[byte_index] = bits
  return bytes.fromhex("2F291103") + bytes(state)


def make_turn_payload(signal: str) -> bytes:
  """Build a verified turn signal active-test payload by signal name."""
  bits = TURN_BITS[signal]
  state = bytearray(8)
  state[3] = bits
  state[7] = bits
  return bytes.fromhex("2F291103") + bytes(state)


def sweep(panda, bits: int, dwell: float) -> None:
  """
  Enumerate the 8 control-state bytes one at a time so you can watch which
  lamp/output each activates. Use this to map left / hazard / etc., then read
  the winning `2F 29 11 03 ...` line back into --payload for a repeatable command.
  """
  print("Sweeping control-state bytes (watch the car):")
  for b in range(8):
    payload = make_payload(b, bits)
    print(f"\n byte[{b}] = 0x{bits:02X}  ->  {payload.hex(' ')}")
    hold(panda, [payload], dwell, rate=0.8)
    isotp_send(panda, RETURN_CONTROL)
    isotp_recv(panda)


def run_sequence(panda) -> None:
  """Run the scripted left/right/hazard timing sequence directly through panda."""
  sequence = (
    ("left", 0.8, 0.4, 5),
    ("right", 0.4, 0.8, 5),
    ("hazard", 0.6, 0.6, 5),
  )
  for signal, on_seconds, off_seconds, repeats in sequence:
    on_payload = make_turn_payload(signal)
    off_payload = bytes.fromhex("2F291103") + bytes([0, 0, 0, 0, 0, 0, 0, TURN_BITS[signal]])
    print(f"\n{signal}: {on_seconds:.1f}s on / {off_seconds:.1f}s off x{repeats}")
    for _ in range(repeats):
      isotp_send(panda, on_payload)
      print(f"    ON  -> {on_payload.hex(' ')}   {_describe(isotp_recv(panda))}")
      time.sleep(on_seconds)
      isotp_send(panda, off_payload)
      print(f"    OFF -> {off_payload.hex(' ')}   {_describe(isotp_recv(panda))}")
      time.sleep(off_seconds)


def run_enqueue(args) -> None:
  """On-device offroad path: hand frames to pandad via Params OffroadCanQueue (no panda claim)."""
  from openpilot.common.params import Params
  from openpilot.sunnypilot.autolock_commands import frame_record, LOCK_CMD, UNLOCK_CMD
  from openpilot.sunnypilot.turn_signal_commands import (
    build_turn_signal_pulse_queue,
    build_turn_signal_queue,
    build_turn_signal_sweep_queue,
    turn_signal_payload,
  )

  if args.sequence:
    Params().put_bool("OffroadTurnSignalSequence", True)
    print("requested turn signal sequence via pandad internal timer")
    return

  if args.lock:
    queue, what = frame_record(LOCK_CMD), "door lock"
  elif args.unlock:
    queue, what = frame_record(UNLOCK_CMD), "door unlock"
  elif args.lock_test:
    queue, what = (frame_record(LOCK_CMD) * 6) + frame_record(UNLOCK_CMD), "door lock + unlock"
  elif args.left:
    queue = build_turn_signal_pulse_queue("left", session=not args.no_session)
    what = "left turn signal"
  elif args.hazard:
    queue = build_turn_signal_pulse_queue("hazard", session=not args.no_session)
    what = "hazard lights"
  elif args.sweep:
    queue = build_turn_signal_sweep_queue(bits=args.bits, session=not args.no_session)
    what = "turn signal byte sweep"
  else:
    if args.payload:
      queue = build_turn_signal_queue(bytes.fromhex(args.payload.replace(" ", "")), session=not args.no_session)
    else:
      queue = build_turn_signal_pulse_queue("right", session=not args.no_session)
    what = "right turn signal"

  Params().put("OffroadCanQueue", queue)
  print(f"queued {what}: {len(queue) // 12} frame(s) written to OffroadCanQueue " +
        "(pandad drains them offroad via ELM327, 200ms apart)")


BANNER = """
============================================================
 Toyota/Lexus TURN SIGNAL active-test  (BENCH USE ONLY)
 Car ON, in PARK, stationary. openpilot must be STOPPED.
 Verified only on 2019+ Lexus ES (TSS2). Others: unverified.
 On a running device use --enqueue instead (offroad).
============================================================"""


def main() -> None:
  p = argparse.ArgumentParser(description="Toyota/Lexus turn-signal CAN active-test (bench)")
  mode = p.add_mutually_exclusive_group()
  mode.add_argument("--right", action="store_true", help="verified right-turn test (replays the capture)")
  mode.add_argument("--left", action="store_true", help="verified left-turn test")
  mode.add_argument("--hazard", action="store_true", help="verified hazard-lights test")
  mode.add_argument("--sequence", action="store_true", help="scripted left/right/hazard timing sequence")
  mode.add_argument("--sweep", action="store_true", help="enumerate control bytes to discover left/hazard/etc.")
  mode.add_argument("--off", action="store_true", help="return control to the ECU (stop any active test)")
  mode.add_argument("--payload", metavar="HEX", help="raw active-test payload, e.g. 2F2911030000000800000008")
  mode.add_argument("--byte", type=int, metavar="N", help="set control-state byte N (0-7); use with --bits")
  mode.add_argument("--lock", action="store_true", help="door lock sanity frame (0x750)")
  mode.add_argument("--unlock", action="store_true", help="door unlock sanity frame (0x750)")
  mode.add_argument("--lock-test", action="store_true", help="lock then unlock (audible panda-TX check)")
  p.add_argument("--enqueue", action="store_true",
                 help="on-device offroad path: write Params OffroadCanQueue for pandad to drain (no panda claim)")
  p.add_argument("--bits", type=lambda x: int(x, 0), default=0x08, help="bit value for --byte/--sweep (default 0x08)")
  p.add_argument("--duration", type=float, default=5.0, help="seconds to hold the test (default 5)")
  p.add_argument("--dwell", type=float, default=2.0, help="seconds per byte in --sweep (default 2)")
  p.add_argument("--rate", type=float, default=1.0, help="keep-alive resend interval (default 1.0s, matches Techstream)")
  p.add_argument("--no-session", action="store_true", help="skip DiagnosticSessionControl 0x1003")
  p.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt")
  args = p.parse_args()

  if args.enqueue:
    run_enqueue(args)
    return

  print(BANNER)
  if not args.yes:
    try:
      if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        print("aborted")
        return
    except EOFError:
      print("aborted (no tty; pass --yes to run non-interactively)")
      return

  panda = open_panda()
  print(f"panda opened, elm327 safety set, tx=0x{REQUEST_ADDR:03X} rx=0x{RESPONSE_ADDR:03X} bus={BUS}")

  if args.off:
    isotp_send(panda, RETURN_CONTROL)
    print(f"  returnControlToECU: {_describe(isotp_recv(panda))}")
    return

  if args.lock or args.unlock or args.lock_test:
    from openpilot.sunnypilot.autolock_commands import LOCK_CMD, UNLOCK_CMD
    if args.lock or args.lock_test:
      panda.can_send(0x750, LOCK_CMD, BUS)
      print(f"  sent door lock:   0x750  {LOCK_CMD.hex(' ')}")
    if args.lock_test:
      time.sleep(1.0)
    if args.unlock or args.lock_test:
      panda.can_send(0x750, UNLOCK_CMD, BUS)
      print(f"  sent door unlock: 0x750  {UNLOCK_CMD.hex(' ')}")
    return

  try:
    if not args.no_session:
      begin_session(panda)

    if args.sweep:
      sweep(panda, args.bits, args.dwell)
    elif args.sequence:
      run_sequence(panda)
    else:
      if args.payload is not None:
        frames = [bytes.fromhex(args.payload.replace(" ", ""))]
      elif args.byte is not None:
        frames = [make_payload(args.byte, args.bits)]
      elif args.left:
        frames = [make_turn_payload("left")]
      elif args.hazard:
        frames = [make_turn_payload("hazard")]
      else:  # default / --right : replay both captured frames
        frames = [RIGHT_A, RIGHT_B]
      hold(panda, frames, args.duration, rate=args.rate)
  finally:
    end_session(panda)


if __name__ == "__main__":
  main()
