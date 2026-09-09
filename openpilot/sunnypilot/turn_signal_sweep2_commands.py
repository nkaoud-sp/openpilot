"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Sweep 2.0: UDS service 0x2F (InputOutputControlByIdentifier) DID discovery on the main body ECU
(request 0x750 / response 0x758). This is a DIFFERENT search space than turn_signal_probe_commands.py,
which sweeps KWP2000 service 0x30 with a 1-byte Local Identifier. The Techstream turn-signal active
test uses service 0x2F with a 16-bit DID (0x2911) -- but on 0x7C0, which is speed-gated. This module
asks whether the body ECU at 0x750 exposes a 0x2F DID for the lamps that the 0x7C0 gate does not cover.

Influenced by the user's lexus_turn_did_scan.py, adapted to this branch's offroad ELM327 path:

  - No SocketCAN. Each request is queued through OffroadCanQueue (see autolock_commands.py /
    panda_safety.cc); pandad drains it one frame per 200 ms via ELM327, offroad only.
  - Toyota extended addressing. This car's 0x750 answers the lock/window/mirror commands in the
    form `40 05 30 11 ...` -- a target sub-address byte (0x40 = main body ECU) then the ISO-TP
    single-frame PCI then the service. So a 0x2F request uses the same envelope: `40 <len> 2F ...`,
    NOT the plain `05 2F ...` a gateway-less ISO-TP stack would use.

Safety model (kept from the source script):
  - The blind sweep is DISCOVERY ONLY: it sends ReturnControlToECU (control option 0x00), which asks
    the ECU whether a DID exists WITHOUT actuating anything. A DID that exists answers positive (0x6F)
    or with a "blocked but real" NRC (conditionsNotCorrect, securityAccessDenied, ...). A DID that
    does not exist answers requestOutOfRange / serviceNotSupported. So the sweep can safely walk the
    whole 16-bit space and come back with a shortlist of live DIDs.
  - Actively actuating a discovered DID (control option 0x03) is a separate, explicit, one-DID step --
    never part of the blind sweep -- because a valid DID could drive some other body output.
"""
from collections.abc import Iterator
from dataclasses import dataclass

from openpilot.sunnypilot.autolock_commands import frame_record

# Main body ECU, UDS over Toyota extended addressing.
SWEEP2_REQ_ADDR = 0x750
SWEEP2_RESP_ADDR = 0x758
SWEEP2_BUS = 0
BODY_ECU_SUB_ADDR = 0x40  # same target sub-address the lock/window/mirror commands answer on

IOCTL_BY_ID_SERVICE = 0x2F        # UDS InputOutputControlByIdentifier
POSITIVE_RESPONSE_SID = 0x6F      # 0x2F + 0x40
NEGATIVE_RESPONSE_SID = 0x7F

# Control option records (UDS ISO 14229).
CONTROL_RETURN_TO_ECU = 0x00      # non-actuating: used for the whole blind discovery sweep
CONTROL_SHORT_TERM_ADJUST = 0x03  # actuating: only ever for one explicit DID, parked

# The turn-signal DID Techstream used on 0x7C0. We don't expect the same number on 0x750, but the
# body ECU's IO-control DIDs plausibly cluster near it, so the sweep tries the 0x29xx block first.
KNOWN_7C0_TURN_DID = 0x2911

NRC_NAMES = {
  0x10: "generalReject",
  0x11: "serviceNotSupported",
  0x12: "subFunctionNotSupported",
  0x13: "incorrectMessageLengthOrInvalidFormat",
  0x14: "responseTooLong",
  0x21: "busyRepeatRequest",
  0x22: "conditionsNotCorrect",
  0x24: "requestSequenceError",
  0x31: "requestOutOfRange",
  0x33: "securityAccessDenied",
  0x35: "invalidKey",
  0x36: "exceedNumberOfAttempts",
  0x37: "requiredTimeDelayNotExpired",
  0x70: "uploadDownloadNotAccepted",
  0x78: "responsePending",
  0x7E: "subFunctionNotSupportedInActiveSession",
  0x7F: "serviceNotSupportedInActiveSession",
}

# NRCs that mean "this DID/service path is real, just not usable right now" -- worth reporting as a
# lead. requestOutOfRange (0x31) / serviceNotSupported (0x11) mean the DID simply is not there.
INTERESTING_NRCS = frozenset({0x22, 0x24, 0x33, 0x37, 0x7E, 0x7F})

# Open an extended diagnostic session before sweeping, in the same extended-addressing envelope.
# 10 03 = DiagnosticSessionControl -> extendedDiagnosticSession. Padded to 8 bytes for elm327_tx_hook.
EXTENDED_SESSION_RECORD = frame_record(bytes([BODY_ECU_SUB_ADDR, 0x02, 0x10, 0x03]).ljust(8, b"\x00"),
                                       addr=SWEEP2_REQ_ADDR, bus=SWEEP2_BUS)


@dataclass(frozen=True)
class Did2FCandidate:
  """One 0x2F InputOutputControl request for a single DID, in Toyota extended addressing."""
  did: int
  control: int = CONTROL_RETURN_TO_ECU
  state: int | None = None
  sub_addr: int = BODY_ECU_SUB_ADDR

  @property
  def uds(self) -> bytes:
    # 2F <DID_hi> <DID_lo> <controlOption> [<controlState>]
    body = bytes([IOCTL_BY_ID_SERVICE, (self.did >> 8) & 0xFF, self.did & 0xFF, self.control])
    if self.state is not None:
      body += bytes([self.state])
    return body

  @property
  def payload(self) -> bytes:
    # [sub_addr, isotp_len, <uds...>], padded to 8 -- the same envelope as LOCK/UNLOCK.
    return bytes([self.sub_addr, len(self.uds)]) + self.uds

  @property
  def record(self) -> bytes:
    """12-byte OffroadCanQueue record pandad drains offroad via ELM327.

    The payload is padded to a full 8 bytes: panda's elm327_tx_hook only permits 0x700-0x7FF frames
    at length 8, so a short frame would be dropped (the lock/window commands are all 8 bytes too).
    """
    return frame_record(self.payload.ljust(8, b"\x00"), addr=SWEEP2_REQ_ADDR, bus=SWEEP2_BUS)

  def __str__(self) -> str:
    verb = "TEST" if self.control == CONTROL_SHORT_TERM_ADJUST else "scan"
    return f"DID 0x{self.did:04X} {verb} [{self.payload.hex()}]"


def discover_candidate(did: int, sub_addr: int = BODY_ECU_SUB_ADDR) -> Did2FCandidate:
  """Non-actuating probe for one DID (ReturnControlToECU)."""
  return Did2FCandidate(did, control=CONTROL_RETURN_TO_ECU, sub_addr=sub_addr)


def active_candidate(did: int, sub_addr: int = BODY_ECU_SUB_ADDR) -> Did2FCandidate:
  """Actuating request for one DID (shortTermAdjustment, state on). Parked, one DID, never in a sweep."""
  return Did2FCandidate(did, control=CONTROL_SHORT_TERM_ADJUST, state=0x01, sub_addr=sub_addr)


def release_candidate(did: int, sub_addr: int = BODY_ECU_SUB_ADDR) -> Did2FCandidate:
  """ReturnControlToECU for one DID, to hand a tested output back to the ECU immediately."""
  return Did2FCandidate(did, control=CONTROL_RETURN_TO_ECU, sub_addr=sub_addr)


def did_sweep_order() -> list[int]:
  """Every 16-bit DID, with the 0x29xx block (around the known 0x7C0 turn DID) front-loaded.

  The neighbourhood of the working 0x7C0 DID is the best guess for the body ECU's lighting IO
  controls, so it is tested in the first ~90 s; the rest of the space follows for an exhaustive run.
  """
  head_block = range(KNOWN_7C0_TURN_DID & 0xFF00, (KNOWN_7C0_TURN_DID & 0xFF00) + 0x100)
  head = set(head_block)
  return list(head_block) + [d for d in range(0x10000) if d not in head]


def discover_sweep(sub_addr: int = BODY_ECU_SUB_ADDR) -> Iterator[Did2FCandidate]:
  """The blind discovery sweep: non-actuating ReturnControlToECU across the whole DID space."""
  for did in did_sweep_order():
    yield discover_candidate(did, sub_addr=sub_addr)


@dataclass(frozen=True)
class Sweep2Response:
  kind: str            # "positive" | "negative" | "other" | "empty"
  did: int | None
  nrc: int | None
  description: str

  @property
  def interesting(self) -> bool:
    if self.kind == "positive":
      return True
    return self.kind == "negative" and self.nrc in INTERESTING_NRCS


def parse_2f_response(data: bytes) -> Sweep2Response:
  """Parse a 0x758 reply, tolerating Toyota extended addressing (a leading sub-address byte).

  Single-frame ISO-TP only; a multi-frame positive response is reported verbatim rather than
  reassembled (discovery only needs the SID/NRC, which live in the first frame).
  """
  if not data:
    return Sweep2Response("empty", None, None, "empty")

  b = data
  # Strip a leading target sub-address byte if present: a real ISO-TP PCI has a high nibble of
  # 0..3 (SF/FF/CF/FC), so anything else in byte 0 is Toyota extended addressing.
  if (b[0] >> 4) > 3:
    b = b[1:]
  if not b:
    return Sweep2Response("empty", None, None, "empty (sub-address only)")

  if (b[0] >> 4) != 0:  # not a single frame
    return Sweep2Response("other", None, None, f"non-single-frame: {data.hex(' ')}")

  uds = b[1:1 + (b[0] & 0x0F)]
  if not uds:
    return Sweep2Response("empty", None, None, "no UDS payload")

  sid = uds[0]
  if sid == POSITIVE_RESPONSE_SID:
    did = (uds[1] << 8) | uds[2] if len(uds) >= 3 else None
    where = f" DID 0x{did:04X}" if did is not None else ""
    return Sweep2Response("positive", did, None, f"positive 0x6F{where}")

  if sid == NEGATIVE_RESPONSE_SID and len(uds) >= 3:
    nrc = uds[2]
    return Sweep2Response("negative", None, nrc, f"NRC 0x{nrc:02X} ({NRC_NAMES.get(nrc, 'unknownNRC')})")

  return Sweep2Response("other", None, None, f"SID 0x{sid:02X}: {uds.hex(' ')}")
