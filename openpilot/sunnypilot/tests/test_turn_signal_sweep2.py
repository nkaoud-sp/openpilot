import contextlib

from openpilot.sunnypilot.turn_signal_sweep2_commands import (
  MAX_DID,
  active_candidate,
  did_sweep_from,
  discover_candidate,
  discover_sweep,
  parse_2f_response,
)


class _FakeParams:
  def __init__(self):
    self.puts: list[tuple[str, object]] = []

  def put(self, key, value):
    self.puts.append((key, value))


class _FakeProbe:
  """Minimal stand-in for TurnSignalProbe: run_sweep2 only uses params.put and drain_frames."""
  def __init__(self, frames_fn):
    self.params = _FakeParams()
    self._frames_fn = frames_fn

  def drain_frames(self):
    return self._frames_fn()


@contextlib.contextmanager
def _fast_sweep2(**overrides):
  # Shrink the timing/threshold constants so the loop runs instantly and trips its limits quickly.
  import openpilot.sunnypilot.turn_signal_probe as tsp
  defaults = {"SWEEP2_SEND_SETTLE_S": 0.0, "SWEEP2_RESP_WINDOW_S": 0.0}
  saved = {}
  try:
    for name, value in {**defaults, **overrides}.items():
      saved[name] = getattr(tsp, name)
      setattr(tsp, name, value)
    yield tsp
  finally:
    for name, value in saved.items():
      setattr(tsp, name, value)


def test_discover_frame_is_non_actuating_toyota_envelope():
  # 40 (sub-addr) 04 (len) 2F <hi> <lo> 00 (ReturnControlToECU), padded to 8.
  cand = discover_candidate(0x2911)
  assert cand.payload.hex() == "40042f291100"
  # elm327_tx_hook needs len 8: record carries addr 0x750, bus 0, dlc 8, payload padded to 8.
  assert cand.record[:4].hex() == "07500008"
  assert cand.record[4:].hex() == "40042f291100" + "0000"
  assert cand.control == 0x00
  assert cand.state is None


def test_active_frame_carries_short_term_adjust_and_state():
  # 40 05 2F <hi> <lo> 03 01 -- the only actuating form, one DID, parked.
  cand = active_candidate(0x2911)
  assert cand.payload.hex() == "40052f29110301"
  assert cand.control == 0x03
  assert cand.state == 0x01


def test_sweep_from_zero_covers_whole_space_linearly():
  order = did_sweep_from(0)
  assert len(order) == 0x10000
  assert order[0] == 0x0000 and order[-1] == MAX_DID
  assert order == list(range(0x10000))  # linear, so position == DID


def test_sweep_from_a_did_starts_there_and_runs_to_the_end():
  order = did_sweep_from(0x2900)
  assert order[0] == 0x2900
  assert order[-1] == MAX_DID
  assert len(order) == MAX_DID - 0x2900 + 1
  # the candidate stream begins at the selected DID, so a run resumes exactly where asked
  first = next(iter(discover_sweep(0x2900)))
  assert first.did == 0x2900


def test_sweep_start_did_is_clamped():
  assert did_sweep_from(-5)[0] == 0x0000
  assert did_sweep_from(0x20000) == [MAX_DID]


def test_parse_positive_response_with_sub_address():
  # 40 (sub-addr echo) 03 (len) 6F 29 11  -> positive 0x2F response for DID 0x2911.
  resp = parse_2f_response(bytes.fromhex("40036f2911"))
  assert resp.kind == "positive"
  assert resp.did == 0x2911
  assert resp.interesting


def test_parse_negative_response_conditions_not_correct_is_interesting():
  # 40 03 7F 2F 22 -> NRC 0x22 conditionsNotCorrect: DID exists but not usable now.
  resp = parse_2f_response(bytes.fromhex("40037f2f22"))
  assert resp.kind == "negative"
  assert resp.nrc == 0x22
  assert resp.interesting


def test_parse_request_out_of_range_is_not_interesting():
  # 40 03 7F 2F 31 -> NRC 0x31 requestOutOfRange: the DID simply is not there.
  resp = parse_2f_response(bytes.fromhex("40037f2f31"))
  assert resp.kind == "negative"
  assert resp.nrc == 0x31
  assert not resp.interesting


def test_parse_plain_isotp_without_sub_address():
  # Same positive response but framed as plain single-frame ISO-TP (no sub-address byte).
  resp = parse_2f_response(bytes.fromhex("036f2911"))
  assert resp.kind == "positive"
  assert resp.did == 0x2911


def test_run_sweep2_pauses_when_bus_goes_silent():
  # An unattended run must stop, not grind through and skip, when the body bus stops answering. With
  # nothing ever on 0x758, it aborts after SWEEP2_SLEEP_ABORT misses and resumes at the first silent
  # DID (0 here, since it never answered) rather than counting the silent run as scanned.
  with _fast_sweep2(SWEEP2_SLEEP_ABORT=5) as tsp:
    probe = _FakeProbe(list)  # never answers (list() -> [])
    statuses: list[dict] = []
    hits = tsp.run_sweep2(probe, list(discover_sweep(0))[:50], report=statuses.append)
    assert hits == []
    assert statuses[-1]["state"] == "aborted"
    assert statuses[-1]["index"] == 0          # resume at the first unanswered DID
    assert len(statuses) < 50                    # stopped early, did not walk all 50


def test_run_sweep2_checkpoints_and_collects_hits():
  # When the ECU answers every DID, it never pauses, records a hit per DID, and checkpoints the
  # resume count every SWEEP2_CHECKPOINT_EVERY DIDs so a power-off resumes near where it stopped.
  positive = bytes.fromhex("40036f2911")  # positive 0x2F response with a sub-address byte
  with _fast_sweep2(SWEEP2_RESP_WINDOW_S=0.02, SWEEP2_SLEEP_ABORT=10_000, SWEEP2_CHECKPOINT_EVERY=3) as tsp:
    probe = _FakeProbe(lambda: [(0, 0x758, positive)])
    saved: list[int] = []
    hits = tsp.run_sweep2(probe, list(discover_sweep(0))[:7], report=lambda s: None, checkpoint=saved.append)
    assert len(hits) == 7          # every DID answered positive -> a lead each
    assert saved == [3, 6]         # checkpoint at completed-counts 3 and 6
