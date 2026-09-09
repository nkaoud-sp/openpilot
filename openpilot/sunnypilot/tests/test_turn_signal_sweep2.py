from openpilot.sunnypilot.turn_signal_sweep2_commands import (
  MAX_DID,
  active_candidate,
  did_sweep_from,
  discover_candidate,
  discover_sweep,
  parse_2f_response,
)


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
