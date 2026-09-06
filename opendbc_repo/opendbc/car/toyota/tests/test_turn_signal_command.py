from opendbc.car import structs
from opendbc.car.toyota.carcontroller import get_turn_signal_command, toyota_turn_signal_sequence


def test_typed_turn_signal_commands_emit_expected_sequences():
  expected = {
    "left": ["0210030000000000", "100c2f2911030000", "2100100000001000"],
    "right": ["0210030000000000", "100c2f2911030000", "2100080000000800"],
    "hazard": ["0210030000000000", "100c2f2911030000", "2100180000001800"],
  }

  for command, frames in expected.items():
    control_sp = structs.CarControlSP(turnSignalCommand=command)
    assert get_turn_signal_command(control_sp) == command
    assert [frame.hex() for frame in toyota_turn_signal_sequence(command, was_active=False)] == frames


def test_turn_signal_none_clears_and_returns_control():
  control_sp = structs.CarControlSP(turnSignalCommand="none")

  assert get_turn_signal_command(control_sp) == "none"
  assert [frame.hex() for frame in toyota_turn_signal_sequence("none", was_active=True)] == [
    "100c2f2911030000",
    "2100000000001800",
    "042f291100000000",
  ]


def test_lighting_candidates_do_not_trigger_the_diagnostic_turn_signal_path():
  from opendbc.car.toyota.carcontroller import TOYOTA_LIGHTING_CANDIDATES
  for cand in TOYOTA_LIGHTING_CANDIDATES:
    control_sp = structs.CarControlSP(turnSignalCommand=cand)
    # A candidate command must NOT be seen as a turn signal, so the 0x7C0 sequence stays empty.
    assert get_turn_signal_command(control_sp) == "none"
    assert toyota_turn_signal_sequence(get_turn_signal_command(control_sp), was_active=False) == []
  # The prime suspect is the 2-byte 0x367 = 08 80 frame.
  assert TOYOTA_LIGHTING_CANDIDATES["cand367"] == (0x367, b"\x08\x80")
