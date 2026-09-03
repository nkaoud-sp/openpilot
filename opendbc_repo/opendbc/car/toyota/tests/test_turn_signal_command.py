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
