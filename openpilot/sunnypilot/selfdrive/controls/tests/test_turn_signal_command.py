from openpilot.sunnypilot.selfdrive.controls.turn_signal_test import TURN_SIGNAL_TEST_PARAM, TurnSignalTestController


class FakeParams:
  def __init__(self):
    self.values = {}

  def get(self, key):
    return self.values.get(key)

  def remove(self, key):
    self.values.pop(key, None)


def test_turn_signal_test_request_holds_command_for_duration():
  params = FakeParams()
  params.values[TURN_SIGNAL_TEST_PARAM] = {
    "signal": "left",
    "durationMs": 1500,
    "requestId": 123,
  }
  controller = TurnSignalTestController()

  assert controller.update(params, 10.0) == "left"
  assert params.get(TURN_SIGNAL_TEST_PARAM) is not None

  assert controller.update(params, 11.49) == "left"
  assert params.get(TURN_SIGNAL_TEST_PARAM) is not None

  assert controller.update(params, 11.5) == "none"
  assert params.get(TURN_SIGNAL_TEST_PARAM) is None


def test_turn_signal_test_request_rejects_unknown_command():
  params = FakeParams()
  params.values[TURN_SIGNAL_TEST_PARAM] = {
    "signal": "high_beams",
    "durationMs": 1500,
    "requestId": 123,
  }
  controller = TurnSignalTestController()

  assert controller.update(params, 10.0) == "none"
  assert params.get(TURN_SIGNAL_TEST_PARAM) is None
