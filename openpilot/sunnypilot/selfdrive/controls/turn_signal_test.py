TURN_SIGNAL_TEST_PARAM = "ToyotaTurnSignalTestRequest"
TURN_SIGNAL_TEST_COMMANDS = {"left", "right", "hazard"}


class TurnSignalTestController:
  def __init__(self):
    self.request_id: int | None = None
    self.command = "none"
    self.end_time = 0.0

  def update(self, params, now: float) -> str:
    request = params.get(TURN_SIGNAL_TEST_PARAM)

    if isinstance(request, dict):
      signal = request.get("signal")
      request_id = request.get("requestId")
      duration_ms = request.get("durationMs", 0)
      if signal in TURN_SIGNAL_TEST_COMMANDS and request_id != self.request_id:
        self.request_id = request_id
        self.command = signal
        self.end_time = now + max(int(duration_ms), 0) / 1000.0
      elif signal not in TURN_SIGNAL_TEST_COMMANDS:
        params.remove(TURN_SIGNAL_TEST_PARAM)

    if self.command != "none" and now >= self.end_time:
      self.command = "none"
      params.remove(TURN_SIGNAL_TEST_PARAM)

    return self.command
