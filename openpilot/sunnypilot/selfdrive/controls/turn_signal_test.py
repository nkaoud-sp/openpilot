TURN_SIGNAL_TEST_PARAM = "ToyotaTurnSignalTestRequest"
# left/right/hazard drive the 0x7C0 diagnostic path; cand* are onroad broadcast-candidate replays
# (see TOYOTA_LIGHTING_CANDIDATES in the Toyota carcontroller).
TURN_SIGNAL_TEST_COMMANDS = {"left", "right", "hazard", "cand367", "cand361", "cand2d8"}


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
