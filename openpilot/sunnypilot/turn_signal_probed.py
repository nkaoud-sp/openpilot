#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Turn signal discovery probe daemon (offroad).

The UI-driven counterpart to turn_signal_probe.py, so the probe can be run from the Tweaks menu
without SSH. It is started by the manager only while offroad and only while a request is pending
(see turn_signal_probe gate in process_config.py), runs the requested probe once, publishes progress
and results through the TurnSignalProbeStatus param, then clears the request so the manager stops it.

Params:
  TurnSignalProbeRequest  (written by the UI)   {"mode": "shortlist"|"full", "requestId": <int>}
  TurnSignalProbeStatus   (written by this daemon, read by the UI)
      {"state": baseline|running|done|aborted|error, "index", "total", "message", "lastCandidate",
       "hits": [str, ...], "requestId"}

Stopping: the UI removes TurnSignalProbeRequest (or writes a new requestId). This daemon polls the
request between candidates and aborts cleanly when its own requestId is gone or has changed; the
manager also stops it once the request param is gone.
"""
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.turn_signal_probe_commands import full_sweep, shortlist
from openpilot.sunnypilot.turn_signal_probe import (
  STATE_BASELINE,
  STATE_ERROR,
  TurnSignalProbe,
  make_status,
  resolve_dbc,
  run_probe,
)

REQUEST_PARAM = "TurnSignalProbeRequest"
STATUS_PARAM = "TurnSignalProbeStatus"


def _read_request(params: Params) -> dict | None:
  # JSON params decode to a dict (or None when unset) on get().
  request = params.get(REQUEST_PARAM)
  return request if isinstance(request, dict) else None


def main() -> None:
  cloudlog.warning("turn_signal_probed: starting")
  params = Params()

  request = _read_request(params)
  if request is None:
    cloudlog.warning("turn_signal_probed: no request, exiting")
    return

  request_id = request.get("requestId")
  mode = request.get("mode", "shortlist")

  def publish(status: dict) -> None:
    status["requestId"] = request_id
    params.put(STATUS_PARAM, status)  # JSON param: put serializes the dict

  def request_gone() -> bool:
    # Stop if the UI cleared the request or queued a different one.
    current = _read_request(params)
    return current is None or current.get("requestId") != request_id

  def clear_request() -> None:
    # Only clear our own request, never a newer one the UI may have just queued.
    if not request_gone():
      params.remove(REQUEST_PARAM)

  try:
    probe = TurnSignalProbe(resolve_dbc(params, None))
  except Exception:
    cloudlog.exception("turn_signal_probed: failed to init probe")
    publish(make_status(STATE_ERROR, 0, 0, [], "Probe init failed. Is this a Toyota/Lexus?"))
    clear_request()
    return

  if not probe._offroad:
    publish(make_status(STATE_ERROR, 0, 0, [], "Car is not offroad; cannot probe."))
    clear_request()
    return

  publish(make_status(STATE_BASELINE, 0, 0, [], "Reading BLINKERS_STATE baseline..."))
  if not probe.capture_baseline():
    publish(make_status(STATE_ERROR, 0, 0, [],
                        "No BLINKERS_STATE seen. Open the driver door to wake the body bus, then retry."))
    clear_request()
    return

  candidates = list(full_sweep()) if mode == "full" else shortlist()
  cloudlog.warning(f"turn_signal_probed: mode={mode} candidates={len(candidates)}")

  # Abort between candidates if the user stops the run or the car goes onroad.
  run_probe(probe, candidates, report=publish,
            should_abort=lambda: request_gone() or not probe._offroad)

  cloudlog.warning("turn_signal_probed: done")
  clear_request()


if __name__ == "__main__":
  main()
