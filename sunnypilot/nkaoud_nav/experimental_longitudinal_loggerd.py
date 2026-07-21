#!/usr/bin/env python3
from __future__ import annotations

import time

import cereal.messaging as messaging

from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.nkaoud_nav.experimental_longitudinal_logger import (
  ExperimentalLongitudinalSessionLogger,
  finalize_orphan_sessions,
)

LOGGING_PARAM = "ExperimentalLongitudinalLogging"
POST_EVENT_SECONDS = 5.0
LOG_RATE_HZ = 10.0


def _active(sm) -> bool:
  return bool(sm["deviceState"].started and sm["selfdriveState"].enabled and sm["carControl"].enabled)


def _intervention(sm) -> bool:
  CS = sm["carState"]
  return bool(CS.gasPressed or CS.brakePressed)


def main() -> None:
  params = Params()
  finalize_orphan_sessions()

  services = [
    "deviceState", "selfdriveState", "carState", "carControl", "controlsState",
    "longitudinalPlan", "longitudinalPlanSP", "modelV2", "radarState",
  ]
  sm = messaging.SubMaster(services)
  rk = Ratekeeper(LOG_RATE_HZ)
  logger = ExperimentalLongitudinalSessionLogger()

  active_prev = False
  started_prev = False
  post_until = 0.0

  while True:
    sm.update(0)

    enabled = params.get_bool(LOGGING_PARAM)
    started = bool(sm["deviceState"].started)
    active = enabled and _active(sm)
    disengaged = active_prev and not active

    if active and not logger.is_active:
      logger.start({
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "rate_hz": LOG_RATE_HZ,
        "post_event_seconds": POST_EVENT_SECONDS,
      })

    if active and _intervention(sm):
      post_until = time.monotonic() + POST_EVENT_SECONDS
    if disengaged:
      post_until = time.monotonic() + POST_EVENT_SECONDS

    should_log = logger.is_active and enabled and (active or time.monotonic() < post_until)
    if should_log:
      logger.log(sm, disengaged)

    if logger.is_active and ((started_prev and not started) or (not enabled and not active)):
      try:
        logger.end()
      except Exception:  # noqa: BLE001
        cloudlog.exception("experimental_longitudinal_loggerd: failed to end session")

    active_prev = active
    started_prev = started
    rk.keep_time()


if __name__ == "__main__":
  main()
