#!/usr/bin/env python3
"""
Experimental Mapbox-based navigation daemon for nkaoud-sp fork.

Phase 1 (this file): skeleton only.
  - Subscribes liveLocationKalman.
  - Publishes nkaoudNavigationSP at the configured rate as a heartbeat.
  - No Mapbox calls, no route, no maneuver logic yet.

Phase 3 will add Mapbox Directions fetch + navInstruction/navRoute publishing.
Phase 6 will fill in maneuverTargetSpeed for longitudinal turn-slowdown.
"""
from __future__ import annotations

import json

import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper


def _read_destination(params: Params) -> dict | None:
  raw = params.get("NkaoudNavDestination")
  if not raw:
    return None
  try:
    d = json.loads(raw)
  except (ValueError, TypeError):
    return None
  if "latitude" not in d or "longitude" not in d:
    return None
  return d


class NkaoudNavd:
  def __init__(self) -> None:
    self.params = Params()
    self.sm = messaging.SubMaster(['liveLocationKalman'])
    self.pm = messaging.PubMaster(['nkaoudNavigationSP'])
    self.rk = Ratekeeper(5.0)  # matches services.py declaration

  def step(self) -> None:
    self.sm.update(0)

    msg = messaging.new_message('nkaoudNavigationSP')
    msg.valid = bool(self.sm['liveLocationKalman'].gpsOK)
    nav = msg.nkaoudNavigationSP

    nav.enabled = self.params.get_bool("NkaoudNavEnabled")
    destination = _read_destination(self.params)
    nav.active = nav.enabled and destination is not None
    nav.onRoute = False  # filled in once route fetching lands in phase 3
    nav.routeId = ""
    nav.rerouting = False
    nav.maneuverTargetSpeed = 0.0
    nav.distanceToManeuver = 0.0
    nav.maneuverType = ""
    nav.maneuverModifier = ""

    self.pm.send('nkaoudNavigationSP', msg)

  def run(self) -> None:
    while True:
      self.step()
      self.rk.keep_time()


def main() -> None:
  NkaoudNavd().run()


if __name__ == "__main__":
  main()
