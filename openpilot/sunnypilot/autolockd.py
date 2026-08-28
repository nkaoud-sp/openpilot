#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Auto door lock (offroad).

Once the car is switched off and comma is offroad, this waits for the driver to get out
(driver door open then closed), confirms the cabin is empty using the driver-monitoring
camera, and then sends a door-lock CAN frame. If someone is still inside it backs off and
re-checks after the next door open/close, and only locks once the cabin is confirmed empty.

Reused building blocks:
  - doors: BODY_CONTROL_STATE decoded off the raw 'can' stream (works offroad).
  - occupancy: IsDriverViewEnabled spins up the driver camera + DM model; driverStateV2 faceProb.
  - lock send: CanTestTrigger, which pandad fires offroad via ELM327 (the 0x750 lock frame).
"""
import time
from enum import Enum

import openpilot.cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.pandad import can_capnp_to_list

# Toyota-specific door decode (matches the tweaks test panel).
DOOR_DBC = "toyota_nodsu_pt_generated"
DOOR_MSG = "BODY_CONTROL_STATE"
DOOR_BUS = 0
DOOR_SIGNALS = ("DOOR_OPEN_FL", "DOOR_OPEN_FR", "DOOR_OPEN_RL", "DOOR_OPEN_RR")
DRIVER_DOOR = "DOOR_OPEN_FL"  # left-hand-drive: driver = front-left

# Door reads are only trusted this recently; older than this the powertrain bus is asleep.
DOOR_FRESH_S = 2.0

FACE_THRESHOLD = 0.7          # matches selfdrive/monitoring/policy.py _FACE_THRESHOLD
DM_SETTLE_S = 3.0             # let the camera/model settle after it comes alive before sampling
DM_SAMPLE_S = 2.0             # sample the face probabilities over this window, take the max
DM_TIMEOUT_S = 20.0          # give up (and do NOT lock) if the camera never comes up


class State(Enum):
  IDLE = 0                   # disarmed: waiting for offroad + ignition off
  WAIT_DRIVER_OPEN = 1       # armed: waiting for the driver door to open
  WAIT_DRIVER_CLOSE = 2      # driver door opened, waiting for it to close
  WAIT_ALL_CLOSED = 3        # waiting for every door to be closed
  DM_START = 4               # driver camera warming up
  DM_SAMPLE = 5              # sampling face probabilities
  WAIT_ANY_OPEN = 6          # someone was inside, waiting for the next door open
  SEND_LOCK = 7              # cabin empty, fire the lock frame
  DONE = 8                   # locked, stay put until the car is used again


class AutoDoorLock:
  def __init__(self):
    self.params = Params()
    self.sm = messaging.SubMaster(["driverStateV2", "managerState", "pandaStates"])
    self.can_sock = messaging.sub_sock("can", conflate=False, timeout=0)

    self.parser = None
    self.door_last_seen = 0.0

    self.state = State.IDLE
    self.state_t = time.monotonic()
    self.dm_alive_t = 0.0
    self.sample_end_t = 0.0
    self.max_face_prob = 0.0

  # --- helpers -----------------------------------------------------------------
  def _ensure_parser(self):
    if self.parser is not None:
      return
    try:
      from opendbc.can import CANParser
      self.parser = CANParser(DOOR_DBC, [(DOOR_MSG, 0)], DOOR_BUS)
    except Exception:
      cloudlog.exception("autolockd: failed to create door CAN parser")
      self.parser = None

  def _update_doors(self):
    if self.parser is None:
      return
    raw = messaging.drain_sock_raw(self.can_sock)
    if raw and self.parser.update(can_capnp_to_list(raw)):
      self.door_last_seen = time.monotonic()

  @property
  def _doors_fresh(self) -> bool:
    return self.parser is not None and (time.monotonic() - self.door_last_seen) < DOOR_FRESH_S

  def _door_open(self, signal: str) -> bool:
    return bool(self.parser.vl[DOOR_MSG][signal])

  def _all_doors_closed(self) -> bool:
    return not any(self._door_open(s) for s in DOOR_SIGNALS)

  def _any_door_open(self) -> bool:
    return any(self._door_open(s) for s in DOOR_SIGNALS)

  @property
  def _ignition(self) -> bool:
    if not self.sm.alive["pandaStates"]:
      return False
    return any(ps.ignitionLine or ps.ignitionCan for ps in self.sm["pandaStates"])

  @property
  def _dm_ready(self) -> bool:
    running = any(p.name == "dmonitoringd" and p.running for p in self.sm["managerState"].processes)
    return running and self.sm.alive["driverStateV2"]

  def _face_prob(self) -> float:
    # Any face in either seat means the cabin is occupied; no seat mapping needed.
    ds = self.sm["driverStateV2"]
    return max(ds.leftDriverData.faceProb, ds.rightDriverData.faceProb)

  def _set_driver_cam(self, on: bool):
    self.params.put_bool("IsDriverViewEnabled", on)

  def _set_state(self, state: State):
    if state != self.state:
      cloudlog.warning(f"autolockd: {self.state.name} -> {state.name}")
      self.state = state
      self.state_t = time.monotonic()

  # --- main loop ---------------------------------------------------------------
  def update(self):
    self.sm.update(0)
    self._update_doors()

    enabled = self.params.get_bool("AutoDoorLock")

    # Global guards: the feature only runs while offroad (guaranteed by the process condition),
    # enabled, and with ignition off. If any breaks, disarm and release the camera.
    # Ignition coming back on (car used again) also clears a completed lock so it can re-arm.
    if not enabled or self._ignition:
      if self.state != State.IDLE:
        self._set_driver_cam(False)
        self._set_state(State.IDLE)
      return

    if self.state == State.IDLE:
      # Armed once the car is off; wait for the driver to open their door.
      self._set_state(State.WAIT_DRIVER_OPEN)

    elif self.state == State.WAIT_DRIVER_OPEN:
      if self._doors_fresh and self._door_open(DRIVER_DOOR):
        self._set_state(State.WAIT_DRIVER_CLOSE)

    elif self.state == State.WAIT_DRIVER_CLOSE:
      if self._doors_fresh and not self._door_open(DRIVER_DOOR):
        self._set_state(State.WAIT_ALL_CLOSED)

    elif self.state == State.WAIT_ALL_CLOSED:
      if self._doors_fresh and self._all_doors_closed():
        self._set_driver_cam(True)
        self.dm_alive_t = 0.0
        self._set_state(State.DM_START)

    elif self.state == State.DM_START:
      if self._dm_ready:
        if self.dm_alive_t == 0.0:
          self.dm_alive_t = time.monotonic()
        if (time.monotonic() - self.dm_alive_t) > DM_SETTLE_S:
          self.max_face_prob = 0.0
          self.sample_end_t = time.monotonic() + DM_SAMPLE_S
          self._set_state(State.DM_SAMPLE)
      elif (time.monotonic() - self.state_t) > DM_TIMEOUT_S:
        # Camera never came up: fail safe, do not lock.
        cloudlog.error("autolockd: driver camera did not start; aborting without locking")
        self._set_driver_cam(False)
        self._set_state(State.IDLE)

    elif self.state == State.DM_SAMPLE:
      if self._dm_ready:
        self.max_face_prob = max(self.max_face_prob, self._face_prob())
      if time.monotonic() > self.sample_end_t:
        occupied = self.max_face_prob > FACE_THRESHOLD
        self._set_driver_cam(False)
        if occupied:
          cloudlog.warning(f"autolockd: cabin occupied (faceProb={self.max_face_prob:.2f}), waiting")
          self._set_state(State.WAIT_ANY_OPEN)
        elif self._doors_fresh and self._all_doors_closed():
          self._set_state(State.SEND_LOCK)
        else:
          # A door opened during the check: never lock with a door open, re-settle first.
          self._set_state(State.WAIT_ALL_CLOSED)

    elif self.state == State.WAIT_ANY_OPEN:
      if self._doors_fresh and self._any_door_open():
        self._set_state(State.WAIT_ALL_CLOSED)

    elif self.state == State.SEND_LOCK:
      # pandad fires the hardcoded 0x750 lock frame offroad when it sees this trigger.
      self.params.put_bool("CanTestTrigger", True)
      cloudlog.warning("autolockd: cabin empty, sent door lock")
      self._set_state(State.DONE)

    elif self.state == State.DONE:
      pass  # stay locked; the ignition-on guard above re-arms us for next time


def main():
  cloudlog.warning("autolockd: starting")
  auto_lock = AutoDoorLock()
  auto_lock._ensure_parser()
  rk = Ratekeeper(10, print_delay_threshold=None)
  while True:
    auto_lock.update()
    rk.keep_time()


if __name__ == "__main__":
  main()
