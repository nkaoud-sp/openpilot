#!/usr/bin/env python3
"""Auto door lock for offroad Toyota use."""

import time
from enum import Enum

from cereal import messaging
from opendbc.can.parser import CANParser

from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.pandad import can_capnp_to_list
from openpilot.starpilot.autolock_commands import build_queue

DOOR_DBC = "toyota_nodsu_pt_generated"
DOOR_MSG = "BODY_CONTROL_STATE"
DOOR_BUS = 0
DOOR_SIGNALS = ("DOOR_OPEN_FL", "DOOR_OPEN_FR", "DOOR_OPEN_RL", "DOOR_OPEN_RR")
DRIVER_DOOR = "DOOR_OPEN_FL"

DOOR_FRESH_S = 2.0
FACE_THRESHOLD = 0.6
DM_SETTLE_S = 2.0
DM_SAMPLE_S = 2.0
DM_TIMEOUT_S = 20.0


class State(Enum):
  IDLE = 0
  WAIT_DRIVER_OPEN = 1
  WAIT_DRIVER_CLOSE = 2
  WAIT_ALL_CLOSED = 3
  DM_START = 4
  DM_SAMPLE = 5
  WAIT_ANY_OPEN = 6
  SEND_LOCK = 7
  DONE = 8


class AutoDoorLock:
  def __init__(self):
    self.params = Params()
    self.sm = messaging.SubMaster(["driverStateV2", "managerState", "pandaStates"])
    self.can_sock = messaging.sub_sock("can", conflate=False, timeout=0)

    self.parser = None
    self.door_last_seen = 0.0

    self.state = State.IDLE
    self.state_t = time.monotonic()
    self.cam_on = False
    self.dm_live_since = 0.0
    self.sample_end_t = 0.0
    self.max_face_prob = 0.0

  def _ensure_parser(self):
    if self.parser is not None:
      return
    try:
      self.parser = CANParser(DOOR_DBC, [(DOOR_MSG, 3)], bus=DOOR_BUS)
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
    ds = self.sm["driverStateV2"]
    return max(ds.leftDriverData.faceProb, ds.rightDriverData.faceProb)

  def _set_driver_cam(self, on: bool):
    self.cam_on = on
    self.params.put_bool("IsDriverViewEnabled", on)

  def _set_state(self, state: State):
    if state != self.state:
      cloudlog.warning(f"autolockd: {self.state.name} -> {state.name}")
      self.state = state
      self.state_t = time.monotonic()

  def update(self):
    self.sm.update(0)
    self._update_doors()

    if self.cam_on and self._dm_ready:
      if self.dm_live_since == 0.0:
        self.dm_live_since = time.monotonic()
    else:
      self.dm_live_since = 0.0

    enabled = self.params.get_bool("AutoDoorLock")
    if not enabled or self._ignition:
      if self.state != State.IDLE:
        self._set_driver_cam(False)
        self._set_state(State.IDLE)
      return

    if self.state == State.IDLE:
      self._set_state(State.WAIT_DRIVER_OPEN)

    elif self.state == State.WAIT_DRIVER_OPEN:
      if self._doors_fresh and self._door_open(DRIVER_DOOR):
        self._set_driver_cam(True)
        self._set_state(State.WAIT_DRIVER_CLOSE)

    elif self.state == State.WAIT_DRIVER_CLOSE:
      if self._doors_fresh and not self._door_open(DRIVER_DOOR):
        self._set_state(State.WAIT_ALL_CLOSED)

    elif self.state == State.WAIT_ALL_CLOSED:
      if self._doors_fresh and self._all_doors_closed():
        self._set_driver_cam(True)
        self._set_state(State.DM_START)

    elif self.state == State.DM_START:
      if self._dm_ready and self.dm_live_since != 0.0 and (time.monotonic() - self.dm_live_since) > DM_SETTLE_S:
        self.max_face_prob = 0.0
        self.sample_end_t = time.monotonic() + DM_SAMPLE_S
        self._set_state(State.DM_SAMPLE)
      elif (time.monotonic() - self.state_t) > DM_TIMEOUT_S:
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
          self._set_state(State.WAIT_ALL_CLOSED)

    elif self.state == State.WAIT_ANY_OPEN:
      if self._doors_fresh and self._any_door_open():
        self._set_driver_cam(True)
        self._set_state(State.WAIT_ALL_CLOSED)

    elif self.state == State.SEND_LOCK:
      queue = build_queue(
        close_windows=self.params.get_bool("AutoDoorLockCloseWindows"),
        fold_mirrors=self.params.get_bool("AutoDoorLockFoldMirrors"),
      )
      self.params.put("OffroadCanQueue", queue)
      cloudlog.warning(f"autolockd: cabin empty, queued {len(queue) // 12} command(s)")
      self._set_state(State.DONE)

    elif self.state == State.DONE:
      if self._doors_fresh and self._any_door_open():
        self._set_driver_cam(True)
        self._set_state(State.WAIT_ALL_CLOSED)


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
