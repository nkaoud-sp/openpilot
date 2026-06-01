#!/usr/bin/env python3
# Secure-on-exit daemon.
#
# After the driver leaves the car, lock the doors and (optionally) fold the
# mirrors and close the windows. "Left the car" means: the ignition is off, no
# face is seen in the driver-monitoring camera, and every door is closed, for a
# configurable number of seconds (LockDoorsTimer).
#
# Ported from FrogPilot (frogpilot/common/frogpilot_utilities.py) onto the
# sunnypilot tree. The raw Toyota diagnostic CAN commands originate from
# AlexandreSato. Toyota/Lexus only.
import time
from typing import NoReturn

import cereal.messaging as messaging
from cereal import log
from opendbc.can.parser import CANParser
from opendbc.car.structs import CarParams
from openpilot.common.params import Params
from openpilot.common.realtime import DT_DMON, DT_HW
from openpilot.common.swaglog import cloudlog
from panda import Panda

SAFETY_TOYOTA = CarParams.SafetyModel.toyota
SAFETY_ALLOUTPUT = CarParams.SafetyModel.allOutput

# Lock / unlock door commands - Credit goes to AlexandreSato!
LOCK_CMD = b"\x40\x05\x30\x11\x00\x80\x00\x00"
UNLOCK_CMD = b"\x40\x05\x30\x11\x00\x40\x00\x00"

# Fold mirrors
MIRR_FOLD_R = b"\xA5\x04\x30\x21\x00\x08\x00\x00"
MIRR_FOLD_L = b"\xA6\x04\x30\x21\x00\x08\x00\x00"

# Close windows
WINDOW_CLOSE_FR = b"\x91\x04\x30\x01\x05\x20\x00\x00"
WINDOW_CLOSE_FL = b"\x90\x04\x30\x01\x05\x20\x00\x00"
WINDOW_CLOSE_RR = b"\x92\x04\x30\x01\x05\x20\x00\x00"
WINDOW_CLOSE_RL = b"\x93\x04\x30\x01\x05\x20\x00\x00"

# Toyota UDS diagnostic request address all of the commands above are sent to.
TOYOTA_DIAG_ADDR = 0x750

# DBC used to read door-open and lock-status feedback. Overridable via the
# "DoorLockDBC" param for platforms that use a different generated DBC.
DEFAULT_DBC = "toyota_nodsu_pt_generated"


def ignition_on(sm: messaging.SubMaster) -> bool:
  return any(ps.ignitionLine or ps.ignitionCan for ps in sm["pandaStates"]
             if ps.pandaType != log.PandaState.PandaType.unknown)


def dmonitoringd_running(sm: messaging.SubMaster) -> bool:
  return any(proc.name == "dmonitoringd" and proc.running for proc in sm["managerState"].processes)


def wait_for_no_driver(sm: messaging.SubMaster, params: Params, dbc: str, time_threshold: int) -> bool:
  """Block until the driver has been gone for `time_threshold` seconds.

  Returns True if the cabin is empty and the doors are shut, or False if the
  ignition came back on (someone got back in) and we should abort.
  """
  can_parser = CANParser(dbc, [("BODY_CONTROL_STATE", 3)], bus=0)
  can_sock = messaging.sub_sock("can", timeout=100)

  # wait for the onroad driver-monitoring stack to shut down
  while dmonitoringd_running(sm):
    sm.update()
    if ignition_on(sm):
      return False
    time.sleep(DT_HW)

  # bring the driver-view camera back up so we can watch the cabin while offroad
  params.put_bool("IsDriverViewEnabled", True)
  try:
    while not dmonitoringd_running(sm):
      sm.update()
      if ignition_on(sm):
        return False
      time.sleep(DT_HW)

    start_time = time.monotonic()
    while time.monotonic() - start_time < time_threshold:
      sm.update()

      if ignition_on(sm):
        return False

      # reset the timer while a face is still detected (someone is in the car)
      if sm["driverMonitoringState"].faceDetected or not sm.alive["driverMonitoringState"]:
        start_time = time.monotonic()

      can_parser.update_strings(messaging.drain_sock_raw(can_sock, wait_for_one=True))
      door_open = any([can_parser.vl["BODY_CONTROL_STATE"]["DOOR_OPEN_FL"],
                       can_parser.vl["BODY_CONTROL_STATE"]["DOOR_OPEN_FR"],
                       can_parser.vl["BODY_CONTROL_STATE"]["DOOR_OPEN_RL"],
                       can_parser.vl["BODY_CONTROL_STATE"]["DOOR_OPEN_RR"]])
      if door_open:
        start_time = time.monotonic()

      time.sleep(DT_DMON)
  finally:
    params.remove("IsDriverViewEnabled")

  return True


def secure_vehicle(sm: messaging.SubMaster, params: Params, dbc: str) -> None:
  """Lock the doors and optionally fold the mirrors / close the windows."""
  can_parser = CANParser(dbc, [("DOOR_LOCKS", 3)], bus=0)
  can_sock = messaging.sub_sock("can", timeout=100)

  fold_mirrors = params.get_bool("FoldMirrors")
  close_windows = params.get_bool("CloseWindows")

  while True:
    sm.update()
    if ignition_on(sm):
      break

    with Panda(disable_checks=True) as panda:
      panda.set_safety_mode(SAFETY_TOYOTA)
      panda.can_send(TOYOTA_DIAG_ADDR, LOCK_CMD, 0)
      time.sleep(0.150)
      panda.send_heartbeat()

      if fold_mirrors:
        for command in (MIRR_FOLD_R, MIRR_FOLD_L):
          panda.set_safety_mode(SAFETY_ALLOUTPUT)
          panda.can_send(TOYOTA_DIAG_ADDR, command, 0)
          time.sleep(0.150)
          panda.send_heartbeat()

      if close_windows:
        for command in (WINDOW_CLOSE_RR, WINDOW_CLOSE_RL, WINDOW_CLOSE_FL, WINDOW_CLOSE_FR):
          panda.set_safety_mode(SAFETY_ALLOUTPUT)
          panda.can_send(TOYOTA_DIAG_ADDR, command, 0)
          time.sleep(0.150)
          panda.send_heartbeat()

    time.sleep(1)

    can_parser.update_strings(messaging.drain_sock_raw(can_sock, wait_for_one=True))
    if can_parser.vl["DOOR_LOCKS"]["LOCK_STATUS"] == 0:
      break


def run_secure_sequence(sm: messaging.SubMaster, params: Params) -> None:
  time_threshold = params.get_int("LockDoorsTimer")
  if time_threshold <= 0:
    return

  dbc = params.get("DoorLockDBC", encoding="utf-8") or DEFAULT_DBC

  try:
    if wait_for_no_driver(sm, params, dbc, time_threshold):
      cloudlog.warning("doorlockd: driver gone, securing vehicle")
      secure_vehicle(sm, params, dbc)
  except Exception:
    cloudlog.exception("doorlockd: failed to secure vehicle")


def main() -> NoReturn:
  params = Params()
  sm = messaging.SubMaster(["deviceState", "pandaStates", "driverMonitoringState", "managerState"])

  was_onroad = params.get_bool("IsOnroad")
  while True:
    sm.update(0)
    onroad = params.get_bool("IsOnroad")

    # trigger on the onroad -> offroad transition (the driver just parked)
    if was_onroad and not onroad:
      run_secure_sequence(sm, params)

    was_onroad = onroad
    time.sleep(DT_HW)


if __name__ == "__main__":
  main()
