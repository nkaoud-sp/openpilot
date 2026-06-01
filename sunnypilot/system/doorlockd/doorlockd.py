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
import traceback
from typing import NoReturn

import cereal.messaging as messaging
from cereal import log
from opendbc.can.parser import CANParser
from opendbc.car.structs import CarParams
from openpilot.common.params import Params
from openpilot.common.realtime import DT_DMON, DT_HW
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.pandad import can_capnp_to_list
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from panda import Panda

# On-screen diagnostics: every status() call updates this offroad alert, which
# shows on the device home screen (tap the alert indicator). Lets us follow the
# sequence without log/file/SSH access.
DOORLOCK_ALERT = "Offroad_DoorlockStatus"


def status(msg: str) -> None:
  cloudlog.warning(f"doorlockd: {msg}")
  try:
    set_offroad_alert(DOORLOCK_ALERT, True, extra_text=msg)
  except Exception:
    cloudlog.exception("doorlockd: failed to set offroad alert")

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

# Cap lock retries so a mis-parsed LOCK_STATUS feedback can't loop forever.
MAX_LOCK_ATTEMPTS = 3

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

  # --- STEP-BY-STEP: driver-monitoring gating temporarily DISABLED ---------
  # Re-enable this block (and the face/dm_alive reset below) once the simpler
  # ignition+door+timer path is confirmed working on the device.
  #
  # # wait for the onroad driver-monitoring stack to shut down
  # cloudlog.warning("doorlockd: phase 1 - waiting for onroad dmonitoringd to stop")
  # while dmonitoringd_running(sm):
  #   sm.update()
  #   if ignition_on(sm):
  #     cloudlog.warning("doorlockd: ignition back on while waiting for dmonitoringd to stop, aborting")
  #     return False
  #   time.sleep(DT_HW)
  #
  # # bring the driver-view camera back up so we can watch the cabin while offroad
  # cloudlog.warning("doorlockd: phase 2 - set IsDriverViewEnabled, waiting for dmonitoringd to come up")
  # params.put_bool("IsDriverViewEnabled", True)
  # while not dmonitoringd_running(sm):
  #   sm.update()
  #   if ignition_on(sm):
  #     cloudlog.warning("doorlockd: ignition back on while waiting for dmonitoringd to start, aborting")
  #     return False
  #   time.sleep(DT_HW)
  # -------------------------------------------------------------------------

  status(f"phase 3 - starting {time_threshold}s countdown (driver-monitoring DISABLED)")
  start_time = time.monotonic()
  last_log = 0.0
  while time.monotonic() - start_time < time_threshold:
    sm.update()

    if ignition_on(sm):
      status("ignition back on during countdown, aborting")
      return False

    # # reset the timer while a face is still detected (someone is in the car)
    # face = sm["driverMonitoringState"].faceDetected
    # dm_alive = sm.alive["driverMonitoringState"]
    # if face or not dm_alive:
    #   start_time = time.monotonic()

    can_parser.update(can_capnp_to_list(messaging.drain_sock_raw(can_sock, wait_for_one=True)))
    door_open = any([can_parser.vl["BODY_CONTROL_STATE"]["DOOR_OPEN_FL"],
                     can_parser.vl["BODY_CONTROL_STATE"]["DOOR_OPEN_FR"],
                     can_parser.vl["BODY_CONTROL_STATE"]["DOOR_OPEN_RL"],
                     can_parser.vl["BODY_CONTROL_STATE"]["DOOR_OPEN_RR"]])
    if door_open:
      start_time = time.monotonic()

    # throttled trace so we can see *why* the countdown is (or isn't) progressing
    now = time.monotonic()
    if now - last_log >= 2.0:
      status(f"countdown remaining={time_threshold - (now - start_time):.1f}s door_open={door_open}")
      last_log = now

    time.sleep(DT_DMON)

  status("phase 4 - countdown complete")
  return True


def secure_vehicle(sm: messaging.SubMaster, params: Params, dbc: str) -> None:
  """STEP-BY-STEP: lock the doors + fold the mirrors.

  Window-close and the driver-monitoring gate are still disabled.

  The mirror fold (allOutput) worked but the lock under *toyota* safety did not
  and LOCK_STATUS stayed 1 -> toyota safety is blocking the 0x750 diagnostic TX.
  So we now send everything (lock + mirrors) under allOutput, which we know
  reaches 0x750. The panda is opened once and the safety mode set once, to keep
  the relay from clicking on every command/attempt.
  """
  can_parser = CANParser(dbc, [("DOOR_LOCKS", 3)], bus=0)
  can_sock = messaging.sub_sock("can", timeout=100)

  status("securing - LOCK + MIRROR FOLD under allOutput (windows + DM disabled)")
  with Panda(disable_checks=True) as panda:
    panda.set_safety_mode(SAFETY_ALLOUTPUT)

    attempt = 0
    while True:
      sm.update()
      if ignition_on(sm):
        status("ignition back on, stopping lock attempts")
        break

      attempt += 1
      status(f"sending LOCK_CMD + mirror fold (attempt {attempt})")
      panda.send_heartbeat()
      panda.can_send(TOYOTA_DIAG_ADDR, LOCK_CMD, 0)
      time.sleep(0.150)
      panda.send_heartbeat()

      for command in (MIRR_FOLD_R, MIRR_FOLD_L):
        panda.can_send(TOYOTA_DIAG_ADDR, command, 0)
        time.sleep(0.150)
        panda.send_heartbeat()

      # # --- close windows (disabled) ---
      # for command in (WINDOW_CLOSE_RR, WINDOW_CLOSE_RL, WINDOW_CLOSE_FL, WINDOW_CLOSE_FR):
      #   panda.can_send(TOYOTA_DIAG_ADDR, command, 0)
      #   time.sleep(0.150)
      #   panda.send_heartbeat()

      time.sleep(1)

      can_parser.update(can_capnp_to_list(messaging.drain_sock_raw(can_sock, wait_for_one=True)))
      lock_status = can_parser.vl["DOOR_LOCKS"]["LOCK_STATUS"]
      status(f"LOCK_STATUS={lock_status} (attempt {attempt})")
      if lock_status == 0:
        status("doors confirmed locked")
        break

      if attempt >= MAX_LOCK_ATTEMPTS:
        status(f"gave up after {attempt} attempts, last LOCK_STATUS={lock_status}")
        break


def run_secure_sequence(sm: messaging.SubMaster, params: Params) -> None:
  try:
    time_threshold = params.get("LockDoorsTimer", return_default=True)
    status(f"run_secure_sequence, LockDoorsTimer={time_threshold!r}")
    if time_threshold <= 0:
      status("timer disabled (<=0), nothing to do")
      return

    dbc = params.get("DoorLockDBC", return_default=True) or DEFAULT_DBC

    if wait_for_no_driver(sm, params, dbc, time_threshold):
      secure_vehicle(sm, params, dbc)
  except Exception as e:
    cloudlog.exception("doorlockd: failed to secure vehicle")
    # surface the actual exception on screen (last frame + type/message), since
    # there's no log access on the device
    tb = traceback.extract_tb(e.__traceback__)
    where = f"{tb[-1].name}:{tb[-1].lineno}" if tb else "?"
    detail = f"ERROR at {where}: {type(e).__name__}: {e}"
    try:
      set_offroad_alert(DOORLOCK_ALERT, True, extra_text=detail[:300])
    except Exception:
      cloudlog.exception("doorlockd: failed to set error offroad alert")


def main() -> NoReturn:
  params = Params()
  sm = messaging.SubMaster(["deviceState", "pandaStates", "driverMonitoringState", "managerState"])

  was_onroad = params.get_bool("IsOnroad")
  cloudlog.warning(f"doorlockd: started, was_onroad={was_onroad}")
  while True:
    sm.update(0)
    onroad = params.get_bool("IsOnroad")

    # trigger on the onroad -> offroad transition (the driver just parked)
    if was_onroad and not onroad:
      status("onroad->offroad transition detected")
      run_secure_sequence(sm, params)

    was_onroad = onroad
    time.sleep(DT_HW)


if __name__ == "__main__":
  main()
