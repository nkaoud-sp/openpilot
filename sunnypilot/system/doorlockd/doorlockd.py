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
from openpilot.selfdrive.pandad import can_capnp_to_list
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from panda import Panda

# Phase breadcrumbs go to the cloud log via status(). The on-screen offroad
# banner (show_alert) is reserved for the one case worth surfacing to the
# driver: when the driver-view dmonitoringd can't be started, so the daemon
# falls back to timer-only gating.
DOORLOCK_ALERT = "Offroad_DoorlockStatus"


def status(msg: str) -> None:
  cloudlog.warning(f"doorlockd: {msg}")


def show_alert(msg: str | None) -> None:
  """Show the on-screen offroad banner, or clear it when msg is None."""
  try:
    set_offroad_alert(DOORLOCK_ALERT, msg is not None, extra_text=msg or "")
  except Exception:
    cloudlog.exception("doorlockd: failed to update offroad alert")

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

# Delay after each diagnostic command. MIRROR_GAP is the (doubled) separation
# between the right and left mirror fold so they don't step on each other.
CMD_DELAY = 0.150
MIRROR_GAP = CMD_DELAY * 2

# Cap lock retries so a mis-parsed LOCK_STATUS feedback can't loop forever.
MAX_LOCK_ATTEMPTS = 2

# Max seconds to wait for the driver-view dmonitoringd to stop/start before
# giving up. If it never comes up we fall back to ignition+door+timer gating.
DM_WAIT_TIMEOUT = 30.0

# DBC used to read door-open and lock-status feedback. Overridable via the
# "DoorLockDBC" param for platforms that use a different generated DBC.
DEFAULT_DBC = "toyota_nodsu_pt_generated"


def ignition_on(sm: messaging.SubMaster) -> bool:
  return any(ps.ignitionLine or ps.ignitionCan for ps in sm["pandaStates"]
             if ps.pandaType != log.PandaState.PandaType.unknown)


def dmonitoringd_running(sm: messaging.SubMaster) -> bool:
  return any(proc.name == "dmonitoringd" and proc.running for proc in sm["managerState"].processes)


def send_diag(panda: Panda, cmd: bytes, delay: float = CMD_DELAY) -> None:
  """Send one diagnostic command to 0x750. allOutput is re-asserted every call
  because pandad reverts the safety mode to silent between sends."""
  panda.set_safety_mode(SAFETY_ALLOUTPUT)
  panda.can_send(TOYOTA_DIAG_ADDR, cmd, 0)
  time.sleep(delay)
  panda.send_heartbeat()


def wait_for_no_driver(sm: messaging.SubMaster, params: Params, dbc: str, time_threshold: int) -> bool:
  """Block until the driver has been gone for `time_threshold` seconds.

  Returns True if the cabin is empty and the doors are shut, or False if the
  ignition came back on (someone got back in) and we should abort.
  """
  can_parser = CANParser(dbc, [("BODY_CONTROL_STATE", 3)], bus=0)
  can_sock = messaging.sub_sock("can", timeout=100)

  # phase 1: wait for the onroad driver-monitoring stack to shut down (bounded)
  status("phase 1 - waiting for onroad dmonitoringd to stop")
  t0 = time.monotonic()
  while dmonitoringd_running(sm):
    sm.update()
    if ignition_on(sm):
      status("ignition back on while waiting for dmonitoringd to stop, aborting")
      return False
    if time.monotonic() - t0 > DM_WAIT_TIMEOUT:
      status("timeout waiting for onroad dmonitoringd to stop, continuing")
      break
    time.sleep(DT_HW)

  # phase 2: bring the driver-view camera up so we can watch the cabin offroad.
  # bounded: if dmonitoringd never comes up, fall back to no face gating.
  status("phase 2 - enabling driver view, waiting for dmonitoringd")
  params.put_bool("IsDriverViewEnabled", True)
  try:
    dm_available = True
    t0 = time.monotonic()
    while not dmonitoringd_running(sm):
      sm.update()
      if ignition_on(sm):
        status("ignition back on while waiting for dmonitoringd to start, aborting")
        return False
      if time.monotonic() - t0 > DM_WAIT_TIMEOUT:
        status("timeout: dmonitoringd never came up; proceeding WITHOUT face gating")
        show_alert("driver-view camera unavailable - securing on timer only")
        dm_available = False
        break
      time.sleep(DT_HW)

    status(f"phase 3 - dm_available={dm_available}, starting {time_threshold}s countdown")
    start_time = time.monotonic()
    last_log = 0.0
    while time.monotonic() - start_time < time_threshold:
      sm.update()

      if ignition_on(sm):
        status("ignition back on during countdown, aborting")
        return False

      # reset the timer while a face is still detected (someone is in the car).
      # if DM is supposed to be up but isn't publishing, also reset so we never
      # lock blind. if DM never came up at all, skip face gating entirely.
      face = sm["driverMonitoringState"].faceDetected
      dm_alive = sm.alive["driverMonitoringState"]
      if dm_available and (face or not dm_alive):
        start_time = time.monotonic()

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
        status(f"countdown remaining={time_threshold - (now - start_time):.1f}s "
               f"face={face} dm_alive={dm_alive} door_open={door_open}")
        last_log = now

      time.sleep(DT_DMON)
  finally:
    params.remove("IsDriverViewEnabled")

  status("phase 4 - countdown complete")
  return True


def secure_vehicle(sm: messaging.SubMaster, params: Params, dbc: str) -> None:
  """Lock the doors, fold the mirrors, and close the windows.

  The mirror fold (allOutput) worked but the lock under *toyota* safety did not
  and LOCK_STATUS stayed 1 -> toyota safety is blocking the 0x750 diagnostic TX.
  So everything is sent under allOutput (via send_diag), which we know reaches
  0x750.

  pandad runs concurrently and keeps reverting the safety mode back to silent
  between our sends, so allOutput must be re-asserted immediately before *every*
  can_send (setting it once only let the first command through). This re-clicks
  the relay per command, which is expected.
  """
  can_parser = CANParser(dbc, [("DOOR_LOCKS", 3)], bus=0)
  can_sock = messaging.sub_sock("can", timeout=100)

  # optional extras, gated by their params (the door lock itself always runs)
  fold_mirrors = params.get_bool("FoldMirrors")
  close_windows = params.get_bool("CloseWindows")
  status(f"securing - lock + mirrors={fold_mirrors} + windows={close_windows} (allOutput, DM gated)")
  with Panda(disable_checks=True) as panda:
    attempt = 0
    while True:
      sm.update()
      if ignition_on(sm):
        status("ignition back on, stopping lock attempts")
        break

      #send_diag(panda, MIRR_FOLD_R, MIRROR_GAP)
      
      attempt += 1
      status(f"sending lock (mirrors={fold_mirrors} windows={close_windows}) attempt {attempt}")
      send_diag(panda, LOCK_CMD)
      
      if fold_mirrors:
        send_diag(panda, MIRR_FOLD_R)
      if fold_mirrors:
        send_diag(panda, MIRR_FOLD_L)
      if fold_mirrors:
        send_diag(panda, MIRR_FOLD_R)
      if fold_mirrors:
        send_diag(panda, MIRR_FOLD_L)
        
      if close_windows:
        send_diag(panda, WINDOW_CLOSE_RR)
      if close_windows:
        send_diag(panda, WINDOW_CLOSE_RL)
      if close_windows:
        send_diag(panda, WINDOW_CLOSE_FL)
      if close_windows:
        send_diag(panda, WINDOW_CLOSE_FR)
      if close_windows:
        send_diag(panda, WINDOW_CLOSE_RR)
      if close_windows:
        send_diag(panda, WINDOW_CLOSE_RL)
      if close_windows:
        send_diag(panda, WINDOW_CLOSE_FL)
      if close_windows:
        send_diag(panda, WINDOW_CLOSE_FR)

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
  show_alert(None)  # clear any banner from a previous park
  try:
    if not params.get_bool("AutoLockEnabled"):
      status("auto lock disabled, nothing to do")
      return

    time_threshold = params.get("LockDoorsTimer", return_default=True)
    status(f"run_secure_sequence, LockDoorsTimer={time_threshold!r}")
    if time_threshold <= 0:
      status("timer disabled (<=0), nothing to do")
      return

    dbc = params.get("DoorLockDBC", return_default=True) or DEFAULT_DBC

    if wait_for_no_driver(sm, params, dbc, time_threshold):
      secure_vehicle(sm, params, dbc)
  except Exception:
    cloudlog.exception("doorlockd: failed to secure vehicle")


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
