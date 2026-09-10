#!/usr/bin/env python3
import time
from typing import NoReturn

from cereal import car, log, messaging
from openpilot.common.params import Params
from openpilot.common.realtime import DT_DMON, DT_HW
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.selfdrived.alertmanager import set_offroad_alert
from panda import Panda

DOORLOCK_ALERT = "Offroad_DoorlockStatus"
SAFETY_ALLOUTPUT = car.CarParams.SafetyModel.allOutput

LOCK_CMD = b"\x40\x05\x30\x11\x00\x80\x00\x00"
MIRR_FOLD_R = b"\xA5\x04\x30\x21\x00\x08\x00\x00"
MIRR_FOLD_L = b"\xA6\x04\x30\x21\x00\x08\x00\x00"
WINDOW_CLOSE_FR = b"\x91\x04\x30\x01\x05\x20\x00\x00"
WINDOW_CLOSE_FL = b"\x90\x04\x30\x01\x05\x20\x00\x00"
WINDOW_CLOSE_RR = b"\x92\x04\x30\x01\x05\x20\x00\x00"
WINDOW_CLOSE_RL = b"\x93\x04\x30\x01\x05\x20\x00\x00"

TOYOTA_DIAG_ADDR = 0x750
CMD_DELAY = 0.150
MAX_LOCK_ATTEMPTS = 2
DM_WAIT_TIMEOUT = 30.0


def status(msg: str) -> None:
  cloudlog.warning(f"doorlockd: {msg}")


def show_alert(msg: str | None) -> None:
  try:
    set_offroad_alert(DOORLOCK_ALERT, msg is not None, extra_text=msg or "")
  except Exception:
    cloudlog.exception("doorlockd: failed to update offroad alert")


def ignition_on(sm: messaging.SubMaster) -> bool:
  return any(ps.ignitionLine or ps.ignitionCan for ps in sm["pandaStates"]
             if ps.pandaType != log.PandaState.PandaType.unknown)


def dmonitoringd_running(sm: messaging.SubMaster) -> bool:
  return any(proc.name == "dmonitoringd" and proc.running for proc in sm["managerState"].processes)


def door_open(sm: messaging.SubMaster) -> bool:
  return bool(sm.recv_frame["carState"] > 0 and sm["carState"].doorOpen)


def send_diag(panda: Panda, cmd: bytes, delay: float = CMD_DELAY) -> None:
  panda.set_safety_mode(SAFETY_ALLOUTPUT)
  panda.can_send(TOYOTA_DIAG_ADDR, cmd, 0)
  time.sleep(delay)
  panda.send_heartbeat()


def wait_for_no_driver(sm: messaging.SubMaster, params: Params, time_threshold: int) -> bool:
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
        status("timeout: dmonitoringd never came up; proceeding without face gating")
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

      face = sm["driverMonitoringState"].faceDetected
      dm_alive = sm.alive["driverMonitoringState"]
      if dm_available and (face or not dm_alive):
        start_time = time.monotonic()

      doors_open = door_open(sm)
      if doors_open:
        start_time = time.monotonic()

      now = time.monotonic()
      if now - last_log >= 2.0:
        status(f"countdown remaining={time_threshold - (now - start_time):.1f}s "
               f"face={face} dm_alive={dm_alive} door_open={doors_open}")
        last_log = now

      time.sleep(DT_DMON)
  finally:
    params.remove("IsDriverViewEnabled")

  status("phase 4 - countdown complete")
  return True


def secure_vehicle(sm: messaging.SubMaster, params: Params) -> None:
  fold_mirrors = params.get_bool("FoldMirrors")
  close_windows = params.get_bool("CloseWindows")
  status(f"securing - lock + mirrors={fold_mirrors} + windows={close_windows}")
  with Panda(disable_checks=True) as panda:
    for attempt in range(1, MAX_LOCK_ATTEMPTS + 1):
      sm.update()
      if ignition_on(sm):
        status("ignition back on, stopping lock attempts")
        break

      send_diag(panda, LOCK_CMD)

      if fold_mirrors:
        for cmd in (MIRR_FOLD_R, MIRR_FOLD_L, MIRR_FOLD_R, MIRR_FOLD_L):
          send_diag(panda, cmd)

      if close_windows:
        for cmd in (WINDOW_CLOSE_RR, WINDOW_CLOSE_RL, WINDOW_CLOSE_FL, WINDOW_CLOSE_FR,
                    WINDOW_CLOSE_RR, WINDOW_CLOSE_RL, WINDOW_CLOSE_FL, WINDOW_CLOSE_FR):
          send_diag(panda, cmd)

      time.sleep(1)
      status(f"sent secure commands (attempt {attempt})")


def run_secure_sequence(sm: messaging.SubMaster, params: Params) -> None:
  show_alert(None)
  try:
    if not params.get_bool("AutoLockEnabled"):
      status("auto lock disabled, nothing to do")
      return

    time_threshold = params.get_int("LockDoorsTimer", return_default=True)
    status(f"run_secure_sequence, LockDoorsTimer={time_threshold!r}")
    if time_threshold <= 0:
      status("timer disabled, nothing to do")
      return

    if wait_for_no_driver(sm, params, time_threshold):
      secure_vehicle(sm, params)
  except Exception:
    cloudlog.exception("doorlockd: failed to secure vehicle")


def main() -> NoReturn:
  params = Params()
  sm = messaging.SubMaster(["deviceState", "pandaStates", "driverMonitoringState", "managerState", "carState"])

  was_onroad = not params.get_bool("IsOffroad")
  cloudlog.warning(f"doorlockd: started, was_onroad={was_onroad}")
  while True:
    sm.update(0)
    onroad = not params.get_bool("IsOffroad")

    if was_onroad and not onroad:
      status("onroad->offroad transition detected")
      run_secure_sequence(sm, params)
    elif not params.get_bool("AutoLockEnabled"):
      show_alert(None)

    was_onroad = onroad
    time.sleep(DT_HW)


if __name__ == "__main__":
  main()
