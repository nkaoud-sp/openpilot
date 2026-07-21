"""
Per-drive experimental-longitudinal diagnostics logger.

Records model action/disengage predictions, planner output, radar lead state,
and user interventions while openpilot is active. Sessions are zipped and
queued through the existing nkaoud_nav SMTP mailer when the drive ends.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import time

from openpilot.common.swaglog import cloudlog

LOG_DIR = os.environ.get("EXPERIMENTAL_LONG_LOG_DIR", "/data/media/0/experimental_longitudinal_logs")
SESSION_PREFIX = "session_"
ZIP_PREFIX = "experimental_longitudinal_log_"
MAX_STORED_ZIPS = 10

CSV_FIELDS = [
  "wall_time", "mono_time", "started", "enabled", "experimental_mode", "dec_active", "dec_state",
  "v_ego", "a_ego", "v_cruise", "v_cruise_cluster",
  "gas_pressed", "brake_pressed", "steering_pressed", "cruise_override", "user_intervention", "disengaged",
  "model_desired_accel", "model_should_stop", "model_confidence",
  "plan_a_target", "plan_should_stop", "plan_allow_throttle", "plan_source", "plan_has_lead",
  "mpc_accel_0", "mpc_accel_1",
  "sp_source", "sp_v_target", "sp_a_target",
  "e2e_caution_gap",
  "radar_lead_status", "radar_d_rel", "radar_v_lead", "radar_v_rel", "radar_a_lead",
  "model_lead_prob", "model_lead_x", "model_lead_x_std", "model_lead_v", "model_lead_v_std",
  "gas_press_p0", "gas_press_p1", "gas_press_p2",
  "brake_press_p0", "brake_press_p1", "brake_press_p2",
  "gas_disengage_p0", "gas_disengage_p1", "gas_disengage_p2",
  "brake_disengage_p0", "brake_disengage_p1", "brake_disengage_p2",
  "hard_brake_3_p0", "hard_brake_3_p1",
  "hard_brake_4_p0", "hard_brake_4_p1",
  "hard_brake_5_p0", "hard_brake_5_p1",
]


def _stamp() -> str:
  return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _fmt(value, digits: int = 3) -> str:
  try:
    return f"{float(value):.{digits}f}"
  except (TypeError, ValueError):
    return ""


def _bool(value) -> int:
  return int(bool(value))


def _list_value(values, idx: int) -> str:
  try:
    if len(values) > idx:
      return _fmt(values[idx], 4)
  except (TypeError, ValueError):
    pass
  return ""


def _enum_name(value) -> str:
  try:
    return str(value).split(".")[-1]
  except Exception:  # noqa: BLE001
    return ""


class ExperimentalLongitudinalSessionLogger:
  def __init__(self) -> None:
    self._dir: str | None = None
    self._csv_file = None
    self._csv = None
    self._rows = 0
    self._interventions = 0
    self._disengagements = 0

  @property
  def is_active(self) -> bool:
    return self._dir is not None

  def start(self, config: dict) -> None:
    if self.is_active:
      return
    try:
      session_dir = os.path.join(LOG_DIR, f"{SESSION_PREFIX}{_stamp()}")
      os.makedirs(session_dir, exist_ok=True)
      with open(os.path.join(session_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)
      self._csv_file = open(os.path.join(session_dir, "experimental_longitudinal.csv"), "w", newline="")
      self._csv = csv.DictWriter(self._csv_file, fieldnames=CSV_FIELDS)
      self._csv.writeheader()
      self._dir = session_dir
      self._rows = 0
      self._interventions = 0
      self._disengagements = 0
    except OSError:
      cloudlog.exception("experimental_longitudinal_logger: could not start session")
      self._close_files()
      self._dir = None

  def log(self, sm, disengaged: bool) -> None:
    if not self.is_active or self._csv is None:
      return
    try:
      CS = sm["carState"]
      CC = sm["carControl"]
      SS = sm["selfdriveState"]
      LP = sm["longitudinalPlan"]
      LSP = sm["longitudinalPlanSP"]
      model = sm["modelV2"]
      lead = sm["radarState"].leadOne
      preds = model.meta.disengagePredictions

      user_intervention = bool(CS.gasPressed or CS.brakePressed)
      if user_intervention:
        self._interventions += 1
      if disengaged:
        self._disengagements += 1

      model_lead = model.leadsV3[0] if len(model.leadsV3) else None
      model_lead_prob = model_lead.prob if model_lead is not None else ""
      model_lead_x = model_lead.x[0] if model_lead is not None and len(model_lead.x) else ""
      model_lead_x_std = model_lead.xStd[0] if model_lead is not None and len(model_lead.xStd) else ""
      model_lead_v = model_lead.v[0] if model_lead is not None and len(model_lead.v) else ""
      model_lead_v_std = model_lead.vStd[0] if model_lead is not None and len(model_lead.vStd) else ""

      mpc_accel_0 = LP.accels[0] if len(LP.accels) > 0 else ""
      mpc_accel_1 = LP.accels[1] if len(LP.accels) > 1 else ""

      row = {
        "wall_time": time.strftime("%H:%M:%S", time.localtime()),
        "mono_time": _fmt(time.monotonic(), 2),
        "started": _bool(sm["deviceState"].started),
        "enabled": _bool(SS.enabled),
        "experimental_mode": _bool(SS.experimentalMode),
        "dec_active": _bool(LSP.dec.active),
        "dec_state": _enum_name(LSP.dec.state),
        "v_ego": _fmt(CS.vEgo),
        "a_ego": _fmt(CS.aEgo),
        "v_cruise": _fmt(CS.vCruise),
        "v_cruise_cluster": _fmt(CS.vCruiseCluster),
        "gas_pressed": _bool(CS.gasPressed),
        "brake_pressed": _bool(CS.brakePressed),
        "steering_pressed": _bool(CS.steeringPressed),
        "cruise_override": _bool(CC.cruiseControl.override),
        "user_intervention": _bool(user_intervention),
        "disengaged": _bool(disengaged),
        "model_desired_accel": _fmt(model.action.desiredAcceleration),
        "model_should_stop": _bool(model.action.shouldStop),
        "model_confidence": _enum_name(model.confidence),
        "plan_a_target": _fmt(LP.aTarget),
        "plan_should_stop": _bool(LP.shouldStop),
        "plan_allow_throttle": _bool(LP.allowThrottle),
        "plan_source": _enum_name(LP.longitudinalPlanSource),
        "plan_has_lead": _bool(LP.hasLead),
        "mpc_accel_0": _fmt(mpc_accel_0),
        "mpc_accel_1": _fmt(mpc_accel_1),
        "sp_source": _enum_name(LSP.longitudinalPlanSource),
        "sp_v_target": _fmt(LSP.vTarget),
        "sp_a_target": _fmt(LSP.aTarget),
        "e2e_caution_gap": _fmt(float(mpc_accel_0) - model.action.desiredAcceleration if mpc_accel_0 != "" else ""),
        "radar_lead_status": _bool(lead.status),
        "radar_d_rel": _fmt(lead.dRel),
        "radar_v_lead": _fmt(lead.vLead),
        "radar_v_rel": _fmt(lead.vRel),
        "radar_a_lead": _fmt(getattr(lead, "aLeadK", "")),
        "model_lead_prob": _fmt(model_lead_prob, 4),
        "model_lead_x": _fmt(model_lead_x),
        "model_lead_x_std": _fmt(model_lead_x_std),
        "model_lead_v": _fmt(model_lead_v),
        "model_lead_v_std": _fmt(model_lead_v_std),
        "gas_press_p0": _list_value(preds.gasPressProbs, 0),
        "gas_press_p1": _list_value(preds.gasPressProbs, 1),
        "gas_press_p2": _list_value(preds.gasPressProbs, 2),
        "brake_press_p0": _list_value(preds.brakePressProbs, 0),
        "brake_press_p1": _list_value(preds.brakePressProbs, 1),
        "brake_press_p2": _list_value(preds.brakePressProbs, 2),
        "gas_disengage_p0": _list_value(preds.gasDisengageProbs, 0),
        "gas_disengage_p1": _list_value(preds.gasDisengageProbs, 1),
        "gas_disengage_p2": _list_value(preds.gasDisengageProbs, 2),
        "brake_disengage_p0": _list_value(preds.brakeDisengageProbs, 0),
        "brake_disengage_p1": _list_value(preds.brakeDisengageProbs, 1),
        "brake_disengage_p2": _list_value(preds.brakeDisengageProbs, 2),
        "hard_brake_3_p0": _list_value(preds.brake3MetersPerSecondSquaredProbs, 0),
        "hard_brake_3_p1": _list_value(preds.brake3MetersPerSecondSquaredProbs, 1),
        "hard_brake_4_p0": _list_value(preds.brake4MetersPerSecondSquaredProbs, 0),
        "hard_brake_4_p1": _list_value(preds.brake4MetersPerSecondSquaredProbs, 1),
        "hard_brake_5_p0": _list_value(preds.brake5MetersPerSecondSquaredProbs, 0),
        "hard_brake_5_p1": _list_value(preds.brake5MetersPerSecondSquaredProbs, 1),
      }
      self._csv.writerow(row)
      self._csv_file.flush()
      self._rows += 1
    except Exception:  # noqa: BLE001 -- keep diagnostics from crashing controls-adjacent logging
      cloudlog.exception("experimental_longitudinal_logger: csv write failed")

  def end(self) -> str | None:
    if not self.is_active:
      return None
    session_dir, self._dir = self._dir, None
    try:
      with open(os.path.join(session_dir, "summary.json"), "w") as f:
        json.dump({
          "rows": self._rows,
          "interventions": self._interventions,
          "disengagements": self._disengagements,
        }, f, indent=2)
    except OSError:
      cloudlog.exception("experimental_longitudinal_logger: summary write failed")
    self._close_files()
    return _zip_and_queue(session_dir)

  def _close_files(self) -> None:
    if self._csv_file is not None:
      try:
        self._csv_file.close()
      except OSError:
        pass
    self._csv_file = None
    self._csv = None


def _zip_and_queue(session_dir: str) -> str | None:
  try:
    if not os.path.isdir(session_dir):
      return None
    csv_path = os.path.join(session_dir, "experimental_longitudinal.csv")
    if not os.path.isfile(csv_path):
      shutil.rmtree(session_dir, ignore_errors=True)
      return None
    stamp = os.path.basename(session_dir)[len(SESSION_PREFIX):]
    zip_base = os.path.join(LOG_DIR, f"{ZIP_PREFIX}{stamp}")
    zip_path = shutil.make_archive(zip_base, "zip", session_dir)
    shutil.rmtree(session_dir, ignore_errors=True)
    _prune_zips()

    from sunnypilot.nkaoud_nav import mailer
    mailer.queue_log(zip_path)
    return zip_path
  except Exception:  # noqa: BLE001
    cloudlog.exception("experimental_longitudinal_logger: zip/queue failed")
    return None


def _prune_zips() -> None:
  try:
    zips = sorted(e.path for e in os.scandir(LOG_DIR)
                  if e.is_file() and e.name.startswith(ZIP_PREFIX) and e.name.endswith(".zip"))
    for path in zips[:-MAX_STORED_ZIPS]:
      os.remove(path)
  except OSError:
    cloudlog.exception("experimental_longitudinal_logger: prune failed")


def finalize_orphan_sessions() -> int:
  count = 0
  try:
    if not os.path.isdir(LOG_DIR):
      return 0
    for entry in sorted(os.scandir(LOG_DIR), key=lambda e: e.name):
      if entry.is_dir() and entry.name.startswith(SESSION_PREFIX):
        if _zip_and_queue(entry.path):
          count += 1
  except OSError:
    cloudlog.exception("experimental_longitudinal_logger: orphan finalize failed")
  return count
