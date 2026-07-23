"""
Offroad exporter for experimental longitudinal diagnostics.

This mines the normal qlog/rlog route files after a drive instead of sampling
live while driving. It only emits a compact CSV around gas/brake interventions.
"""
from __future__ import annotations

import csv
import gc
import json
import os
import re
import shutil
import time
from collections import deque
from dataclasses import dataclass

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware.hw import Paths
from openpilot.tools.lib.logreader import LogReader

LOG_DIR = os.environ.get("EXPERIMENTAL_LONG_LOG_DIR", "/data/media/0/experimental_longitudinal_logs")
ZIP_PREFIX = "experimental_longitudinal_log_"
LAST_EXPORTED_PARAM = "ExperimentalLongitudinalLastExportedRoute"
PENDING_ROUTE_PARAM = "ExperimentalLongitudinalPendingRoute"
MAX_EXPORTED_ROUTE_HISTORY = 20
MAX_STORED_ZIPS = 10
PRE_INTERVENTION_SECONDS = 10.0
POST_INTERVENTION_SECONDS = 5.0
ROUTE_SETTLE_SECONDS = 30.0
SEGMENT_RE = re.compile(r"^(?P<route>.+)--(?P<segment>[0-9]+)$")

SERVICES = {
  "carState", "carControl", "selfdriveState", "longitudinalPlan",
  "longitudinalPlanSP", "modelV2", "radarState",
}

CSV_FIELDS = [
  "route", "time_s", "source_msg",
  "enabled", "experimental_mode", "v_ego", "a_ego", "v_cruise", "v_cruise_cluster",
  "gas_pressed", "brake_pressed", "steering_pressed", "cruise_override", "user_intervention",
  "model_desired_accel", "model_should_stop", "model_confidence",
  "plan_a_target", "plan_should_stop", "plan_allow_throttle", "plan_source", "plan_has_lead",
  "mpc_accel_0", "mpc_accel_1", "e2e_caution_gap",
  "sp_source", "sp_v_target", "sp_a_target",
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


@dataclass(frozen=True)
class RouteCandidate:
  route: str
  mtime: float
  paths: tuple[str, ...]
  source_files: tuple[str, ...]


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


def _route_candidates(require_settled: bool = True) -> list[RouteCandidate]:
  root = Paths.log_root()
  if not os.path.isdir(root):
    return []

  grouped: dict[str, list[tuple[int, str]]] = {}
  mtimes: dict[str, float] = {}
  for name in os.listdir(root):
    match = SEGMENT_RE.match(name)
    if match is None:
      continue
    path = os.path.join(root, name)
    if not os.path.isdir(path) or os.path.exists(os.path.join(path, "rlog.lock")):
      continue
    route = match.group("route")
    rlog = os.path.join(path, "rlog.zst")
    qlog = os.path.join(path, "qlog.zst")
    # modelV2 is not qlogged in this tree, and it is the most important signal
    # for this diagnostic. Prefer rlog and only fall back to qlog if needed.
    log_path = rlog if os.path.isfile(rlog) else qlog if os.path.isfile(qlog) else ""
    if not log_path:
      continue
    grouped.setdefault(route, []).append((int(match.group("segment")), log_path))
    mtimes[route] = max(mtimes.get(route, 0.0), os.path.getmtime(path), os.path.getmtime(log_path))

  now = time.time()
  candidates = [
    RouteCandidate(
      route=route,
      mtime=mtimes[route],
      paths=tuple(path for _, path in sorted(paths)),
      source_files=tuple(f"{segment}:{os.path.basename(path)}" for segment, path in sorted(paths)),
    )
    for route, paths in grouped.items()
    if not require_settled or now - mtimes[route] >= ROUTE_SETTLE_SECONDS
  ]
  return sorted(candidates, key=lambda c: c.mtime, reverse=True)


def latest_route_name() -> str:
  candidates = _route_candidates(require_settled=False)
  return candidates[0].route if candidates else ""


def _exported_routes(params: Params) -> list[str]:
  raw = (params.get(LAST_EXPORTED_PARAM) or "").strip()
  if not raw:
    return []
  try:
    value = json.loads(raw)
  except ValueError:
    return [raw]
  if isinstance(value, list):
    return [str(route) for route in value if str(route).strip()]
  if isinstance(value, str) and value.strip():
    return [value.strip()]
  return []


def _mark_exported(params: Params, route: str) -> None:
  routes = [r for r in _exported_routes(params) if r != route]
  routes.append(route)
  params.put(LAST_EXPORTED_PARAM, json.dumps(routes[-MAX_EXPORTED_ROUTE_HISTORY:]))


def _blank_state() -> dict:
  return {service: None for service in SERVICES}


def _row(route: str, time_s: float, source_msg: str, state: dict) -> dict:
  CS = state["carState"]
  CC = state["carControl"]
  SS = state["selfdriveState"]
  LP = state["longitudinalPlan"]
  LSP = state["longitudinalPlanSP"]
  model = state["modelV2"]
  radar = state["radarState"]
  lead = radar.leadOne if radar is not None else None
  preds = model.meta.disengagePredictions if model is not None else None
  model_lead = model.leadsV3[0] if model is not None and len(model.leadsV3) else None

  gas_pressed = bool(CS is not None and CS.gasPressed)
  brake_pressed = bool(CS is not None and CS.brakePressed)
  mpc_accel_0 = LP.accels[0] if LP is not None and len(LP.accels) > 0 else ""
  mpc_accel_1 = LP.accels[1] if LP is not None and len(LP.accels) > 1 else ""
  model_accel = model.action.desiredAcceleration if model is not None else ""

  return {
    "route": route,
    "time_s": _fmt(time_s, 2),
    "source_msg": source_msg,
    "enabled": _bool(SS is not None and SS.enabled),
    "experimental_mode": _bool(SS is not None and SS.experimentalMode),
    "v_ego": _fmt(CS.vEgo if CS is not None else ""),
    "a_ego": _fmt(CS.aEgo if CS is not None else ""),
    "v_cruise": _fmt(CS.vCruise if CS is not None else ""),
    "v_cruise_cluster": _fmt(CS.vCruiseCluster if CS is not None else ""),
    "gas_pressed": _bool(gas_pressed),
    "brake_pressed": _bool(brake_pressed),
    "steering_pressed": _bool(CS is not None and CS.steeringPressed),
    "cruise_override": _bool(CC is not None and CC.cruiseControl.override),
    "user_intervention": _bool(gas_pressed or brake_pressed),
    "model_desired_accel": _fmt(model_accel),
    "model_should_stop": _bool(model is not None and model.action.shouldStop),
    "model_confidence": _enum_name(model.confidence if model is not None else ""),
    "plan_a_target": _fmt(LP.aTarget if LP is not None else ""),
    "plan_should_stop": _bool(LP is not None and LP.shouldStop),
    "plan_allow_throttle": _bool(LP is not None and LP.allowThrottle),
    "plan_source": _enum_name(LP.longitudinalPlanSource if LP is not None else ""),
    "plan_has_lead": _bool(LP is not None and LP.hasLead),
    "mpc_accel_0": _fmt(mpc_accel_0),
    "mpc_accel_1": _fmt(mpc_accel_1),
    "e2e_caution_gap": _fmt(float(mpc_accel_0) - float(model_accel) if mpc_accel_0 != "" and model_accel != "" else ""),
    "sp_source": _enum_name(LSP.longitudinalPlanSource if LSP is not None else ""),
    "sp_v_target": _fmt(LSP.vTarget if LSP is not None else ""),
    "sp_a_target": _fmt(LSP.aTarget if LSP is not None else ""),
    "radar_lead_status": _bool(lead is not None and lead.status),
    "radar_d_rel": _fmt(lead.dRel if lead is not None else ""),
    "radar_v_lead": _fmt(lead.vLead if lead is not None else ""),
    "radar_v_rel": _fmt(lead.vRel if lead is not None else ""),
    "radar_a_lead": _fmt(getattr(lead, "aLeadK", "") if lead is not None else ""),
    "model_lead_prob": _fmt(model_lead.prob if model_lead is not None else "", 4),
    "model_lead_x": _fmt(model_lead.x[0] if model_lead is not None and len(model_lead.x) else ""),
    "model_lead_x_std": _fmt(model_lead.xStd[0] if model_lead is not None and len(model_lead.xStd) else ""),
    "model_lead_v": _fmt(model_lead.v[0] if model_lead is not None and len(model_lead.v) else ""),
    "model_lead_v_std": _fmt(model_lead.vStd[0] if model_lead is not None and len(model_lead.vStd) else ""),
    "gas_press_p0": _list_value(preds.gasPressProbs, 0) if preds is not None else "",
    "gas_press_p1": _list_value(preds.gasPressProbs, 1) if preds is not None else "",
    "gas_press_p2": _list_value(preds.gasPressProbs, 2) if preds is not None else "",
    "brake_press_p0": _list_value(preds.brakePressProbs, 0) if preds is not None else "",
    "brake_press_p1": _list_value(preds.brakePressProbs, 1) if preds is not None else "",
    "brake_press_p2": _list_value(preds.brakePressProbs, 2) if preds is not None else "",
    "gas_disengage_p0": _list_value(preds.gasDisengageProbs, 0) if preds is not None else "",
    "gas_disengage_p1": _list_value(preds.gasDisengageProbs, 1) if preds is not None else "",
    "gas_disengage_p2": _list_value(preds.gasDisengageProbs, 2) if preds is not None else "",
    "brake_disengage_p0": _list_value(preds.brakeDisengageProbs, 0) if preds is not None else "",
    "brake_disengage_p1": _list_value(preds.brakeDisengageProbs, 1) if preds is not None else "",
    "brake_disengage_p2": _list_value(preds.brakeDisengageProbs, 2) if preds is not None else "",
    "hard_brake_3_p0": _list_value(preds.brake3MetersPerSecondSquaredProbs, 0) if preds is not None else "",
    "hard_brake_3_p1": _list_value(preds.brake3MetersPerSecondSquaredProbs, 1) if preds is not None else "",
    "hard_brake_4_p0": _list_value(preds.brake4MetersPerSecondSquaredProbs, 0) if preds is not None else "",
    "hard_brake_4_p1": _list_value(preds.brake4MetersPerSecondSquaredProbs, 1) if preds is not None else "",
    "hard_brake_5_p0": _list_value(preds.brake5MetersPerSecondSquaredProbs, 0) if preds is not None else "",
    "hard_brake_5_p1": _list_value(preds.brake5MetersPerSecondSquaredProbs, 1) if preds is not None else "",
  }


def _extract_rows(candidate: RouteCandidate) -> list[dict]:
  state = _blank_state()
  rows: list[dict] = []
  pending = deque()
  post_until = -1.0
  first_time_ns: int | None = None
  last_emit_s = -1.0

  for path in candidate.paths:
    try:
      for msg in LogReader(path):
        which = msg.which()
        if which not in SERVICES:
          continue

        if first_time_ns is None:
          first_time_ns = msg.logMonoTime
        time_s = (msg.logMonoTime - first_time_ns) / 1e9
        state[which] = getattr(msg, which)

        if which not in ("carState", "modelV2", "longitudinalPlan"):
          continue
        if time_s - last_emit_s < 0.18:
          continue
        last_emit_s = time_s

        row = _row(candidate.route, time_s, which, state)
        if not row["model_desired_accel"] or not row["plan_a_target"]:
          continue

        pending.append(row)
        while pending and time_s - float(pending[0]["time_s"]) > PRE_INTERVENTION_SECONDS:
          if float(pending[0]["time_s"]) <= post_until:
            rows.append(pending.popleft())
          else:
            pending.popleft()

        enabled_context = any(p["enabled"] == 1 or p["enabled"] == "1" for p in pending)
        if row["user_intervention"] and enabled_context:
          post_until = time_s + POST_INTERVENTION_SECONDS
          while pending:
            rows.append(pending.popleft())
        elif time_s <= post_until:
          rows.append(row)
    finally:
      gc.collect()

  deduped = []
  seen = set()
  for row in rows:
    key = (row["time_s"], row["source_msg"])
    if key in seen:
      continue
    seen.add(key)
    deduped.append(row)
  if not any(row["enabled"] == 1 or row["enabled"] == "1" for row in deduped):
    return []
  return deduped


def _zip_and_queue(session_dir: str) -> str | None:
  try:
    stamp = os.path.basename(session_dir)
    zip_base = os.path.join(LOG_DIR, f"{ZIP_PREFIX}{stamp}")
    zip_path = shutil.make_archive(zip_base, "zip", session_dir)
    shutil.rmtree(session_dir, ignore_errors=True)
    _prune_zips()

    from sunnypilot.nkaoud_nav import mailer
    mailer.queue_log(zip_path)
    return zip_path
  except Exception:  # noqa: BLE001
    cloudlog.exception("experimental_longitudinal_exporter: zip/queue failed")
    return None


def _prune_zips() -> None:
  try:
    zips = sorted(e.path for e in os.scandir(LOG_DIR)
                  if e.is_file() and e.name.startswith(ZIP_PREFIX) and e.name.endswith(".zip"))
    for path in zips[:-MAX_STORED_ZIPS]:
      os.remove(path)
  except OSError:
    cloudlog.exception("experimental_longitudinal_exporter: prune failed")


def export_route(route: str) -> bool:
  """Export one specific route. Returns True once the route was considered.

  False means it was not ready yet or failed before we could make a decision,
  so the caller should retry later.
  """
  route = route.strip()
  if not route:
    return True

  params = Params()
  if route in set(_exported_routes(params)):
    return True

  candidates = [c for c in _route_candidates(require_settled=True) if c.route == route]
  if not candidates:
    return False

  candidate = candidates[0]
  try:
    rows = _extract_rows(candidate)
  except Exception:  # noqa: BLE001
    cloudlog.exception(f"experimental_longitudinal_exporter: failed to read {candidate.route}")
    return False

  _mark_exported(params, candidate.route)
  if not rows:
    return True

  try:
    session_dir = os.path.join(LOG_DIR, f"{candidate.route.replace('|', '_')}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}")
    os.makedirs(session_dir, exist_ok=True)
    with open(os.path.join(session_dir, "experimental_longitudinal_events.csv"), "w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
      writer.writeheader()
      writer.writerows(rows)
    with open(os.path.join(session_dir, "summary.json"), "w") as f:
      json.dump({
        "route": candidate.route,
        "segments": len(candidate.paths),
        "source_files": list(candidate.source_files),
        "rows": len(rows),
        "pre_intervention_seconds": PRE_INTERVENTION_SECONDS,
        "post_intervention_seconds": POST_INTERVENTION_SECONDS,
        "route_settle_seconds": ROUTE_SETTLE_SECONDS,
      }, f, indent=2)
    _zip_and_queue(session_dir)
    return True
  except OSError:
    cloudlog.exception("experimental_longitudinal_exporter: write failed")
    return False


def export_latest_route() -> str | None:
  exported = set(_exported_routes(Params()))
  candidates = [c for c in _route_candidates(require_settled=True) if c.route not in exported]
  if not candidates:
    return None
  route = candidates[0].route
  return route if export_route(route) else None
