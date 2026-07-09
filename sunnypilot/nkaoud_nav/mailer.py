#!/usr/bin/env python3
"""
nkaoud_nav navigation-log mailer.

Two roles in one module:

  1. Library. `queue_log` / `send_pending_log` / `smtp_test` are used by the
     daemon below and by the settings web form's "Test" button.

  2. Daemon (`main`). A tiny always-on process (registered in
     system/manager/process_config.py, gated on NkaoudNavEnabled) that watches
     deviceState.started. The navd process only runs while onroad -- it writes
     the per-drive CSV and stores its path in NkaoudNavCurrentLog -- but it is
     killed the moment the drive ends, so it cannot do the send itself. This
     daemon survives offroad, detects the drive-end transition, queues the log,
     and retries the SMTP send until the network is up. After a successful send
     every CSV in the log directory is deleted.

SMTP credentials come from the NkaoudNavEmailConfig param, a single JSON object
entered through the settings web form (sunnypilot.nkaoud_nav.token_server).
Nothing secret is committed to the repo.
"""
from __future__ import annotations

import glob
import json
import os
import smtplib
import ssl
import time
from email.message import EmailMessage

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

EMAIL_CONFIG_PARAM = "NkaoudNavEmailConfig"
CURRENT_LOG_PARAM = "NkaoudNavCurrentLog"
LAST_LOG_PARAM = "NkaoudNavLastDriveLog"
PENDING_LOG_PARAM = "NkaoudNavEmailPendingLog"
STATUS_PARAM = "NkaoudNavEmailLastStatus"
AUTO_EMAIL_PARAM = "NkaoudNavAutoEmail"

# Keep in sync with NAV_LOG_DIR in navd.py.
NAV_LOG_DIR = os.environ.get("NKAOUD_NAV_LOG_DIR", "/data/media/0/nkaoud_nav_logs")

EMAIL_SUBJECT = "nkaoud_nav navigation log"
EMAIL_BODY = (
  "Attached is the latest nkaoud_nav navigation maneuver log.\n\n"
  "This message was generated automatically when the drive ended."
)
DEFAULT_SMTP_HOST = "smtp-relay.brevo.com"
DEFAULT_SMTP_PORT = 587
SMTP_TIMEOUT = 20
SEND_RETRY_SECONDS = 60


def _status_prefix() -> str:
  return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def set_status(message: str) -> None:
  Params().put(STATUS_PARAM, f"{_status_prefix()} - {message}")


def _read_email_config() -> dict:
  raw = (Params().get(EMAIL_CONFIG_PARAM) or "").strip()
  if not raw:
    return {}
  try:
    cfg = json.loads(raw)
  except ValueError:
    cloudlog.exception("nkaoud_nav mailer: NkaoudNavEmailConfig is not valid JSON")
    return {}
  return cfg if isinstance(cfg, dict) else {}


def _validate_config(cfg: dict) -> list[str]:
  """Return a list of missing/invalid config problems (empty == OK)."""
  problems = []
  if not str(cfg.get("from", "")).strip():
    problems.append("from address")
  if not str(cfg.get("to", "")).strip():
    problems.append("recipient address")

  login = str(cfg.get("login", "")).strip()
  password = str(cfg.get("password", "")).strip()
  if bool(login) != bool(password):
    problems.append("matching SMTP login/password")

  try:
    int(cfg.get("port", DEFAULT_SMTP_PORT) or DEFAULT_SMTP_PORT)
  except (TypeError, ValueError):
    problems.append("valid port")

  return problems


def _send_via_smtp(cfg: dict, message: EmailMessage) -> None:
  host = str(cfg.get("host", "")).strip() or DEFAULT_SMTP_HOST
  port = int(cfg.get("port", DEFAULT_SMTP_PORT) or DEFAULT_SMTP_PORT)
  login = str(cfg.get("login", "")).strip()
  password = str(cfg.get("password", "")).strip()
  context = ssl.create_default_context()

  if port == 465:
    with smtplib.SMTP_SSL(host, port, timeout=SMTP_TIMEOUT, context=context) as server:
      if login and password:
        server.login(login, password)
      server.send_message(message)
  else:
    with smtplib.SMTP(host, port, timeout=SMTP_TIMEOUT) as server:
      server.ehlo()
      if port in (587, 2525):
        server.starttls(context=context)
        server.ehlo()
      if login and password:
        server.login(login, password)
      server.send_message(message)


def _build_message(cfg: dict, log_path: str, subject: str, body: str) -> EmailMessage:
  message = EmailMessage()
  message["Subject"] = subject
  message["From"] = str(cfg["from"]).strip()
  message["To"] = str(cfg["to"]).strip()
  message.set_content(body)
  if log_path:
    with open(log_path, "rb") as log_file:
      message.add_attachment(
        log_file.read(), maintype="text", subtype="csv",
        filename=os.path.basename(log_path),
      )
  return message


def _purge_all_logs() -> int:
  """Delete every CSV in the log directory. Returns how many were removed."""
  removed = 0
  for path in glob.glob(os.path.join(NAV_LOG_DIR, "*.csv")):
    try:
      os.remove(path)
      removed += 1
    except OSError:
      cloudlog.exception(f"nkaoud_nav mailer: failed to delete {path}")
  return removed


def _get_pending(params: Params) -> list[str]:
  """The queue of logs awaiting email, oldest first. Stored as a JSON list in
  PENDING_LOG_PARAM (which is PERSISTENT so it survives reboots)."""
  raw = (params.get(PENDING_LOG_PARAM) or "").strip()
  if not raw:
    return []
  try:
    val = json.loads(raw)
  except ValueError:
    # Back-compat: an earlier version stored a single bare path.
    return [raw]
  if isinstance(val, list):
    return [str(p) for p in val if str(p).strip()]
  if isinstance(val, str) and val.strip():
    return [val.strip()]
  return []


def _set_pending(params: Params, paths: list[str]) -> None:
  if paths:
    params.put(PENDING_LOG_PARAM, json.dumps(paths))
  else:
    params.remove(PENDING_LOG_PARAM)


def queue_log(log_path: str | None) -> bool:
  """Append a completed drive's log to the pending queue. Multiple drives can
  accumulate (e.g. no network across several drives / reboots); each is kept
  until it is individually emailed."""
  params = Params()
  if not log_path or not os.path.isfile(log_path):
    set_status("No navigation log was captured for the last drive")
    return False
  pending = _get_pending(params)
  if log_path not in pending:
    pending.append(log_path)
    _set_pending(params, pending)
  set_status(f"Queued {os.path.basename(log_path)} ({len(pending)} pending)")
  return True


def send_pending_log() -> bool:
  """Email every queued log, deleting each ONLY after its own send succeeds.
  On the first failure, stop and leave the rest queued for the next retry, so
  no log is ever deleted without having been sent. Returns True once the queue
  is fully drained."""
  params = Params()
  pending = _get_pending(params)
  if not pending:
    return False

  cfg = _read_email_config()
  if not cfg:
    set_status("Email not configured yet (open Email settings to set it up)")
    return False

  problems = _validate_config(cfg)
  if problems:
    set_status(f"Email config incomplete: {', '.join(problems)}")
    return False

  remaining = list(pending)
  sent = 0
  for path in pending:
    if not os.path.isfile(path):
      # Log vanished (manually deleted / cleaned); drop it from the queue.
      remaining.remove(path)
      _set_pending(params, remaining)
      continue

    try:
      subject = f"{EMAIL_SUBJECT} - {os.path.basename(path)}"
      message = _build_message(cfg, path, subject, EMAIL_BODY)
      _send_via_smtp(cfg, message)
    except Exception:  # noqa: BLE001 -- surface any SMTP/network failure as a retry
      cloudlog.exception("nkaoud_nav mailer: send failed")
      set_status(f"Sent {sent}; {len(remaining)} still queued (will retry) "
                 f"-- {os.path.basename(path)} failed")
      return False

    # Delete this log only now that its own email has been sent.
    try:
      os.remove(path)
    except OSError:
      cloudlog.exception(f"nkaoud_nav mailer: could not delete {path}")
    remaining.remove(path)
    _set_pending(params, remaining)
    sent += 1

  # Queue fully drained: every queued log was emailed. Now clear ALL CSV logs
  # on disk regardless -- including any that were never queued (orphans, or
  # logs kept from a drive when auto-email was off).
  purged = _purge_all_logs()
  set_status(f"Sent {sent} log file(s); deleted all {sent + purged} CSV log(s)")
  return True


def smtp_test(config_json: str) -> dict:
  """Web-form Test handler: validate config, then connect + login + send a
  test email so the user can confirm their SMTP settings before saving."""
  try:
    cfg = json.loads(config_json)
  except ValueError as e:
    return {"ok": False, "error": f"invalid JSON: {e}"}
  if not isinstance(cfg, dict):
    return {"ok": False, "error": "config must be a JSON object"}

  problems = _validate_config(cfg)
  if problems:
    return {"ok": False, "error": f"missing/invalid: {', '.join(problems)}"}

  try:
    subject = f"{EMAIL_SUBJECT} - test {_status_prefix()}"
    body = (
      "This is a test email from nkaoud_nav. If you received it, your SMTP "
      "settings work and navigation logs will be delivered here."
    )
    message = _build_message(cfg, "", subject, body)
    _send_via_smtp(cfg, message)
  except Exception as e:  # noqa: BLE001 -- report anything back to the page
    return {"ok": False, "error": f"{type(e).__name__}: {e}"}

  return {"ok": True, "message": f"Test email sent to {str(cfg['to']).strip()}."}


def _handle_drive_end(params: Params) -> None:
  current = (params.get(CURRENT_LOG_PARAM) or "").strip()
  if current:
    params.put(LAST_LOG_PARAM, current)

  if params.get_bool(AUTO_EMAIL_PARAM):
    queue_log(current)
  elif current:
    set_status(f"Saved {os.path.basename(current)}")
  else:
    set_status("No navigation log was captured for the last drive")

  params.remove(CURRENT_LOG_PARAM)


def main() -> None:
  # Imported here (not at module top) so the settings UI can reuse smtp_test
  # via a lightweight import without pulling in cereal/messaging.
  import cereal.messaging as messaging
  from openpilot.common.realtime import Ratekeeper

  params = Params()
  sm = messaging.SubMaster(['deviceState'])
  rk = Ratekeeper(2.0)

  started_prev = False
  last_send_attempt = 0.0

  while True:
    sm.update(0)
    started = bool(sm['deviceState'].started)

    if started_prev and not started:
      _handle_drive_end(params)
    started_prev = started

    # Retry a queued send while offroad until the network comes up.
    if not started and params.get_bool(AUTO_EMAIL_PARAM):
      if (params.get(PENDING_LOG_PARAM) or "").strip():
        if time.monotonic() - last_send_attempt >= SEND_RETRY_SECONDS:
          send_pending_log()
          last_send_attempt = time.monotonic()

    rk.keep_time()


if __name__ == "__main__":
  main()
