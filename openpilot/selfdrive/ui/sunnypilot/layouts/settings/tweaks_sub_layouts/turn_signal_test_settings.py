"""Toyota/Lexus turn signal diagnostic test controls."""
from collections.abc import Callable
import time

import pyray as rl
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.sunnypilot.autolock_commands import LOCK_CMD, UNLOCK_CMD, frame_record
from openpilot.sunnypilot.broadcast_lighting_commands import (
  BLINKER_D3_LEFT,
  BLINKER_D3_RIGHT,
  blinker_record,
  hazard_record,
)
from openpilot.sunnypilot.turn_signal_probe_commands import full_sweep
from openpilot.system.ui.sunnypilot.widgets.list_view import ListItemSP, button_item_sp, multiple_button_item_sp, option_item_sp
from openpilot.system.ui.widgets import DialogResult, Widget
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog, alert_dialog
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller

OFFROAD_CAN_QUEUE_PARAM = "OffroadCanQueue"


SIGNALS = ("left", "right", "hazard")
SIGNAL_LABELS = (lambda: tr("Left"), lambda: tr("Right"), lambda: tr("Hazard"))

PROBE_REQUEST_PARAM = "TurnSignalProbeRequest"
PROBE_STATUS_PARAM = "TurnSignalProbeStatus"
PROBE_START_INDEX_PARAM = "TurnSignalProbeStartIndex"
PROBE_ACTIVE_STATES = ("baseline", "running", "active")

# Rough wall-clock per candidate (SEND_DRAIN_S + OBSERVE_S in turn_signal_probe.py), for the ETA.
PROBE_PER_CANDIDATE_S = 3.5


def _toyota_available() -> bool:
  return ui_state.CP is not None and ui_state.CP.brand == "toyota"


class TurnSignalTestSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()

    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)
    self._selected_signal = 0
    self._status = ""
    self._send_status = ""
    self._probe_status: dict = {}
    self._probe_mode = ""  # last mode started, so the summary/result can word capture vs probe
    self._sweep_total = sum(1 for _ in full_sweep())  # candidate count, for the start-index range

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._direction = multiple_button_item_sp(
      title=lambda: tr("Signal"),
      description=lambda: tr("Select which exterior turn signal active test to run."),
      buttons=SIGNAL_LABELS,
      selected_index=self._selected_signal,
      button_width=250,
      callback=self._set_signal,
      inline=False,
    )
    self._duration = option_item_sp(
      title=lambda: tr("Duration"),
      description=lambda: tr("Approximate time to hold the selected signal before returning control to the car."),
      param="ToyotaTurnSignalTestDurationMs",
      min_value=500,
      max_value=5000,
      value_change_step=250,
      label_callback=lambda value: f"{value / 1000:.2f} s",
      inline=True,
    )
    self._run_test = button_item_sp(
      title=lambda: tr("Run Test"),
      button_text=lambda: tr("RUN"),
      description=lambda: tr("Requests the selected Toyota/Lexus turn signal command through the live car controller path. Onroad only."),
      callback=self._run_turn_signal_test,
      enabled=lambda: not ui_state.is_offroad() and _toyota_available(),
    )
    self._run_test.set_right_value(lambda: self._status)

    # --- Send-path check (offroad) ---------------------------------------------------------------
    # A known-good control: fire the exact door lock/unlock command the auto-lock feature uses,
    # through the same OffroadCanQueue -> pandad -> ELM327 path the probe uses. If the door locks,
    # the send path works, so a probe that finds nothing means the signal command isn't in the
    # swept range -- not that the plumbing is broken.
    self._lock_test = button_item_sp(
      title=lambda: tr("Send-Path Check"),
      button_text=lambda: tr("LOCK"),
      description=lambda: tr("Offroad only. Sends the known door-lock command through the same path the probe " +
                             "uses. If the door locks, the send path works."),
      callback=self._test_lock,
      enabled=lambda: self._probe_enabled(),
    )
    self._lock_test.set_right_value(lambda: self._send_status)

    self._unlock_test = button_item_sp(
      title=lambda: tr("Send-Path Check"),
      button_text=lambda: tr("UNLOCK"),
      description=lambda: tr("Sends the known door-unlock command through the same path."),
      callback=self._test_unlock,
      enabled=lambda: self._probe_enabled(),
    )

    # --- Broadcast lighting (operational frames, from Austin Fisk's 2023 RAV4 write-up) -----------
    # These are ordinary bus-0 broadcast frames, not the 0x750 diagnostic path -- so they are not
    # speed-gated. 0x623 is Fisk's demonstrated hazard flash; the 0x614 left/right frames are
    # experimental (that address is normally broadcast by the body ECU itself).
    self._flash_hazards = button_item_sp(
      title=lambda: tr("Broadcast Lighting"),
      button_text=lambda: tr("HAZARDS"),
      description=lambda: tr("Offroad only. Injects the known-good 0x623 hazard-flash frame. If the hazards " +
                             "flash, operational lighting injection works on your car -- the whole point."),
      callback=self._flash_hazard,
      enabled=lambda: self._probe_enabled(),
    )
    self._flash_hazards.set_right_value(lambda: self._send_status)

    self._signal_left = button_item_sp(
      title=lambda: tr("Broadcast Lighting"),
      button_text=lambda: tr("LEFT"),
      description=lambda: tr("Injects the 0x614 left-signal command (29 80 00 10 ...), the format an ESORICS-2024 " +
                             "study injected on a Toyota Corolla via an OBD dongle. Sent as a ~5 s burst."),
      callback=self._signal_left_cb,
      enabled=lambda: self._probe_enabled(),
    )
    self._signal_right = button_item_sp(
      title=lambda: tr("Broadcast Lighting"),
      button_text=lambda: tr("RIGHT"),
      description=lambda: tr("Injects the 0x614 right-signal command (29 80 00 20 ...). Sent as a ~5 s burst."),
      callback=self._signal_right_cb,
      enabled=lambda: self._probe_enabled(),
    )
    # set_right_value never renders on a button row, so carry the send/inject feedback in a plain
    # title row that does render every frame.
    self._send_status_row = ListItemSP(title=lambda: self._send_status_text())

    self._capture = button_item_sp(
      title=lambda: tr("Capture Lighting Frames"),
      button_text=lambda: tr("CAPTURE"),
      description=lambda: tr("Offroad only. Records the idle body-ECU bus for 5 s, then for 30 s flags any frame " +
                             "that changes. Press CAPTURE, wait for 'operate...', then work the hazard button, " +
                             "turn stalk and fob lock. VIEW shows the frames that moved. Find your car's own command."),
      callback=self._run_capture,
      enabled=lambda: self._probe_enabled(),
    )

    # --- Signal command discovery probe (offroad) ------------------------------------------------
    self._probe_shortlist = button_item_sp(
      title=lambda: tr("Signal Discovery Probe"),
      button_text=lambda: tr("RUN"),
      description=lambda: tr("Offroad only. Sends candidate body-ECU commands and watches the car's blinker state " +
                             "to find one that lights the signals without the speed-locked diagnostic test. Try " +
                             "this shortlist first; it takes about a minute."),
      callback=self._run_shortlist,
      enabled=lambda: self._probe_enabled(),
    )

    self._probe_lattice = button_item_sp(
      title=lambda: tr("Lattice Probe"),
      button_text=lambda: tr("LATTICE"),
      description=lambda: tr("Every body-ECU function found so far sits on a LID where lid mod 8 == 1 (windows " +
                             "0x01, locks 0x11, sunshade 0x19, mirrors 0x21). This probes only that lattice, " +
                             "covering the whole LID range in ~14 minutes instead of ~90. Park EMPTY."),
      callback=self._confirm_lattice,
      enabled=lambda: self._probe_enabled(),
    )

    self._sweep_start = option_item_sp(
      title=lambda: tr("Sweep Start Index"),
      description=lambda: tr("Where the sweep begins, so a long run can be split across sessions. It resumes here " +
                             "automatically after a stop, and resets to 0 when a full sweep finishes."),
      param=PROBE_START_INDEX_PARAM,
      min_value=0,
      max_value=self._sweep_total,
      value_change_step=50,
      label_callback=lambda value: f"{value} / {self._sweep_total}",
      inline=True,
    )

    self._probe_full = button_item_sp(
      title=lambda: tr("Full Sweep"),
      button_text=lambda: tr("SWEEP"),
      description=lambda: tr("If the shortlist finds nothing: a blind sweep of every body-ECU output. Park EMPTY " +
                             "with ALL WINDOWS DOWN and stay clear of the mirrors. Takes up to ~90 minutes."),
      callback=self._confirm_full,
      enabled=lambda: self._probe_enabled(),
    )

    # A button row draws its action on the right, so set_right_value never renders there; carry the
    # live progress in this dedicated row's title instead (titles are resolved and drawn each frame).
    self._probe_progress = ListItemSP(title=lambda: self._probe_progress_text())

    self._probe_stop = button_item_sp(
      title=lambda: tr("Stop Probe"),
      button_text=lambda: tr("STOP"),
      description=lambda: tr("Stop a running probe."),
      callback=self._stop_probe,
      enabled=lambda: self._probe_running(),
    )

    self._probe_result = button_item_sp(
      title=lambda: tr("Probe Result"),
      button_text=lambda: tr("VIEW"),
      description=lambda: tr("Show the last probe's result and any commands that worked."),
      callback=self._show_result,
      enabled=lambda: bool(self._probe_status),
    )

    return [
      self._direction,
      self._duration,
      self._run_test,
      self._lock_test,
      self._unlock_test,
      self._flash_hazards,
      self._signal_left,
      self._signal_right,
      self._send_status_row,
      self._capture,
      self._probe_shortlist,
      self._probe_lattice,
      self._sweep_start,
      self._probe_full,
      self._probe_progress,
      self._probe_stop,
      self._probe_result,
    ]

  def _set_signal(self, selected_signal: int):
    self._selected_signal = selected_signal
    self._status = ""

  def _run_turn_signal_test(self):
    if ui_state.is_offroad() or not _toyota_available():
      self._status = tr("Unavailable")
      return

    signal = SIGNALS[self._selected_signal]
    duration_ms = int(ui_state.params.get("ToyotaTurnSignalTestDurationMs", return_default=True))
    ui_state.params.put("ToyotaTurnSignalTestRequest", {
      "signal": signal,
      "durationMs": duration_ms,
      "requestId": time.monotonic_ns(),
    })
    self._status = tr("Queued")

  def _send_status_text(self) -> str:
    return tr("Last send: {}").format(self._send_status) if self._send_status else tr("Last send: none")

  # --- send-path check ---------------------------------------------------------------------------
  def _send_offroad_frame(self, record: bytes, label: str, repeat: int = 1):
    if not self._probe_enabled():
      self._send_status = tr("unavailable (need offroad + Toyota)")
      return
    # pandad drains OffroadCanQueue one frame per ~200ms via ELM327. A diagnostic one-shot (lock)
    # needs one record; a broadcast command has to be repeated to be a sustained cyclic message, so
    # queue `repeat` copies (~200ms apart). Show the exact address+bytes so a non-effect can be told
    # apart from a non-send.
    addr = (record[0] << 8) | record[1]
    ui_state.params.put(OFFROAD_CAN_QUEUE_PARAM, record * max(1, repeat))
    suffix = f" x{repeat}" if repeat > 1 else ""
    self._send_status = f"{label} 0x{addr:X} [{record[4:].hex()}]{suffix}"

  def _test_lock(self):
    self._send_offroad_frame(frame_record(LOCK_CMD), tr("Lock"))

  def _test_unlock(self):
    self._send_offroad_frame(frame_record(UNLOCK_CMD), tr("Unlock"))

  def _flash_hazard(self):
    # Broadcast commands are cyclic; a single frame is ignored. Send a ~5 s burst (~25 x 200 ms).
    self._send_offroad_frame(hazard_record(), tr("Hazards"), repeat=25)

  def _signal_left_cb(self):
    self._send_offroad_frame(blinker_record(BLINKER_D3_LEFT), tr("Left"), repeat=25)

  def _signal_right_cb(self):
    self._send_offroad_frame(blinker_record(BLINKER_D3_RIGHT), tr("Right"), repeat=25)

  # --- probe controls ----------------------------------------------------------------------------
  def _probe_enabled(self) -> bool:
    # Opposite of the active test: the probe path only works offroad, and never while one is running.
    return ui_state.is_offroad() and _toyota_available() and not self._probe_running()

  def _probe_running(self) -> bool:
    return self._probe_status.get("state") in PROBE_ACTIVE_STATES

  def _start_probe(self, mode: str):
    self._probe_mode = mode
    request = {"mode": mode, "requestId": time.monotonic_ns()}
    if mode == "full":
      # Read the saved index fresh (the daemon may have advanced it past what the control shows).
      request["start"] = int(ui_state.params.get(PROBE_START_INDEX_PARAM, return_default=True))
    ui_state.params.put(PROBE_REQUEST_PARAM, request)
    # Show immediate feedback until the daemon publishes its first status.
    self._probe_status = {"state": "baseline", "message": tr("Starting..."), "hits": []}

  def _run_shortlist(self):
    if self._probe_enabled():
      self._start_probe("shortlist")

  def _run_capture(self):
    if self._probe_enabled():
      self._start_probe("capture")

  def _confirm_blind_probe(self, mode: str, confirm_text: str):
    """Both blind modes poke unknown body outputs, so both go behind the same empty-car confirm."""
    if not self._probe_enabled():
      return

    def on_result(result: DialogResult):
      if result == DialogResult.CONFIRM:
        self._start_probe(mode)

    dialog = ConfirmDialog(
      tr("This pokes body-ECU outputs blindly. Make sure the car is EMPTY, ALL WINDOWS are DOWN, " +
         "and nobody is near the mirrors. Continue?"),
      confirm_text,
      callback=on_result,
    )
    gui_app.push_widget(dialog)

  def _confirm_full(self):
    self._confirm_blind_probe("full", tr("Start Sweep"))

  def _confirm_lattice(self):
    self._confirm_blind_probe("structured", tr("Start Lattice"))

  def _stop_probe(self):
    ui_state.params.remove(PROBE_REQUEST_PARAM)

  def _show_result(self):
    if not self._probe_status:
      return
    hits = self._probe_status.get("hits", [])
    if self._probe_mode == "capture":
      if hits:
        body = tr("Frames that changed (bus addr data):") + "\n\n" + "\n".join(hits)
      else:
        body = tr("No frames changed. Make sure the bus was awake and you operated the controls.")
    elif hits:
      body = tr("Commands that lit the signals:") + "\n\n" + "\n".join(hits)
    else:
      msg = self._probe_status.get("message", "")
      body = tr("No command lit the signals.") + (f"\n\n{msg}" if msg else "")
    gui_app.push_widget(alert_dialog(body))

  def _probe_summary(self) -> str:
    state = self._probe_status.get("state")
    if state is None:
      return ""
    if state == "baseline":
      return tr("Idle baseline...") if self._probe_mode == "capture" else tr("Baseline...")
    if state == "active":
      return tr("Recording: {} changed").format(self._probe_status.get("index", 0))
    if state == "running":
      idx, total = self._probe_status.get("index", 0), self._probe_status.get("total", 0)
      hits = len(self._probe_status.get("hits", []))
      pct = int(idx * 100 / total) if total else 0
      summary = f"{idx}/{total} ({pct}%)"
      eta = self._format_eta((total - idx) * PROBE_PER_CANDIDATE_S)
      if eta:
        summary += f" • {eta} left"
      if hits:
        summary += f" • {hits} hit"
      return summary
    if state == "done":
      hits = len(self._probe_status.get("hits", []))
      if self._probe_mode == "capture":
        return tr("{} changed").format(hits) if hits else tr("Nothing changed")
      return tr("Found {}").format(hits) if hits else tr("None found")
    if state == "aborted":
      return tr("Stopped")
    if state == "error":
      return tr("Error")
    return ""

  def _probe_progress_text(self) -> str:
    summary = self._probe_summary()
    if summary:
      return tr("Probe: {}").format(summary)
    # Idle: surface the saved sweep resume point (read live, so it reflects the daemon's last stop).
    start = int(ui_state.params.get(PROBE_START_INDEX_PARAM, return_default=True))
    if start > 0:
      return tr("Probe: idle (sweep resumes at {}/{})").format(start, self._sweep_total)
    return tr("Probe: idle")

  @staticmethod
  def _format_eta(seconds: float) -> str:
    # Only show an ETA worth showing; the tiny shortlist finishes before it matters.
    seconds = int(seconds)
    if seconds < 60:
      return ""
    minutes = seconds // 60
    return f"~{minutes}m"

  def _refresh_probe_status(self):
    # The daemon publishes progress/results here; it clears the request when done but leaves the
    # status, so a finished run's result stays visible. get() decodes the JSON param to a dict (or
    # None when unset); keep the last value when it's empty.
    status = ui_state.params.get(PROBE_STATUS_PARAM)
    if isinstance(status, dict):
      self._probe_status = status

  def _render(self, rect):
    self._back_button.set_position(self._rect.x, self._rect.y + 20)
    self._back_button.render()

    content_rect = rl.Rectangle(rect.x, rect.y + self._back_button.rect.height + 40,
                                rect.width, rect.height - self._back_button.rect.height - 40)
    self._scroller.render(content_rect)

  def _update_state(self):
    super()._update_state()
    self._refresh_probe_status()

  def show_event(self):
    self._status = ""
    self._send_status = ""
    self._refresh_probe_status()
    # OptionControlSP reads its param only at construction, so re-sync the start index from the param
    # here; the daemon advances it after a stopped or finished sweep.
    self._sweep_start.action_item.current_value = int(ui_state.params.get(PROBE_START_INDEX_PARAM, return_default=True))
    self._scroller.show_event()

  def hide_event(self):
    self._scroller.hide_event()
