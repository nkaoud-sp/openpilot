"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import datetime
import os
import time
from pathlib import Path

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.layouts.settings.developer import DeveloperLayout
from openpilot.common.hardware import PC
from openpilot.common.hardware.hw import Paths
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.list_view import button_item

from openpilot.sunnypilot.autolock_commands import frame_record, LOCK_CMD, UNLOCK_CMD
from openpilot.sunnypilot.turn_signal_commands import (
  SWEEP_BASE_BYTE,
  SWEEP_BITS,
  SWEEP_CLEAR_DELAY_RECORDS,
  SWEEP_BYTES,
  SWEEP_REPEATS_PER_BYTE,
  build_turn_signal_pulse_queue,
  build_turn_signal_sweep_queue,
)
from openpilot.system.ui.sunnypilot.widgets.html_render import HtmlModalSP
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp

PREBUILT_PATH = os.path.join(Paths.comma_home(), "prebuilt") if PC else "/data/openpilot/prebuilt"
OFFROAD_CAN_QUEUE_DRAIN_INTERVAL = 0.2
TURN_SIGNAL_SWEEP_START_DELAY = OFFROAD_CAN_QUEUE_DRAIN_INTERVAL
TURN_SIGNAL_SWEEP_ACTIVE_SECONDS = SWEEP_REPEATS_PER_BYTE * 2 * OFFROAD_CAN_QUEUE_DRAIN_INTERVAL
TURN_SIGNAL_SWEEP_CLEAR_DELAY_SECONDS = SWEEP_CLEAR_DELAY_RECORDS * OFFROAD_CAN_QUEUE_DRAIN_INTERVAL
TURN_SIGNAL_SWEEP_CLEAR_SECONDS = 2 * OFFROAD_CAN_QUEUE_DRAIN_INTERVAL
TURN_SIGNAL_SWEEP_RETURN_SECONDS = OFFROAD_CAN_QUEUE_DRAIN_INTERVAL
TURN_SIGNAL_SWEEP_STEP_SECONDS = (TURN_SIGNAL_SWEEP_ACTIVE_SECONDS + TURN_SIGNAL_SWEEP_CLEAR_DELAY_SECONDS +
                                  TURN_SIGNAL_SWEEP_CLEAR_SECONDS + TURN_SIGNAL_SWEEP_RETURN_SECONDS)
TURN_SIGNAL_SWEEP_TOTAL_SECONDS = TURN_SIGNAL_SWEEP_START_DELAY + len(SWEEP_BYTES) * TURN_SIGNAL_SWEEP_STEP_SECONDS + OFFROAD_CAN_QUEUE_DRAIN_INTERVAL


class DeveloperLayoutSP(DeveloperLayout):
  def __init__(self):
    super().__init__()
    self.error_log_path = os.path.join(Paths.crash_log_root(), "error.log")
    self._turn_signal_sweep_started_at: float | None = None
    self._is_release_branch: bool = self._is_release or ui_state.params.get_bool("IsReleaseSpBranch")
    self._is_development_branch: bool = ui_state.params.get_bool("IsTestedBranch") or ui_state.params.get_bool("IsDevelopmentBranch")
    self._initialize_items()

    for item in self.items:
      self._scroller.add_widget(item)

  def _initialize_items(self):
    self.show_advanced_controls = toggle_item_sp(tr("Show Advanced Controls"),
                                                 tr("Toggle visibility of advanced sunnypilot controls.<br>This only changes the visibility of the toggles; " +
                                                    "it does not change the actual enabled/disabled state."), param="ShowAdvancedControls")

    self.enable_github_runner_toggle = toggle_item_sp(tr("GitHub Runner Service"), tr("Enables or disables the GitHub runner service."),
                                                      param="EnableGithubRunner")

    self.enable_copyparty_toggle = toggle_item_sp(tr("copyparty Service"),
                                                  tr("copyparty is a very capable file server, you can use it to download your routes, view your logs " +
                                                     "and even make some edits on some files from your browser. " +
                                                     "Requires you to connect to your comma locally via its IP address."), param="EnableCopyparty")

    self.prebuilt_toggle = toggle_item_sp(tr("Quickboot Mode"), "", param="QuickBootToggle", callback=self._on_prebuilt_toggled)

    self.error_log_btn = button_item(tr("Error Log"), tr("VIEW"), tr("View the error log for sunnypilot crashes."), callback=self._on_error_log_clicked)

    self.door_lock_test_btn = button_item(
      tr("Door Lock CAN Test"),
      tr("RUN"),
      tr("Sanity check that the panda is transmitting: queues the known-good lock then unlock frames " +
         "(0x750) via pandad's offroad ELM327 path. You should hear the doors lock, then unlock. " +
         "Toyota/Lexus only. Offroad only."),
      callback=self._on_door_lock_test_clicked,
      enabled=ui_state.is_offroad,
    )

    self.turn_signal_test_btn = button_item(
      tr("Toyota Right Signal CAN Test"),
      tr("RUN"),
      tr("Briefly flashes the RIGHT turn signal by queuing the verified Techstream active-test ON and OFF " +
         "messages (UDS 0x2F to the combination meter, 0x7C0) via pandad's offroad ELM327 path. " +
         "Toyota/Lexus only. Offroad only; ignition on so the meter actuates the lamps."),
      callback=self._on_turn_signal_test_clicked,
      enabled=ui_state.is_offroad,
    )

    self.left_signal_test_btn = button_item(
      tr("Toyota Left Signal CAN Test"),
      tr("RUN"),
      tr("Briefly flashes the LEFT turn signal by queuing the verified Techstream active-test ON and OFF " +
         "messages (UDS 0x2F to the combination meter, 0x7C0) via pandad's offroad ELM327 path. " +
         "Toyota/Lexus only. Offroad only; ignition on so the meter actuates the lamps."),
      callback=self._on_left_signal_test_clicked,
      enabled=ui_state.is_offroad,
    )

    self.hazard_signal_test_btn = button_item(
      tr("Toyota Hazard CAN Test"),
      tr("RUN"),
      tr("Briefly flashes the HAZARD lights by queuing the verified Techstream active-test ON and OFF " +
         "messages (UDS 0x2F to the combination meter, 0x7C0) via pandad's offroad ELM327 path. " +
         "Toyota/Lexus only. Offroad only; ignition on so the meter actuates the lamps."),
      callback=self._on_hazard_signal_test_clicked,
      enabled=ui_state.is_offroad,
    )

    self.signal_sequence_test_btn = button_item(
      tr("Toyota Signal Sequence Test"),
      tr("RUN"),
      tr("Runs a scripted signal sequence through the verified Techstream active-test ON and OFF messages: " +
         "left 0.8s/0.4s five times, right 0.4s/0.8s five times, then hazard 0.6s/0.6s five times. " +
         "Toyota/Lexus only. Offroad only; ignition on so the meter actuates the lamps."),
      callback=self._on_signal_sequence_test_clicked,
      enabled=ui_state.is_offroad,
    )

    self.turn_signal_sweep_btn = button_item(
      tr("Toyota Turn Signal Sweep"),
      self._turn_signal_sweep_button_text,
      tr("Sweeps the 8 possible control-state bytes for the Techstream turn-signal active test so you can " +
         "watch which lamp/output each byte activates. Keeps byte[7] set because the known right-turn " +
         "payload requires it, then sends a zeroed state after each stage. Toyota/Lexus only. Offroad only; " +
         "ignition on so the meter actuates the lamps."),
      callback=self._on_turn_signal_sweep_clicked,
      enabled=lambda: ui_state.is_offroad() and not self._is_turn_signal_sweep_running(),
    )
    self.turn_signal_sweep_btn.action_item.set_value(self._turn_signal_sweep_status_text)

    self.items: list = [self.show_advanced_controls, self.enable_github_runner_toggle, self.enable_copyparty_toggle,
                        self.prebuilt_toggle, self.error_log_btn, self.door_lock_test_btn, self.turn_signal_test_btn,
                        self.left_signal_test_btn, self.hazard_signal_test_btn, self.signal_sequence_test_btn,
                        self.turn_signal_sweep_btn,]

  @staticmethod
  def _on_prebuilt_toggled(state):
    if state:
      Path(PREBUILT_PATH).touch(exist_ok=True)
    else:
      os.remove(PREBUILT_PATH)
    ui_state.params.put_bool("QuickBootToggle", state)

  def _on_delete_confirm(self, result):
    if result == DialogResult.CONFIRM:
      if os.path.exists(self.error_log_path):
        os.remove(self.error_log_path)

  def _on_error_log_closed(self, result, log_exists):
    if result == DialogResult.CONFIRM and log_exists:
      dialog2 = ConfirmDialog(tr("Would you like to delete this log?"), tr("Yes"), tr("No"), rich=False, callback=self._on_delete_confirm)
      gui_app.push_widget(dialog2)

  def _enqueue_offroad_can(self, queue: bytes):
    # pandad drains OffroadCanQueue offroad via ELM327, one frame per 200 ms (see panda_safety.cc).
    if queue and ui_state.is_offroad():
      ui_state.params.put("OffroadCanQueue", queue)

  def _turn_signal_sweep_elapsed(self) -> float | None:
    if self._turn_signal_sweep_started_at is None:
      return None

    elapsed = time.monotonic() - self._turn_signal_sweep_started_at
    if elapsed >= TURN_SIGNAL_SWEEP_TOTAL_SECONDS:
      self._turn_signal_sweep_started_at = None
      return None
    return elapsed

  def _is_turn_signal_sweep_running(self) -> bool:
    return self._turn_signal_sweep_elapsed() is not None

  def _get_turn_signal_sweep_stage(self) -> int | None:
    elapsed = self._turn_signal_sweep_elapsed()
    if elapsed is None or elapsed < TURN_SIGNAL_SWEEP_START_DELAY:
      return None

    stage_elapsed = elapsed - TURN_SIGNAL_SWEEP_START_DELAY
    byte_index = int(stage_elapsed // TURN_SIGNAL_SWEEP_STEP_SECONDS)
    if byte_index >= len(SWEEP_BYTES):
      self._turn_signal_sweep_started_at = None
      return None

    stage_offset = stage_elapsed % TURN_SIGNAL_SWEEP_STEP_SECONDS
    return SWEEP_BYTES[byte_index] if stage_offset < TURN_SIGNAL_SWEEP_ACTIVE_SECONDS else None

  def _is_turn_signal_sweep_clearing(self) -> bool:
    elapsed = self._turn_signal_sweep_elapsed()
    if elapsed is None or elapsed < TURN_SIGNAL_SWEEP_START_DELAY:
      return False

    stage_elapsed = elapsed - TURN_SIGNAL_SWEEP_START_DELAY
    stage_offset = stage_elapsed % TURN_SIGNAL_SWEEP_STEP_SECONDS
    clear_start = TURN_SIGNAL_SWEEP_ACTIVE_SECONDS + TURN_SIGNAL_SWEEP_CLEAR_DELAY_SECONDS
    return clear_start <= stage_offset < clear_start + TURN_SIGNAL_SWEEP_CLEAR_SECONDS

  def _turn_signal_sweep_status_text(self) -> str:
    elapsed = self._turn_signal_sweep_elapsed()
    if elapsed is None:
      return ""
    if elapsed < TURN_SIGNAL_SWEEP_START_DELAY:
      return tr("starting...")

    byte_index = self._get_turn_signal_sweep_stage()
    if self._is_turn_signal_sweep_clearing():
      return f"byte[{SWEEP_BASE_BYTE}] = 0x00"
    if byte_index is None:
      return tr("waiting...")
    return f"byte[{byte_index}] + byte[{SWEEP_BASE_BYTE}] = 0x{SWEEP_BITS:02X}"

  def _turn_signal_sweep_button_text(self) -> str:
    byte_index = self._get_turn_signal_sweep_stage()
    if not self._is_turn_signal_sweep_running():
      return tr("RUN")
    if self._is_turn_signal_sweep_clearing():
      return tr("CLEAR")
    return tr("WAIT") if byte_index is None else f"BYTE {byte_index}"

  def _on_door_lock_test_confirm(self, result):
    if result == DialogResult.CONFIRM:
      # Known-good frames drained 200 ms apart: hold lock ~1 s (idempotent repeats), then unlock,
      # so you hear the lock, a pause, then the unlock.
      self._enqueue_offroad_can((frame_record(LOCK_CMD) * 6) + frame_record(UNLOCK_CMD))

  def _on_door_lock_test_clicked(self):
    content = (
      f"<h1>{tr('Door Lock CAN Test')}</h1><br>" +
      f"<p>{tr('Queues the known-good lock then unlock frames (0x750) to confirm the panda is transmitting.')} " +
      f"{tr('You should hear the doors lock, then unlock.')}</p>" +
      f"<p><b>{tr('Toyota/Lexus only. Offroad only.')}</b></p>"
    )
    dialog = ConfirmDialog(content, tr("Run"), rich=True, callback=self._on_door_lock_test_confirm)
    gui_app.push_widget(dialog)

  def _on_turn_signal_test_confirm(self, result):
    if result == DialogResult.CONFIRM:
      self._enqueue_offroad_can(build_turn_signal_pulse_queue("right"))

  def _on_left_signal_test_confirm(self, result):
    if result == DialogResult.CONFIRM:
      self._enqueue_offroad_can(build_turn_signal_pulse_queue("left"))

  def _on_hazard_signal_test_confirm(self, result):
    if result == DialogResult.CONFIRM:
      self._enqueue_offroad_can(build_turn_signal_pulse_queue("hazard"))

  def _on_signal_sequence_test_confirm(self, result):
    if result == DialogResult.CONFIRM:
      if ui_state.is_offroad():
        ui_state.params.put_bool("OffroadTurnSignalSequence", True)

  def _on_turn_signal_sweep_confirm(self, result):
    if result == DialogResult.CONFIRM:
      self._turn_signal_sweep_started_at = time.monotonic()
      self._enqueue_offroad_can(build_turn_signal_sweep_queue())

  def _on_turn_signal_test_clicked(self):
    content = (
      f"<h1>{tr('Toyota Right Signal CAN Test')}</h1><br>" +
      f"<p>{tr('Queues the verified Techstream active-test ON message, then the OFF message shortly after, to briefly flash the RIGHT turn signal.')}</p>" +
      f"<p><b>{tr('Toyota/Lexus only')}</b> {tr('(right bit value verified on 2019+ Lexus ES). Offroad only, ignition on so the meter actuates the lamps.')}</p>"
    )
    dialog = ConfirmDialog(content, tr("Run"), rich=True, callback=self._on_turn_signal_test_confirm)
    gui_app.push_widget(dialog)

  def _on_left_signal_test_clicked(self):
    content = (
      f"<h1>{tr('Toyota Left Signal CAN Test')}</h1><br>" +
      f"<p>{tr('Queues the verified Techstream active-test ON message, then the OFF message shortly after, to briefly flash the LEFT turn signal.')}</p>" +
      f"<p><b>{tr('Toyota/Lexus only')}</b> {tr('(left bit value verified from testing). Offroad only, ignition on so the meter actuates the lamps.')}</p>"
    )
    dialog = ConfirmDialog(content, tr("Run"), rich=True, callback=self._on_left_signal_test_confirm)
    gui_app.push_widget(dialog)

  def _on_hazard_signal_test_clicked(self):
    content = (
      f"<h1>{tr('Toyota Hazard CAN Test')}</h1><br>" +
      f"<p>{tr('Queues the verified Techstream active-test ON message, then the OFF message shortly after, to briefly flash the HAZARD lights.')}</p>" +
      f"<p><b>{tr('Toyota/Lexus only')}</b> {tr('(hazard bit value verified from testing). Offroad only, ignition on so the meter actuates the lamps.')}</p>"
    )
    dialog = ConfirmDialog(content, tr("Run"), rich=True, callback=self._on_hazard_signal_test_confirm)
    gui_app.push_widget(dialog)

  def _on_signal_sequence_test_clicked(self):
    content = (
      f"<h1>{tr('Toyota Signal Sequence Test')}</h1><br>" +
      f"<p>{tr('Runs left turn 0.8s on / 0.4s off five times, right turn 0.4s on / 0.8s off five times, then hazard 0.6s on / 0.6s off five times.')}</p>" +
      f"<p><b>{tr('Toyota/Lexus only')}</b> {tr('Offroad only, ignition on so the meter actuates the lamps.')}</p>"
    )
    dialog = ConfirmDialog(content, tr("Run"), rich=True, callback=self._on_signal_sequence_test_confirm)
    gui_app.push_widget(dialog)

  def _on_turn_signal_sweep_clicked(self):
    content = (
      f"<h1>{tr('Toyota Turn Signal Sweep')}</h1><br>" +
      f"<p>{tr('Queues byte[0] through byte[6] of the Techstream active-test state while keeping byte[7] set, holding each for about six seconds. About one second after each stage it sends a zeroed state so byte[7] returns to 0x00. Watch the car to map which output each byte activates.')}</p>" +
      f"<p><b>{tr('Toyota/Lexus only')}</b> {tr('(byte[3] plus byte[7] is the verified right-turn payload on 2019+ Lexus ES; other sweep outputs are experimental). Offroad only, ignition on so the meter actuates the lamps.')}</p>"
    )
    dialog = ConfirmDialog(content, tr("Run"), rich=True, callback=self._on_turn_signal_sweep_confirm)
    gui_app.push_widget(dialog)

  def _on_error_log_clicked(self):
    text = ""
    if os.path.exists(self.error_log_path):
      text = f"<b>{datetime.datetime.fromtimestamp(os.path.getmtime(self.error_log_path)).strftime('%d-%b-%Y %H:%M:%S').upper()}</b><br><br>"
      try:
        with open(self.error_log_path) as file:
          text += file.read()
      except Exception:
        pass
    dialog = HtmlModalSP(text=text, callback=lambda result: self._on_error_log_closed(result, os.path.exists(self.error_log_path)))
    gui_app.push_widget(dialog)

  def _update_state(self):
    disable_updates = ui_state.params.get_bool("DisableUpdates")
    show_advanced = ui_state.params.get_bool("ShowAdvancedControls")

    if (prebuilt_file := os.path.exists(PREBUILT_PATH)) != ui_state.params.get_bool("QuickBootToggle"):
      ui_state.params.put_bool("QuickBootToggle", prebuilt_file)
      self.prebuilt_toggle.action_item.set_state(prebuilt_file)

    self.prebuilt_toggle.set_visible(show_advanced and not (self._is_release_branch or self._is_development_branch))
    self.prebuilt_toggle.action_item.set_enabled(disable_updates)

    if disable_updates:
      self.prebuilt_toggle.set_description(tr("When toggled on, this creates a prebuilt file to allow accelerated boot times. When toggled off, it " +
                                              "removes the prebuilt file so compilation of locally edited cpp files can be made."))
    else:
      self.prebuilt_toggle.set_description(tr("Quickboot mode requires updates to be disabled.<br>Enable 'Disable Updates' in the Software panel first."))

    self.enable_copyparty_toggle.set_visible(show_advanced)
    self.enable_github_runner_toggle.set_visible(show_advanced and not self._is_release_branch)
    self.error_log_btn.set_visible(not self._is_release_branch)
    # reverse-engineering test tools: keep them out of release builds and behind Show Advanced Controls
    self.door_lock_test_btn.set_visible(show_advanced and not self._is_release_branch)
    self.turn_signal_test_btn.set_visible(show_advanced and not self._is_release_branch)
    self.left_signal_test_btn.set_visible(show_advanced and not self._is_release_branch)
    self.hazard_signal_test_btn.set_visible(show_advanced and not self._is_release_branch)
    self.signal_sequence_test_btn.set_visible(show_advanced and not self._is_release_branch)
    self.turn_signal_sweep_btn.set_visible(show_advanced and not self._is_release_branch)
