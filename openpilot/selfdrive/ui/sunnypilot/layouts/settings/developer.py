"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import datetime
import os
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
from openpilot.sunnypilot.turn_signal_commands import build_turn_signal_pulses
from openpilot.system.ui.sunnypilot.widgets.html_render import HtmlModalSP
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp

PREBUILT_PATH = os.path.join(Paths.comma_home(), "prebuilt") if PC else "/data/openpilot/prebuilt"


class DeveloperLayoutSP(DeveloperLayout):
  def __init__(self):
    super().__init__()
    self.error_log_path = os.path.join(Paths.crash_log_root(), "error.log")
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

    ts_desc = tr("Flashes the turn signal by queuing the Techstream active test (UDS 0x2F to the combination " +
                 "meter, 0x7C0) via pandad's offroad ELM327 path. Toyota/Lexus only, verified on 2019+ Lexus ES. " +
                 "Offroad only; ignition on so the meter actuates the lamps.")
    self.turn_signal_left_btn = button_item(tr("Turn Signal LEFT CAN Test"), tr("RUN"), ts_desc,
                                            callback=lambda: self._on_turn_signal_clicked("left"), enabled=ui_state.is_offroad)
    self.turn_signal_right_btn = button_item(tr("Turn Signal RIGHT CAN Test"), tr("RUN"), ts_desc,
                                             callback=lambda: self._on_turn_signal_clicked("right"), enabled=ui_state.is_offroad)
    self.hazard_test_btn = button_item(
      tr("Hazard CAN Test"), tr("RUN"),
      tr("Same active test with both turn-signal bits set (predicted, unverified). ") + ts_desc,
      callback=lambda: self._on_turn_signal_clicked("hazard"), enabled=ui_state.is_offroad)

    self.items: list = [self.show_advanced_controls, self.enable_github_runner_toggle, self.enable_copyparty_toggle,
                        self.prebuilt_toggle, self.error_log_btn, self.door_lock_test_btn,
                        self.turn_signal_left_btn, self.turn_signal_right_btn, self.hazard_test_btn,]

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

  def _on_turn_signal_clicked(self, side: str):
    names = {"left": tr("LEFT turn signal"), "right": tr("RIGHT turn signal"), "hazard": tr("hazard lights")}
    predicted = tr(" (predicted, unverified)") if side == "hazard" else ""
    content = (
      f"<h1>{tr('Turn Signal CAN Test')}</h1><br>" +
      f"<p>{tr('Queues the Techstream active test (UDS 0x2F to the combination meter, 0x7C0) to pulse the')} " +
      f"{names[side]}{predicted} {tr('four times: one single message held 0.8 s first, then on for 2 s, 1 s, 0.5 s (refreshed), off in between.')}</p>" +
      f"<p><b>{tr('Toyota/Lexus only')}</b> {tr('(verified on 2019+ Lexus ES). Offroad only, ignition on so the meter actuates the lamps.')}</p>"
    )
    dialog = ConfirmDialog(content, tr("Run"), rich=True,
                           callback=lambda result: self._enqueue_offroad_can(build_turn_signal_pulses(side))
                           if result == DialogResult.CONFIRM else None)
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
    can_test_visible = show_advanced and not self._is_release_branch
    for btn in (self.door_lock_test_btn, self.turn_signal_left_btn, self.turn_signal_right_btn, self.hazard_test_btn):
      btn.set_visible(can_test_visible)
