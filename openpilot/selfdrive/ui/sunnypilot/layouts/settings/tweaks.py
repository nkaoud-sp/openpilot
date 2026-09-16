"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import time
from collections.abc import Callable
from enum import IntEnum
import subprocess
import threading

from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.auto_lock_settings import AutoLockSettingsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.dynamic_follow_settings import DynamicFollowSettingsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.launch_assist_settings import LaunchAssistSettingsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.park_assist_settings import ParkAssistSettingsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.speed_assist_settings import SpeedAssistSettingsLayout
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.hazard_flash import (
  build_hazard_flash_frames,
  build_hazard_stop_script,
  encode_script,
  script_duration_s,
)
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.sunnypilot.widgets.list_view import button_item_sp, simple_button_item_sp, toggle_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller_tici import Scroller


FRAME_BENCHMARK_CMD = ["python3", "-m", "openpilot.selfdrive.modeld.frame_downscale", "--device", "QCOM"]
LID_SCAN_CMD = ["python3", "-m", "openpilot.sunnypilot.body_lid_scan"]


class PanelType(IntEnum):
  TWEAKS = 0
  PARK = 1
  LAUNCH = 2
  DYNAMIC_FOLLOW = 3
  SPEED_ASSIST = 4
  AUTO_LOCK = 5


class _BenchmarkDialog(ConfirmDialog):
  # a modal hides the layout below it, so the dialog polls for the benchmark result itself
  def __init__(self, text: str, take_result: Callable[[], str | None]):
    super().__init__(text, tr("OK"), cancel_text="")
    self._take_result = take_result

  def _render(self, rect):
    if (result := self._take_result()) is not None:
      self.set_text(result)
    super()._render(rect)


class TweaksLayout(Widget):
  def __init__(self):
    super().__init__()

    self._current_panel = PanelType.TWEAKS
    self._hazard_flash_until = 0.0
    self._benchmark_thread: threading.Thread | None = None
    self._benchmark_result: str | None = None
    self._lid_scan_thread: threading.Thread | None = None
    self._lid_scan_result: str | None = None
    self._dynamic_follow_layout = DynamicFollowSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._launch_layout = LaunchAssistSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._park_layout = ParkAssistSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._speed_assist_layout = SpeedAssistSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._auto_lock_layout = AutoLockSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._remember_experimental_mode = toggle_item_sp(
      title=lambda: tr("Remember Experimental Mode Status"),
      description=lambda: tr("Keep Experimental Mode set the way you left it after rebooting. Cars without " +
                            "openpilot longitudinal control will still force Experimental Mode off."),
      param="RememberExperimentalModeStatus",
    )

    self._dynamic_follow = toggle_item_sp(
      title=lambda: tr("Dynamic Follow Distance"),
      description=lambda: tr("Vary the follow distance with vehicle speed instead of using the fixed driving " +
                            "personality gap. Requires openpilot longitudinal control."),
      param="DynamicFollow",
    )
    self._dynamic_follow_button = simple_button_item_sp(
      button_text=lambda: tr("Manage Dynamic Follow Settings"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.DYNAMIC_FOLLOW),
    )

    self._launch_assist = toggle_item_sp(
      title=lambda: tr("Lead Launch Assist"),
      description=lambda: tr("When stopped behind a lead that pulls away, launch sooner instead of waiting for " +
                            "the model. The radar-based planner still enforces the safe gap, and it only acts " +
                            "from a full stop and never overrides the brake or gas. Requires openpilot " +
                            "longitudinal control."),
      param="LaunchAssist",
    )
    self._launch_assist_button = simple_button_item_sp(
      button_text=lambda: tr("Manage Launch Assist Settings"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.LAUNCH),
    )

    self._park_assist = toggle_item_sp(
      title=lambda: tr("Lead Halt Assist"),
      description=lambda: tr("When stopped behind a stopped lead, settle at a closer gap than the default. The " +
                            "gap smoothly returns to normal once the lead moves. Only acts near a standstill. " +
                            "Requires openpilot longitudinal control."),
      param="ParkAssist",
    )
    self._park_assist_button = simple_button_item_sp(
      button_text=lambda: tr("Manage Halt Assist Settings"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.PARK),
    )

    self._speed_assist_button = simple_button_item_sp(
      button_text=lambda: tr("Experimental Speed Assist"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.SPEED_ASSIST),
    )

    self._reverse_cruise = toggle_item_sp(
      title=lambda: tr("Reverse Cruise Increase"),
      description=lambda: tr("Reverse the cruise control button behavior so a short press increases the set speed " +
                            "by 5 instead of 1. Lexus/Toyota only. Requires openpilot longitudinal control."),
      param="ToyotaReverseCruise",
      callback=self._on_reverse_cruise,
      enabled=lambda: not ui_state.engaged,
    )

    self._hazard_test = button_item_sp(
      title=lambda: tr("Hazard Flash Test"),
      button_text=lambda: tr("Stop") if self._hazard_flashing() else tr("Flash"),
      description=lambda: tr("Blink the hazard lamps for a minute over the OBD diagnostic path, to check that the " +
                            "car takes commands from the panda before relying on a feature that sends them. Press " +
                            "again to stop early. Offroad only: the frames can only go out while the car is off. " +
                            "Toyota/Lexus."),
      callback=self._on_hazard_test,
      enabled=lambda: ui_state.is_offroad(),
    )

    self._lid_scan = button_item_sp(
      title=lambda: tr("Body ECU LID Scan"),
      button_text=lambda: tr("Scan"),
      description=lambda: tr("Probe every KWP 0x30 local identifier on the body ECU (0x750, 0x40) with an all-zero control " +
                            "record, the way the auto-lock commands are sent, and report which ones the ECU accepts and " +
                            "which touch the turn lamps. Looks for a hazard command that is not speed gated. Run it parked " +
                            "with the trunk clear and watch the car: an unknown identifier may actuate something. Offroad " +
                            "only, about a minute; the full report is written to /data/body_lid_scan.txt. Toyota/Lexus."),
      callback=self._on_lid_scan,
      enabled=lambda: ui_state.is_offroad(),
    )

    self._auto_lock_button = simple_button_item_sp(
      button_text=lambda: tr("Auto Door Lock"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.AUTO_LOCK),
    )

    self._chestnut_native_frames = toggle_item_sp(
      title=lambda: tr("Chestnut Native Camera Frames"),
      description=lambda: tr("Send full 1928x1208 camera frames to chestnut instead of resampling them to the comma four " +
                            "size (1344x760) on the device first. Native frames spend about 12 ms more of each 50 ms frame " +
                            "on the USB link, which makes heavier models lag. comma 3X only, applies on the next drive."),
      param="ChestnutNativeFrames",
    )

    self._frame_benchmark = button_item_sp(
      title=lambda: tr("Chestnut Frame Benchmark"),
      button_text=lambda: tr("Run"),
      description=lambda: tr("Time the comma four frame resample on this device's GPU and show the numbers on screen, " +
                            "along with the frame mode modeld used on the last drive. Offroad only, takes about a minute."),
      callback=self._on_frame_benchmark,
      enabled=lambda: ui_state.is_offroad(),
    )

    return [
      self._remember_experimental_mode,
      self._dynamic_follow,
      self._dynamic_follow_button,
      self._launch_assist,
      self._launch_assist_button,
      self._park_assist,
      self._park_assist_button,
      self._speed_assist_button,
      self._reverse_cruise,
      self._auto_lock_button,
      self._hazard_test,
      self._lid_scan,
      self._chestnut_native_frames,
      self._frame_benchmark,
    ]

  def _hazard_flashing(self) -> bool:
    # There is no feedback from pandad, so track the run against the length of the script that was
    # queued. Going onroad drops it, so that counts as the run being over.
    return ui_state.is_offroad() and time.monotonic() < self._hazard_flash_until

  def _on_frame_benchmark(self):
    if self._benchmark_thread is not None and self._benchmark_thread.is_alive():
      return
    self._benchmark_result = None
    gui_app.push_widget(_BenchmarkDialog(tr("Running the chestnut frame benchmark, this takes about a minute..."), self._take_benchmark_result))
    self._benchmark_thread = threading.Thread(target=self._run_frame_benchmark, daemon=True)
    self._benchmark_thread.start()

  def _take_benchmark_result(self) -> str | None:
    result, self._benchmark_result = self._benchmark_result, None
    return result

  @staticmethod
  def _chestnut_bundle_status() -> str:
    # which chestnut bundle is selected and whether its pkl is actually on disk and hash-valid
    try:
      from openpilot.sunnypilot.models.helpers import get_selected_bundle, _bundle_is_valid_locally
      bundle = get_selected_bundle(ui_state.params, "chestnut")
      if bundle is None:
        return "chestnut bundle: none selected"
      return f"chestnut bundle: {bundle.displayName}, files {'valid' if _bundle_is_valid_locally(bundle) else 'MISSING or hash mismatch'}"
    except Exception as e:
      return f"chestnut bundle: could not check ({e})"

  @staticmethod
  def _run_tool(cmd: list[str], timeout: float) -> str:
    """Run a command line tool and return the tail of what it printed, or why it failed."""
    try:
      proc = subprocess.run(cmd, cwd=BASEDIR, capture_output=True, text=True, timeout=timeout)
      output = proc.stdout.strip()[-800:]
      if proc.returncode != 0 or not output:
        output = f"exit code {proc.returncode}\n{output}\n{proc.stderr.strip()[-600:]}"
    except Exception as e:
      output = f"failed to run: {e}"
    return output

  def _run_frame_benchmark(self):
    mode = ui_state.params.get("ChestnutFrameMode") or "not run yet"
    mode += "\n" + self._chestnut_bundle_status()
    if error := ui_state.params.get("ChestnutLastError"):
      mode += f"\nchestnut error: {error[-700:]}"
    output = self._run_tool(FRAME_BENCHMARK_CMD, timeout=600)
    # picked up by the dialog's render, so its text is only touched from the UI thread
    self._benchmark_result = f"last drive: {mode}\n\n{output}"

  def _on_lid_scan(self):
    # The scan queues an OffroadCanScript, which pandad only plays offroad; the button is greyed
    # out onroad too, this guards the callback itself.
    if not ui_state.is_offroad():
      return
    if self._lid_scan_thread is not None and self._lid_scan_thread.is_alive():
      return
    # A scan and the hazard test share pandad's script slot; whichever is queued last plays.
    self._hazard_flash_until = 0.0
    self._lid_scan_result = None
    gui_app.push_widget(_BenchmarkDialog(tr("Scanning the body ECU, this takes about a minute. Watch the car..."), self._take_lid_scan_result))
    self._lid_scan_thread = threading.Thread(target=self._run_lid_scan, daemon=True)
    self._lid_scan_thread.start()

  def _take_lid_scan_result(self) -> str | None:
    result, self._lid_scan_result = self._lid_scan_result, None
    return result

  def _run_lid_scan(self):
    # 256 probes at 200 ms plus the settle time; the timeout only guards a hung recorder.
    self._lid_scan_result = self._run_tool(LID_SCAN_CMD, timeout=180)

  def _on_hazard_test(self):
    # pandad plays the script offroad only, so don't leave one queued for the next time the car
    # is parked. The button is greyed out onroad too; this guards the callback itself.
    if not ui_state.is_offroad():
      return

    # A minute is long enough to want out of, and a queued script replaces the one playing, so the
    # same button cuts the run short by queuing the frames that put the lamps out.
    if self._hazard_flashing():
      ui_state.params.put("OffroadCanScript", build_hazard_stop_script())
      self._hazard_flash_until = 0.0
      return

    frames = build_hazard_flash_frames()
    ui_state.params.put("OffroadCanScript", encode_script(frames))
    self._hazard_flash_until = time.monotonic() + script_duration_s(frames)

  def _on_reverse_cruise(self, state: bool):
    # The flag is read at car-process init, so request an onroad cycle to apply it without a full reboot.
    ui_state.params.put_bool("ToyotaReverseCruise", state)
    ui_state.params.put_bool("OnroadCycleRequested", True)

  def _render(self, rect):
    if self._current_panel == PanelType.DYNAMIC_FOLLOW:
      self._dynamic_follow_layout.render(rect)
    elif self._current_panel == PanelType.LAUNCH:
      self._launch_layout.render(rect)
    elif self._current_panel == PanelType.PARK:
      self._park_layout.render(rect)
    elif self._current_panel == PanelType.SPEED_ASSIST:
      self._speed_assist_layout.render(rect)
    elif self._current_panel == PanelType.AUTO_LOCK:
      self._auto_lock_layout.render(rect)
    else:
      self._scroller.render(rect)

  def show_event(self):
    self._set_current_panel(PanelType.TWEAKS)
    self._scroller.show_event()

  def _set_current_panel(self, panel: PanelType):
    self._current_panel = panel
    if panel == PanelType.DYNAMIC_FOLLOW:
      self._dynamic_follow_layout.show_event()
    elif panel == PanelType.LAUNCH:
      self._launch_layout.show_event()
    elif panel == PanelType.PARK:
      self._park_layout.show_event()
    elif panel == PanelType.SPEED_ASSIST:
      self._speed_assist_layout.show_event()
    elif panel == PanelType.AUTO_LOCK:
      self._auto_lock_layout.show_event()

  def _update_state(self):
    super()._update_state()
    self._dynamic_follow_button.action_item.set_enabled(self._dynamic_follow.action_item.get_state())
    self._launch_assist_button.action_item.set_enabled(self._launch_assist.action_item.get_state())
    self._park_assist_button.action_item.set_enabled(self._park_assist.action_item.get_state())
