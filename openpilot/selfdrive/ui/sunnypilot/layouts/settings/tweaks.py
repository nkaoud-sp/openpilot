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


FRAME_BENCHMARK_CMD = ["python3", "-m", "openpilot.selfdrive.modeld.chestnut_frames", "--device", "QCOM"]
MODEL_PROBE_CMD = ["python3", "-m", "openpilot.sunnypilot.models.pkl_probe"]
MODEL_DRY_RUN_CMD = MODEL_PROBE_CMD + ["--dry-run"]
JOB_TIMEOUT = 600


def as_html(text: str) -> str:
  # the html renderer drops whitespace between tokens, so every line needs its own paragraph,
  # and stray angle brackets in a traceback would be eaten as tags
  safe = text.replace('<', '[').replace('>', ']')
  return "".join(f"<p>{line if line.strip() else ' '}</p>" for line in safe.splitlines())


class PanelType(IntEnum):
  TWEAKS = 0
  PARK = 1
  LAUNCH = 2
  DYNAMIC_FOLLOW = 3
  SPEED_ASSIST = 4
  AUTO_LOCK = 5
  LANE_POLICY = 6


class LanePolicySettingsLayout(Widget):
  def __init__(self, back_callback: Callable):
    super().__init__()
    self._back_callback = back_callback

    self._back_button = simple_button_item_sp(
      button_text=lambda: tr("Back"),
      button_width=800,
      callback=self._back_callback,
    )
    self._one_line_fallback = toggle_item_sp(
      title=lambda: tr("One-Line Fallback"),
      description=lambda: tr("If one lane line briefly disappears, hold lane centering using the remaining line and the learned lane width."),
      param="LanePolicyOneLineFallback",
      enabled=lambda: ui_state.params.get_bool("LanePolicyEnabled"),
    )
    self._lead_fallback = toggle_item_sp(
      title=lambda: tr("Lead Fallback"),
      description=lambda: tr("When lane-line fallback is unavailable, use a detected lead vehicle as a low-authority lateral reference."),
      param="LanePolicyLeadFallback",
      enabled=lambda: ui_state.params.get_bool("LanePolicyEnabled"),
    )
    self._visual_indicator = toggle_item_sp(
      title=lambda: tr("Colored Lane-Line Indicator"),
      description=lambda: tr("Highlight the lane line the correction steers toward: green two-line, yellow one-line, purple lead fallback."),
      param="LanePolicyVisualIndicator",
      enabled=lambda: ui_state.params.get_bool("LanePolicyEnabled"),
    )

    self._scroller = Scroller([
      self._back_button,
      self._one_line_fallback,
      self._lead_fallback,
      self._visual_indicator,
    ], line_separator=True, spacing=0)

  def _render(self, rect):
    self._scroller.render(rect)

  def show_event(self):
    self._scroller.show_event()


class _JobDialog(ConfirmDialog):
  # a modal hides the layout below it, so the dialog polls for the job's result itself
  def __init__(self, text: str, take_result: Callable[[], str | None], rich: bool = False):
    super().__init__(as_html(text) if rich else text, tr("OK"), cancel_text="", rich=rich)
    self._take_result = take_result
    self._rich_text = rich

  def _render(self, rect):
    if (result := self._take_result()) is not None:
      self.set_text(as_html(result) if self._rich_text else result)
    super()._render(rect)


class TweaksLayout(Widget):
  def __init__(self):
    super().__init__()

    self._current_panel = PanelType.TWEAKS
    self._hazard_flash_until = 0.0
    self._job_thread: threading.Thread | None = None
    self._job_result: str | None = None
    self._dynamic_follow_layout = DynamicFollowSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._launch_layout = LaunchAssistSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._park_layout = ParkAssistSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._speed_assist_layout = SpeedAssistSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._auto_lock_layout = AutoLockSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._lane_policy_layout = LanePolicySettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))

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

    self._lane_policy = toggle_item_sp(
      title=lambda: tr("Lane Centering Policy"),
      description=lambda: tr("When both lane lines are clean, add a bounded lane-centering correction to the model's curvature."),
      param="LanePolicyEnabled",
    )
    self._lane_policy_button = simple_button_item_sp(
      button_text=lambda: tr("Manage Lane Policy Settings"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.LANE_POLICY),
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

    self._chestnut_ray_matching = toggle_item_sp(
      title=lambda: tr("Chestnut C4 Ray Matching"),
      description=lambda: tr("Trace every comma four pixel through the measured 3X lenses instead of plain resampling, so " +
                            "the chestnut model sees true comma four camera geometry: the wide fisheye reprojected, and the " +
                            "narrow built from the 3X narrow inset over the wide surround. Costs a little more GPU time than " +
                            "resampling. Needs both cameras. comma 3X only, applies on the next drive."),
      param="ChestnutRayMatching",
      enabled=lambda: not ui_state.params.get_bool("ChestnutNativeFrames"),
    )

    self._frame_benchmark = button_item_sp(
      title=lambda: tr("Chestnut Frame Benchmark"),
      button_text=lambda: tr("Run"),
      description=lambda: tr("Time both frame modes on this device's GPU and show the numbers on screen, along with " +
                            "the mode modeld used on the last drive. Offroad only, takes a couple of minutes."),
      callback=self._on_frame_benchmark,
      enabled=lambda: ui_state.is_offroad(),
    )

    self._model_probe = button_item_sp(
      title=lambda: tr("Model Pkl Diagnostics"),
      button_text=lambda: tr("Run"),
      description=lambda: tr("Inspect the selected chestnut model file on disk and show what is wrong with it: which " +
                            "loader schema it was built for, how its weights are packed, and what the loader raises " +
                            "when it opens the file. Run this when a big model fails to load. Offroad only, takes a " +
                            "minute."),
      callback=self._on_model_probe,
      enabled=lambda: ui_state.is_offroad(),
    )

    self._model_dry_run = button_item_sp(
      title=lambda: tr("Big Model Dry Run"),
      button_text=lambda: tr("Run"),
      description=lambda: tr("Bring the selected chestnut model up on the eGPU right here and run one frame " +
                            "through it, the same way modeld does on a drive, and show the traceback if it " +
                            "fails. Saves finding out from the car. Offroad only, takes a few minutes."),
      callback=self._on_model_dry_run,
      enabled=lambda: ui_state.is_offroad(),
    )

    return [
      self._remember_experimental_mode,
      self._dynamic_follow,
      self._dynamic_follow_button,
      self._launch_assist,
      self._launch_assist_button,
      self._lane_policy,
      self._lane_policy_button,
      self._park_assist,
      self._park_assist_button,
      self._speed_assist_button,
      self._reverse_cruise,
      self._auto_lock_button,
      self._hazard_test,
      self._chestnut_native_frames,
      self._chestnut_ray_matching,
      self._frame_benchmark,
      self._model_probe,
      self._model_dry_run,
    ]

  def _hazard_flashing(self) -> bool:
    # There is no feedback from pandad, so track the run against the length of the script that was
    # queued. Going onroad drops it, so that counts as the run being over.
    return ui_state.is_offroad() and time.monotonic() < self._hazard_flash_until

  def _start_job(self, message: str, work: Callable[[], str], rich: bool = False):
    # one job at a time: there is a single result slot, and both jobs are heavy subprocesses
    if self._job_thread is not None and self._job_thread.is_alive():
      return
    self._job_result = None
    gui_app.push_widget(_JobDialog(message, self._take_job_result, rich=rich))
    self._job_thread = threading.Thread(target=self._run_job, args=(work,), daemon=True)
    self._job_thread.start()

  def _run_job(self, work: Callable[[], str]):
    try:
      result = work()
    except Exception as e:
      result = f"failed to run: {e}"
    # picked up by the dialog's render, so its text is only touched from the UI thread
    self._job_result = result

  def _take_job_result(self) -> str | None:
    result, self._job_result = self._job_result, None
    return result

  def _on_frame_benchmark(self):
    self._start_job(tr("Running the chestnut frame benchmark, this takes about a minute..."), self._run_frame_benchmark)

  def _on_model_probe(self):
    self._start_job(tr("Inspecting the selected chestnut model file, this takes a minute..."), self._run_model_probe, rich=True)

  def _on_model_dry_run(self):
    self._start_job(tr("Loading the big model onto chestnut and running a frame, this takes a few minutes..."),
                    self._run_model_dry_run, rich=True)

  @staticmethod
  def _run_model_probe() -> str:
    return TweaksLayout._run_probe(MODEL_PROBE_CMD)

  @staticmethod
  def _run_model_dry_run() -> str:
    return TweaksLayout._run_probe(MODEL_DRY_RUN_CMD)

  @staticmethod
  def _run_probe(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=BASEDIR, capture_output=True, text=True, timeout=JOB_TIMEOUT)
    output = proc.stdout.strip()
    if proc.returncode != 0 or not output:
      output = f"exit code {proc.returncode}\n{output}\n{proc.stderr.strip()[-800:]}"
    return output

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

  def _run_frame_benchmark(self) -> str:
    mode = ui_state.params.get("ChestnutFrameMode") or "not run yet"
    mode += "\n" + self._chestnut_bundle_status()
    if error := ui_state.params.get("ChestnutLastError"):
      mode += f"\nchestnut error: {error[-700:]}"
    proc = subprocess.run(FRAME_BENCHMARK_CMD, cwd=BASEDIR, capture_output=True, text=True, timeout=JOB_TIMEOUT)
    output = proc.stdout.strip()[-800:]
    if proc.returncode != 0 or not output:
      output = f"exit code {proc.returncode}\n{output}\n{proc.stderr.strip()[-600:]}"
    return f"last drive: {mode}\n\n{output}"

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
    elif self._current_panel == PanelType.LANE_POLICY:
      self._lane_policy_layout.render(rect)
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
    elif panel == PanelType.LANE_POLICY:
      self._lane_policy_layout.show_event()

  def _update_state(self):
    super()._update_state()
    self._dynamic_follow_button.action_item.set_enabled(self._dynamic_follow.action_item.get_state())
    self._launch_assist_button.action_item.set_enabled(self._launch_assist.action_item.get_state())
    self._lane_policy_button.action_item.set_enabled(self._lane_policy.action_item.get_state())
    self._park_assist_button.action_item.set_enabled(self._park_assist.action_item.get_state())
