from __future__ import annotations

from openpilot.selfdrive.ui.layouts.settings.starpilot.aethergrid import (
  AetherSettingsView,
  DEFAULT_PANEL_STYLE,
  SettingRow,
  SettingSection,
)
from openpilot.selfdrive.ui.layouts.settings.starpilot.panel import _SettingsPage
from openpilot.starpilot.common.experimental_state import sync_persist_experimental_state
from openpilot.system.ui.lib.multilang import tr_noop


class StarPilotTweaksLayout(_SettingsPage):
  def __init__(self):
    super().__init__()

    self._rows = [
      SettingRow("PersistExperimentalState", "toggle", tr_noop("Experimental Persist"),
                 subtitle=tr_noop("Keep manual Experimental Mode status after rebooting."),
                 get_state=lambda: self._params.get_bool("PersistExperimentalState"),
                 set_state=self._set_persist_experimental_state),
      SettingRow("LaunchAssist", "toggle", tr_noop("Lead Launch Assist"),
                 subtitle=tr_noop("When stopped behind a lead that pulls away, launch sooner using the radar planner."),
                 get_state=lambda: self._params.get_bool("LaunchAssist"),
                 set_state=lambda s: self._params.put_bool("LaunchAssist", s)),
      SettingRow("LaunchEagerness", "value", tr_noop("Launch Eagerness"),
                 subtitle=tr_noop("Higher values react to smaller lead movement."),
                 get_value=lambda: f"Level {self._params.get_int('LaunchEagerness')}",
                 on_click=lambda: self._show_slider("LaunchEagerness", 1, 10, step=1, title="Launch Eagerness"),
                 visible=lambda: self._params.get_bool("LaunchAssist")),
      SettingRow("ParkAssist", "toggle", tr_noop("Lead Halt Assist"),
                 subtitle=tr_noop("At low speed behind a lead, settle at a closer standstill gap."),
                 get_state=lambda: self._params.get_bool("ParkAssist"),
                 set_state=lambda s: self._params.put_bool("ParkAssist", s)),
      SettingRow("ParkDistance", "value", tr_noop("Halt Gap"),
                 subtitle=tr_noop("How close to stop behind a lead at low speed."),
                 get_value=lambda: f"{self._params.get_int('ParkDistance') / 100:.2f} m",
                 on_click=lambda: self._show_slider("ParkDistance", 100, 300, step=10, unit=" cm", title="Halt Gap"),
                 visible=lambda: self._params.get_bool("ParkAssist")),
      SettingRow("AutoLockEnabled", "toggle", tr_noop("Auto Lock On Exit"),
                 subtitle=tr_noop("Secure the car after you leave. Toyota and Lexus only."),
                 get_state=lambda: self._params.get_bool("AutoLockEnabled"),
                 set_state=lambda s: self._params.put_bool("AutoLockEnabled", s)),
      SettingRow("LockDoorsTimer", "value", tr_noop("Lock Delay"),
                 subtitle=tr_noop("Seconds to wait after the driver leaves before locking."),
                 get_value=lambda: f"{self._params.get_int('LockDoorsTimer')} s",
                 on_click=lambda: self._show_slider("LockDoorsTimer", 2, 180, step=1, unit=" s", title="Lock Delay"),
                 visible=lambda: self._params.get_bool("AutoLockEnabled")),
      SettingRow("FoldMirrors", "toggle", tr_noop("Fold Mirrors"),
                 subtitle=tr_noop("Fold mirrors when auto lock secures the car."),
                 get_state=lambda: self._params.get_bool("FoldMirrors"),
                 set_state=lambda s: self._params.put_bool("FoldMirrors", s),
                 visible=lambda: self._params.get_bool("AutoLockEnabled")),
      SettingRow("CloseWindows", "toggle", tr_noop("Close Windows"),
                 subtitle=tr_noop("Close windows when auto lock secures the car."),
                 get_state=lambda: self._params.get_bool("CloseWindows"),
                 set_state=lambda s: self._params.put_bool("CloseWindows", s),
                 visible=lambda: self._params.get_bool("AutoLockEnabled")),
    ]

    self._manager_view = AetherSettingsView(
      self,
      [SettingSection(title="", rows=self._rows)],
      header_title=tr_noop("Tweaks"),
      header_subtitle=tr_noop("Small behavior changes for launches, low-speed halts, drive-mode persistence, and exit locking."),
      panel_style=DEFAULT_PANEL_STYLE,
    )

  def _set_persist_experimental_state(self, state: bool):
    sync_persist_experimental_state(self._params, self._params_memory, state)
