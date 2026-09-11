from __future__ import annotations

from openpilot.system.ui.lib.multilang import tr, tr_noop

from openpilot.selfdrive.ui.layouts.settings.starpilot.panel import _SettingsPage
from openpilot.selfdrive.ui.layouts.settings.starpilot.aethergrid import (
  DEFAULT_PANEL_STYLE,
  ParentToggle,
  SettingRow,
  SettingSection,
  AetherSettingsView,
  CardHubManagerView,
)


PANEL_STYLE = DEFAULT_PANEL_STYLE


class TweaksManagerView(CardHubManagerView):
  def __init__(self, controller, sections, **kwargs):
    super().__init__(controller, sections, **kwargs)

  def _build_cards(self):
    return [
      {
        "title": tr("Lead Launch Assist"),
        "desc": tr("When a stopped lead pulls away, launch sooner while preserving the MPC safe gap."),
        "icon": "vehicle",
        "on_click": lambda: self._controller._navigate_to("launch_assist"),
      },
      {
        "title": tr("Lead Halt Assist"),
        "desc": tr("Settle closer behind a stopped lead and smoothly restore the normal gap after launch."),
        "icon": "road",
        "on_click": lambda: self._controller._navigate_to("halt_assist"),
      },
    ]


class StarPilotTweaksLayout(_SettingsPage):
  def __init__(self):
    super().__init__()
    self._build_view()

  def _make_parent(self, key: str, label: str, subtitle: str = "") -> ParentToggle:
    return ParentToggle(
      label=label,
      subtitle=subtitle,
      get_state=lambda k=key: self._params.get_bool(k),
      set_state=lambda s, k=key: self._params.put_bool(k, s),
    )

  def _build_view(self):
    self._launch_assist_rows = [
      SettingRow("LaunchEagerness", "value", tr_noop("Launch Eagerness"),
                 subtitle=tr_noop("Higher launches with less lead movement; lower waits until the lead is clearly moving."),
                 get_value=lambda: tr("Level {}").format(self._params.get_int("LaunchEagerness", return_default=True, default=10)),
                 on_click=lambda: self._show_slider("LaunchEagerness", 1, 10, step=1,
                                                    title=tr_noop("Launch Eagerness"))),
    ]

    self._halt_assist_rows = [
      SettingRow("ParkDistance", "value", tr_noop("Standstill Gap"),
                 subtitle=tr_noop("How close to stop behind a stopped lead. The normal gap returns as speed rises."),
                 get_value=lambda: f"{self._params.get_int('ParkDistance') / 100:.2f} m",
                 on_click=lambda: self._show_slider("ParkDistance", 100, 300, step=10, unit=" cm",
                                                    title=tr_noop("Standstill Gap"))),
    ]

    self._manager_view = TweaksManagerView(
      self, [],
      header_title=tr_noop("Tweaks"),
      header_subtitle=tr_noop("Independent behavior tweaks that sit outside the main tuning groups."),
      panel_style=PANEL_STYLE,
    )

    pt_halt_assist = self._make_parent(
      "ParkAssist",
      "Lead Halt Assist",
      "When stopped behind a stopped lead, settle at a closer gap than the default.",
    )

    pt_launch_assist = self._make_parent(
      "LaunchAssist",
      "Lead Launch Assist",
      "When stopped behind a lead that pulls away, use the radar MPC output to launch sooner.",
    )

    self._sub_panels["launch_assist"] = AetherSettingsView(
      self,
      [SettingSection(title="", rows=self._launch_assist_rows)],
      header_title=tr_noop("Lead Launch Assist"),
      header_subtitle=tr_noop("Tune how eagerly StarPilot reacts when the stopped lead starts moving."),
      parent_toggle=pt_launch_assist,
      panel_style=PANEL_STYLE,
    )

    self._sub_panels["halt_assist"] = AetherSettingsView(
      self,
      [SettingSection(title="", rows=self._halt_assist_rows)],
      header_title=tr_noop("Lead Halt Assist"),
      header_subtitle=tr_noop("Tune the standstill gap and when the closer-gap behavior engages."),
      parent_toggle=pt_halt_assist,
      panel_style=PANEL_STYLE,
    )
    self._wire_sub_panels()
