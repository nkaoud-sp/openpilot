from __future__ import annotations

from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.option_dialog import MultiOptionDialog

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
PARK_ASSIST_MODE_OPTIONS = [
  (0, "From Full Stop"),
  (1, "Any Low Speed"),
]


class TweaksManagerView(CardHubManagerView):
  def __init__(self, controller, sections, **kwargs):
    super().__init__(controller, sections, **kwargs)

  def _build_cards(self):
    return [
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
    self._halt_assist_rows = [
      SettingRow("ParkDistance", "value", tr_noop("Standstill Gap"),
                 subtitle=tr_noop("How close to stop behind a stopped lead. The normal gap returns as speed rises."),
                 get_value=lambda: f"{self._params.get_int('ParkDistance') / 100:.2f} m",
                 on_click=lambda: self._show_slider("ParkDistance", 100, 300, step=10, unit=" cm",
                                                    title=tr_noop("Standstill Gap"))),
      SettingRow("ParkAssistMode", "value", tr_noop("Engage When"),
                 subtitle=tr_noop("From Full Stop waits for a stopped lead; Any Low Speed applies while following slowly."),
                 get_value=self._get_park_mode_display,
                 on_click=self._show_park_mode_selector),
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

    self._sub_panels["halt_assist"] = AetherSettingsView(
      self,
      [SettingSection(title="", rows=self._halt_assist_rows)],
      header_title=tr_noop("Lead Halt Assist"),
      header_subtitle=tr_noop("Tune the standstill gap and when the closer-gap behavior engages."),
      parent_toggle=pt_halt_assist,
      panel_style=PANEL_STYLE,
    )
    self._wire_sub_panels()

  def _get_park_mode_display(self):
    current = self._params.get_int("ParkAssistMode", return_default=True, default=1)
    return tr(next((label for value, label in PARK_ASSIST_MODE_OPTIONS if value == current), "Any Low Speed"))

  def _show_park_mode_selector(self):
    option_labels = [tr(label) for _, label in PARK_ASSIST_MODE_OPTIONS]
    current = self._get_park_mode_display()

    def on_select(res):
      if res == DialogResult.CONFIRM and dialog.selection in option_labels:
        selected_index = option_labels.index(dialog.selection)
        self._params.put_int("ParkAssistMode", PARK_ASSIST_MODE_OPTIONS[selected_index][0])

    dialog = MultiOptionDialog(tr("Engage When"), option_labels, current, callback=on_select)
    gui_app.push_widget(dialog)
