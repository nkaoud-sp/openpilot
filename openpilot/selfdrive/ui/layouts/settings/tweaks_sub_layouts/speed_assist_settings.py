from openpilot.selfdrive.ui.layouts.settings.tweaks_sub_layouts.base import TweaksSubLayout
from openpilot.system.ui.widgets.list_view import multiple_button_item, stepper_item
from openpilot.system.ui.lib.multilang import tr


class SpeedAssistSettingsLayout(TweaksSubLayout):
  def _build_items(self):
    def put(param):
      return lambda index: self._params.put(param, index)

    return [
      multiple_button_item(
        lambda: tr("Mode"),
        lambda: tr("Readout shows when speed assist would apply without changing acceleration. On adds a small, " +
                   "smoothed acceleration nudge only in clean experimental open-road cases."),
        buttons=[lambda: tr("Off"), lambda: tr("Readout"), lambda: tr("On")],
        selected_index=self._params.get("ExperimentalSpeedAssistMode", return_default=True),
        button_width=240,
        callback=put("ExperimentalSpeedAssistMode"),
      ),
      multiple_button_item(
        lambda: tr("Strength"),
        lambda: tr("Maximum added acceleration once far below the set speed: Low +0.15, Medium +0.25, " +
                   "High +0.35 m/s2. The nudge ramps down as you approach the target."),
        buttons=[lambda: tr("Low"), lambda: tr("Medium"), lambda: tr("High")],
        selected_index=self._params.get("ExperimentalSpeedAssistStrength", return_default=True),
        button_width=250,
        callback=put("ExperimentalSpeedAssistStrength"),
      ),
      stepper_item(
        lambda: tr("Minimum Speed"),
        initial_value=self._params.get("ExperimentalSpeedAssistMinKph", return_default=True),
        min_value=30,
        max_value=100,
        step=5,
        label_callback=lambda value: f"{value} kph",
        callback=lambda value: self._params.put("ExperimentalSpeedAssistMinKph", value),
        description=lambda: tr("Speed assist is blocked below this speed."),
      ),
      stepper_item(
        lambda: tr("Maximum Speed"),
        initial_value=self._params.get("ExperimentalSpeedAssistMaxKph", return_default=True),
        min_value=90,
        max_value=150,
        step=5,
        label_callback=lambda value: f"{value} kph",
        callback=lambda value: self._params.put("ExperimentalSpeedAssistMaxKph", value),
        description=lambda: tr("Speed assist is blocked above this speed."),
      ),
      stepper_item(
        lambda: tr("Assist Starts Below Target"),
        initial_value=self._params.get("ExperimentalSpeedAssistStartGapKph", return_default=True),
        min_value=5,
        max_value=30,
        step=1,
        label_callback=lambda value: f"{value} kph",
        callback=lambda value: self._params.put("ExperimentalSpeedAssistStartGapKph", value),
        description=lambda: tr("How far below the cruise set speed the nudge can begin. The boost ramps smoothly " +
                               "from a tiny assist here to full strength around 30 kph below target."),
      ),
      multiple_button_item(
        lambda: tr("Lead Blocking"),
        lambda: tr("Block keeps speed assist disabled whenever radar or the model reports a lead. Ignore lets " +
                   "speed assist ignore lead detection for testing; normal longitudinal planning still runs, " +
                   "but this is less conservative."),
        buttons=[lambda: tr("Block"), lambda: tr("Ignore")],
        selected_index=self._params.get("ExperimentalSpeedAssistLeadMode", return_default=True),
        button_width=260,
        callback=put("ExperimentalSpeedAssistLeadMode"),
      ),
    ]
