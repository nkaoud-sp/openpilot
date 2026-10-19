from openpilot.selfdrive.ui.layouts.settings.tweaks_sub_layouts.base import TweaksSubLayout
from openpilot.system.ui.widgets.list_view import stepper_item
from openpilot.system.ui.lib.multilang import tr


class DynamicFollowSettingsLayout(TweaksSubLayout):
  def _build_items(self):
    return [
      stepper_item(
        lambda: tr("Follow Time At 0 km/h"),
        initial_value=self._params.get("DynamicFollowMinTime", return_default=True),
        min_value=30,
        max_value=150,
        step=5,
        label_callback=lambda value: f"{value / 100:.2f} s",
        callback=lambda value: self._params.put("DynamicFollowMinTime", value),
        description=lambda: tr("Follow time used at a standstill / low speed. Shorter means a closer gap."),
      ),
      stepper_item(
        lambda: tr("Follow Time At 130 km/h"),
        initial_value=self._params.get("DynamicFollowMaxTime", return_default=True),
        min_value=50,
        max_value=250,
        step=5,
        label_callback=lambda value: f"{value / 100:.2f} s",
        callback=lambda value: self._params.put("DynamicFollowMaxTime", value),
        description=lambda: tr("Follow time used at highway speed. Longer means a larger gap."),
      ),
      stepper_item(
        lambda: tr("Curve Shape"),
        initial_value=self._params.get("DynamicFollowCurve", return_default=True),
        min_value=50,
        max_value=300,
        step=5,
        label_callback=lambda value: f"{value / 100:.2f}",
        callback=lambda value: self._params.put("DynamicFollowCurve", value),
        description=lambda: tr("Bend the follow-time curve between the two endpoints. 1.00 is a straight line. " +
                               "Below 1.00 opens the gap earlier; above 1.00 stays tighter until highway speed."),
      ),
    ]
