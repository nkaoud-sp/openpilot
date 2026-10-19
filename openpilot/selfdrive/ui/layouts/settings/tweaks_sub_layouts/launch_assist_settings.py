from openpilot.selfdrive.ui.layouts.settings.tweaks_sub_layouts.base import TweaksSubLayout
from openpilot.system.ui.widgets.list_view import stepper_item
from openpilot.system.ui.lib.multilang import tr


class LaunchAssistSettingsLayout(TweaksSubLayout):
  def _build_items(self):
    return [
      stepper_item(
        lambda: tr("Launch Eagerness"),
        initial_value=self._params.get("LaunchEagerness", return_default=True),
        min_value=1,
        max_value=10,
        step=1,
        label_callback=lambda value: tr("Level {}").format(value),
        callback=lambda value: self._params.put("LaunchEagerness", value),
        description=lambda: tr("How quickly to launch once a stopped lead pulls away. Higher launches with less " +
                               "lead movement; lower waits until the lead is clearly moving. The car still only " +
                               "launches when the gap is actually opening, and never overrides the brake or gas."),
      ),
    ]
