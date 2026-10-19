from openpilot.selfdrive.ui.layouts.settings.tweaks_sub_layouts.base import TweaksSubLayout
from openpilot.system.ui.widgets.list_view import stepper_item
from openpilot.system.ui.lib.multilang import tr


class ParkAssistSettingsLayout(TweaksSubLayout):
  def _build_items(self):
    return [
      stepper_item(
        lambda: tr("Standstill Gap"),
        initial_value=self._params.get("ParkDistance", return_default=True),
        min_value=100,
        max_value=300,
        step=10,
        label_callback=lambda value: f"{value / 100:.2f} m",
        callback=lambda value: self._params.put("ParkDistance", value),
        description=lambda: tr("How close to stop behind a stopped lead. Only applies near a standstill and " +
                               "smoothly restores the normal gap once the lead starts moving."),
      ),
    ]
