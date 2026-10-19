from openpilot.common.params import Params
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.list_view import toggle_item, stepper_item
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.system.ui.lib.multilang import tr, tr_noop

# Description constants
DESCRIPTIONS = {
  "DynamicFollow": tr_noop(
    "Scale the follow distance with speed instead of using a fixed following distance. " +
    "Keeps a closer gap at low speed and opens it up on the highway."
  ),
  "DynamicFollowMinTime": tr_noop("Follow time used at a standstill / low speed. Shorter means a closer gap."),
  "DynamicFollowMaxTime": tr_noop("Follow time used at highway speed. Longer means a larger gap."),
  "DynamicFollowCurve": tr_noop(
    "Bend the follow-time curve between the two endpoints. 1.00 is a straight line. " +
    "Below 1.00 opens the gap earlier; above 1.00 stays tighter until highway speed."
  ),
}


class TweaksLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = Params()
    self._dynamic_follow = self._params.get_bool("DynamicFollow")

    self._dynamic_follow_toggle = toggle_item(
      lambda: tr("Dynamic Follow"),
      description=lambda: tr(DESCRIPTIONS["DynamicFollow"]),
      initial_state=self._dynamic_follow,
      callback=self._on_dynamic_follow,
      icon="speed_limit.png",
    )

    self._dynamic_follow_min = stepper_item(
      lambda: tr("Follow Time At 0 km/h"),
      initial_value=self._params.get("DynamicFollowMinTime", return_default=True),
      min_value=30,
      max_value=150,
      step=5,
      label_callback=lambda value: f"{value / 100:.2f} s",
      callback=lambda value: self._params.put("DynamicFollowMinTime", value),
      description=lambda: tr(DESCRIPTIONS["DynamicFollowMinTime"]),
      enabled=lambda: self._dynamic_follow,
    )
    self._dynamic_follow_max = stepper_item(
      lambda: tr("Follow Time At 130 km/h"),
      initial_value=self._params.get("DynamicFollowMaxTime", return_default=True),
      min_value=50,
      max_value=250,
      step=5,
      label_callback=lambda value: f"{value / 100:.2f} s",
      callback=lambda value: self._params.put("DynamicFollowMaxTime", value),
      description=lambda: tr(DESCRIPTIONS["DynamicFollowMaxTime"]),
      enabled=lambda: self._dynamic_follow,
    )
    self._dynamic_follow_curve = stepper_item(
      lambda: tr("Curve Shape"),
      initial_value=self._params.get("DynamicFollowCurve", return_default=True),
      min_value=50,
      max_value=300,
      step=5,
      label_callback=lambda value: f"{value / 100:.2f}",
      callback=lambda value: self._params.put("DynamicFollowCurve", value),
      description=lambda: tr(DESCRIPTIONS["DynamicFollowCurve"]),
      enabled=lambda: self._dynamic_follow,
    )

    items = [
      self._dynamic_follow_toggle,
      self._dynamic_follow_min,
      self._dynamic_follow_max,
      self._dynamic_follow_curve,
    ]
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _on_dynamic_follow(self, state: bool):
    self._dynamic_follow = state
    self._params.put_bool("DynamicFollow", state)

  def show_event(self):
    super().show_event()
    self._scroller.show_event()
    self._dynamic_follow = self._params.get_bool("DynamicFollow")
    self._dynamic_follow_toggle.action_item.set_state(self._dynamic_follow)

  def _render(self, rect):
    self._scroller.render(rect)
