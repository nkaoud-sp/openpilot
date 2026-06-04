"""
Settings submenu for the experimental Mapbox-based navigation (nkaoud_nav).
"""
from collections.abc import Callable

import pyray as rl
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller


class NavSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()

    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._show_polyline = toggle_item_sp(
      title=lambda: tr("Show Route Polyline"),
      description=lambda: tr("Overlay the active route onto the driving view as a polyline."),
      param="NkaoudNavShowPolyline",
    )
    self._show_banner = toggle_item_sp(
      title=lambda: tr("Show Maneuver Banner"),
      description=lambda: tr("Display the upcoming maneuver (turn direction, street name, distance) on the driving " +
                            "view."),
      param="NkaoudNavShowBanner",
    )
    self._control_speed = toggle_item_sp(
      title=lambda: tr("Slow For Upcoming Turns"),
      description=lambda: tr("Allow navigation to slow the car when approaching a turn on the route. Requires " +
                            "openpilot longitudinal control. Experimental."),
      param="NkaoudNavControlSpeed",
    )

    items = [
      self._show_polyline,
      self._show_banner,
      self._control_speed,
    ]
    return items

  def _render(self, rect):
    self._back_button.set_position(self._rect.x, self._rect.y + 20)
    self._back_button.render()

    content_rect = rl.Rectangle(rect.x, rect.y + self._back_button.rect.height + 40,
                                rect.width, rect.height - self._back_button.rect.height - 40)
    self._scroller.render(content_rect)

  def show_event(self):
    self._scroller.show_event()

  def hide_event(self):
    self._scroller.hide_event()
