"""
Settings submenu for the experimental Mapbox-based navigation (nkaoud_nav).
"""
from collections.abc import Callable

import pyray as rl
from openpilot.common.params import Params
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.nav_token_qr_dialog import NavTokenQrDialog
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, simple_button_item_sp, multiple_button_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller


class NavSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()

    self._params = Params()
    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._token_button = simple_button_item_sp(
      button_text=lambda: self._token_button_label(),
      button_width=800,
      callback=lambda: self._open_token_input(),
    )
    self._clear_destination_button = simple_button_item_sp(
      button_text=lambda: tr("Clear Current Destination"),
      button_width=800,
      callback=lambda: self._params.remove("NkaoudNavDestination"),
    )
    self._show_polyline = toggle_item_sp(
      title=lambda: tr("Show Route Polyline"),
      description=lambda: tr("Overlay the active route onto the driving view as a polyline."),
      param="NkaoudNavShowPolyline",
    )
    self._polyline_style = multiple_button_item_sp(
      title=lambda: tr("Polyline Style"),
      description=lambda: tr("Solid: sharp blue stroke (literal Mapbox geometry). " +
                            "Smooth: Catmull-Rom interpolated curve with width taper. " +
                            "Glow: smooth + neon halo. " +
                            "Chevrons: animated forward-flow chevrons."),
      buttons=[lambda: tr("Solid"), lambda: tr("Smooth"),
               lambda: tr("Glow"), lambda: tr("Chevrons")],
      param="NkaoudNavPolylineStyle",
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
      self._token_button,
      self._clear_destination_button,
      self._show_polyline,
      self._polyline_style,
      self._show_banner,
      self._control_speed,
    ]
    return items

  def _token_button_label(self) -> str:
    token = (self._params.get("NkaoudNavMapboxToken") or "").strip()
    if token:
      # show only a short masked indicator so the token isn't displayed in plain text
      tail = token[-4:] if len(token) >= 4 else token
      return tr("Mapbox Token (set, ...{})").format(tail)
    return tr("Set Mapbox Token")

  def _open_token_input(self) -> None:
    # Pushes the QR dialog. The dialog starts a temporary HTTP server on
    # :8081 in its __init__ and stops it on cancel / token receipt.
    gui_app.push_widget(NavTokenQrDialog())

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
