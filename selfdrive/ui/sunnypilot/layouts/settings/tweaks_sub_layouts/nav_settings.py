"""
Settings submenu for the experimental Mapbox-based navigation (nkaoud_nav).
"""
from collections.abc import Callable

import pyray as rl
from openpilot.common.params import Params
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.nav_token_qr_dialog import (
  NavTokenQrDialog, NavShareEndpointQrDialog,
)
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
    self._share_endpoint_button = simple_button_item_sp(
      button_text=lambda: self._share_endpoint_label(),
      button_width=800,
      callback=lambda: self._open_share_endpoint_input(),
    )
    self._clear_destination_button = simple_button_item_sp(
      button_text=lambda: tr("Clear Current Destination"),
      button_width=800,
      callback=lambda: self._clear_destination(),
    )
    self._show_polyline = toggle_item_sp(
      title=lambda: tr("Show Route Polyline"),
      description=lambda: tr("Overlay the active route onto the driving view as a polyline."),
      param="NkaoudNavShowPolyline",
    )
    self._polyline_style = multiple_button_item_sp(
      title=lambda: tr("Polyline Style"),
      description=lambda: tr("Solid: literal Mapbox geometry. Smooth: interpolated, " +
                            "tapered. Glow: neon halo. Chevrons: animated arrows. " +
                            "Ribbon: filled lane swath. Dashed: dotted waypoint look. " +
                            "Smoke: diffuse trail. Composite: smooth + chevrons."),
      buttons=[lambda: tr("Solid"), lambda: tr("Smooth"),
               lambda: tr("Glow"), lambda: tr("Chevrons"),
               lambda: tr("Ribbon"), lambda: tr("Dashed"),
               lambda: tr("Smoke"), lambda: tr("Composite")],
      param="NkaoudNavPolylineStyle",
      button_width=220,
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
    self._control_steer = toggle_item_sp(
      title=lambda: tr("Steer / Lane-Keep With The Route"),
      description=lambda: tr("Lets the route bias steering near turns (turnLeft/turnRight) and nudge toward the " +
                            "correct lane when an upcoming maneuver requires it (keepLeft/keepRight). " +
                            "Nav never triggers an assertive lane change — the driver blinker still does that. " +
                            "Experimental."),
      param="NkaoudNavControlSteer",
    )
    self._lane_change_cooldown = multiple_button_item_sp(
      title=lambda: tr("Nav Lane Change Cooldown"),
      description=lambda: tr("Minimum cooldown after nav detects a lane index change before it can request the next " +
                            "keepLeft/keepRight step. Useful when multiple route lane changes are needed in sequence. " +
                            "Off allows immediate follow-up requests."),
      buttons=[lambda: tr("Off"), lambda: f"1 {tr('s')}",
               lambda: f"2 {tr('s')}", lambda: f"3 {tr('s')}",
               lambda: f"5 {tr('s')}"],
      param="NkaoudNavLaneChangeCooldown",
      button_width=220,
    )
    self._highway_default = toggle_item_sp(
      title=lambda: tr("Highway Lane Default"),
      description=lambda: tr("When cruising on a motorway with no imminent maneuver, nudge toward the preferred " +
                            "lane using a conservative lane keep. Suppressed if you manually blinker on the highway; " +
                            "automatically re-enabled when the next navigation command fires."),
      param="NkaoudNavHighwayDefault",
    )
    self._highway_lane_pref = multiple_button_item_sp(
      title=lambda: tr("Highway Lane Preference"),
      description=lambda: tr("Which lane to target on motorways when no maneuver is imminent. " +
                            "Rightmost = slow/right lane. Center = middle lane. Leftmost = passing lane. " +
                            "Only active when Highway Lane Default is enabled."),
      buttons=[lambda: tr("Rightmost"), lambda: tr("Center"), lambda: tr("Leftmost")],
      param="NkaoudNavHighwayLanePref",
      button_width=330,
    )
    self._turn_tolerance = multiple_button_item_sp(
      title=lambda: tr("Turn Speed Tolerance"),
      description=lambda: tr("How aggressively to slow for upcoming turns. Speed is computed from road curvature " +
                            "(v = sqrt(a_lat × tolerance / curvature)). Conservative slows more; Aggressive slows " +
                            "less. Default is Normal (100 %)."),
      buttons=[lambda: tr("Conservative"), lambda: tr("Normal"), lambda: tr("Aggressive")],
      param="NkaoudNavTurnTolerance",
      button_width=330,
    )
    self._max_lat_accel = multiple_button_item_sp(
      title=lambda: tr("Turn Max Lateral Accel"),
      description=lambda: tr("Maximum lateral acceleration used to compute the geometry-based turn speed cap. " +
                            "Higher allows faster corner entry; lower forces slower corners. " +
                            "Default is 2.5 m/s²."),
      buttons=[lambda: tr("2.0 m/s²"), lambda: tr("2.5 m/s²"), lambda: tr("3.0 m/s²")],
      param="NkaoudNavMaxLatAccel",
      button_width=330,
    )

    items = [
      self._token_button,
      self._share_endpoint_button,
      self._clear_destination_button,
      self._show_polyline,
      self._polyline_style,
      self._show_banner,
      self._control_speed,
      self._turn_tolerance,
      self._max_lat_accel,
      self._control_steer,
      self._lane_change_cooldown,
      self._highway_default,
      self._highway_lane_pref,
    ]
    return items

  def _token_button_label(self) -> str:
    token = (self._params.get("NkaoudNavMapboxToken") or "").strip()
    if token:
      tail = token[-4:] if len(token) >= 4 else token
      return tr("Mapbox Token (set, ...{})").format(tail)
    return tr("Set Mapbox Token")

  def _open_token_input(self) -> None:
    gui_app.push_widget(NavTokenQrDialog())

  def _share_endpoint_label(self) -> str:
    url = (self._params.get("NkaoudNavShareEndpoint") or "").strip()
    if not url:
      return tr("Set Neon Connection String (for Share)")
    after_scheme = url.split("://", 1)[-1]
    after_creds = after_scheme.split("@", 1)[-1]
    host = after_creds.split("/", 1)[0]
    return tr("Neon Connection (set, {})").format(host)

  def _open_share_endpoint_input(self) -> None:
    gui_app.push_widget(NavShareEndpointQrDialog())

  def _clear_destination(self) -> None:
    self._params.remove("NkaoudNavDestination")
    self._params.remove("NkaoudNavShareTrigger")

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
