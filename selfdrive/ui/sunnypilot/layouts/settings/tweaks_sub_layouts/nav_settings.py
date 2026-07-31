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
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, simple_button_item_sp, multiple_button_item_sp, option_item_sp
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
      title=lambda: tr("Steer / Lane-Change With The Route"),
      description=lambda: tr("Lets the route bias steering: turnLeft / turnRight near a turn, a cautious " +
                            "keepLeft / keepRight lane-change bias ahead of highway exits, forks and merges " +
                            "when you're not in the right lane, and toward your Highway Lane Preference while " +
                            "cruising. The bias requires an AutoLaneChange timer set in your steering settings " +
                            "(anything other than \"Off\") and only applies while the blind-spot monitor and " +
                            "camera see the target side clear. Experimental."),
      param="NkaoudNavControlSteer",
    )
    self._turn_assist = toggle_item_sp(
      title=lambda: tr("Assist Turns With Curvature Nudge"),
      description=lambda: tr("On top of the route's turn steering, add a small feedforward curvature nudge in the " +
                            "turn direction to help the driving model follow through, scaled by how sharp the turn " +
                            "is (gentler for slight turns, firmer for sharp ones; u-turns excluded). Requires " +
                            "\"Steer / Lane-Change With The Route\". The nudge is capped and clears the instant " +
                            "steering disengages. Experimental."),
      param="NkaoudNavTurnAssist",
    )
    self._visual_block_threshold = option_item_sp(
      title=lambda: tr("Camera Block Threshold"),
      description=lambda: tr("For route-requested lane changes, block the move when the visual vehicle detector's " +
                            "target-side car probability reaches this value. Lower is more conservative."),
      param="NkaoudNavVisualBlockThreshold",
      min_value=50,
      max_value=95,
      value_change_step=1,
      label_callback=lambda value: f"{value}%",
      use_float_scaling=True,
    )
    self._highway_lane_pref = multiple_button_item_sp(
      title=lambda: tr("Highway Lane Preference"),
      description=lambda: tr("Which lane to prefer while cruising a highway / main road with no upcoming " +
                            "maneuver. Requires \"Steer / Lane-Change With The Route\", an AutoLaneChange timer " +
                            "and a confident lane fix; the flashing arrow then shows the wanted move (with a " +
                            "pill explaining what blocks it), and the blind-spot monitor and camera still gate " +
                            "any move."),
      buttons=[lambda: tr("Left Most"), lambda: tr("Center"), lambda: tr("Right Most")],
      param="NkaoudNavHighwayLanePref",
      button_width=280,
    )

    items = [
      self._token_button,
      self._share_endpoint_button,
      self._clear_destination_button,
      self._show_polyline,
      self._polyline_style,
      self._show_banner,
      self._control_speed,
      self._control_steer,
      self._turn_assist,
      self._visual_block_threshold,
      self._highway_lane_pref,
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

  def _share_endpoint_label(self) -> str:
    url = (self._params.get("NkaoudNavShareEndpoint") or "").strip()
    if not url:
      return tr("Set Neon Connection String (for Share)")
    # Show just the Neon host so credentials don't appear in the label.
    after_scheme = url.split("://", 1)[-1]
    after_creds = after_scheme.split("@", 1)[-1]
    host = after_creds.split("/", 1)[0]
    return tr("Neon Connection (set, {})").format(host)

  def _open_share_endpoint_input(self) -> None:
    # Same QR/web-form workflow as the Mapbox token, but the form
    # shows an example JSON response so the user can sanity-check their
    # endpoint format.
    gui_app.push_widget(NavShareEndpointQrDialog())

  def _clear_destination(self) -> None:
    # Wipe both the destination AND the share trigger so a pending share
    # fetch (or one that completes after the tap) doesn't re-instate the
    # destination on the next navd tick.
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
