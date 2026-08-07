"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np
import pyray as rl
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.sunnypilot.onroad.nav_route_overlay import NavRouteOverlay
from openpilot.selfdrive.ui.sunnypilot.onroad.nav_path_overlay import NavPathOverlay
from openpilot.selfdrive.ui.ui_state import UIStatus, ui_state
from openpilot.system.ui.lib.application import gui_app

BORDER_COLORS_SP = {
  UIStatus.LAT_ONLY: rl.Color(0x00, 0xC8, 0xC8, 0xFF),  # Cyan for lateral-only state
  UIStatus.LONG_ONLY: rl.Color(0x96, 0x1C, 0xA8, 0xFF),  # Purple for longitudinal-only state
}


class AugmentedRoadViewSP:
  def __init__(self):
    self._fade_texture = gui_app.texture("icons_mici/onroad/onroad_fade.png")
    self._fade_alpha_filter = FirstOrderFilter(0, 0.1, 1 / gui_app.target_fps)
    self.nav_route_overlay = NavRouteOverlay()
    self.nav_path_overlay = NavPathOverlay()

  def update_fade_out_bottom_overlay(self, _content_rect):
    # Fade out bottom of overlays for looks (only when engaged)
    fade_alpha = self._fade_alpha_filter.update(ui_state.status != UIStatus.DISENGAGED)
    if ui_state.torque_bar and fade_alpha > 1e-2:
      # Scale the fade texture to the content rect
      rl.draw_texture_pro(self._fade_texture,
                          rl.Rectangle(0, 0, self._fade_texture.width, self._fade_texture.height),
                          _content_rect, rl.Vector2(0, 0), 0.0,
                          rl.Color(255, 255, 255, int(255 * fade_alpha)))

  def set_nav_route_transform(self, transform: np.ndarray) -> None:
    self.nav_route_overlay.set_transform(transform)
    self.nav_path_overlay.set_transform(transform)

  def render_nav_route(self, content_rect: rl.Rectangle) -> None:
    self.nav_route_overlay.render(content_rect)
    self.nav_path_overlay.render(content_rect)
