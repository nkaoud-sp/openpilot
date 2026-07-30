"""
Onroad button to open the Visual Vehicle Detector image-capture portal.

Sits next to the driver-monitoring widget. Tapping it pushes the capture
QR dialog (which starts the capture web server); while that dialog is open
and the car is onroad, the detector records the selected camera's crop for
training. A red dot marks active recording.

Only visible when the VisualVehicleDetector feature and image-capture button are enabled.
"""
from __future__ import annotations

import time

import pyray as rl
from openpilot.common.params import Params
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.nav_token_qr_dialog import (
  VisualVehicleCaptureQrDialog,
)
from openpilot.sunnypilot.nkaoud_nav.adjacent_vehicle_detector import capture_requested
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget


class VisualVehicleCaptureButton(Widget):
  def __init__(self, button_size: int):
    super().__init__()
    self._params = Params()
    self._rect = rl.Rectangle(0, 0, button_size, button_size)
    self._black_bg = rl.Color(0, 0, 0, 166)
    self._white = rl.Color(255, 255, 255, 255)
    self._rec_color = rl.Color(255, 70, 70, 255)
    self._font = gui_app.font(FontWeight.BOLD)
    self._enabled = False
    self._recording = False
    self._last_poll = 0.0
    # Hidden (and so inert to touches) unless the feature is enabled.
    self.set_visible(lambda: self._enabled)

  def set_rect(self, rect: rl.Rectangle) -> None:
    self._rect.x, self._rect.y = rect.x, rect.y

  def _update_state(self) -> None:
    now = time.monotonic()
    if now - self._last_poll < 0.5:
      return
    self._last_poll = now
    self._enabled = self._params.get_bool("VisualVehicleDetector") and self._params.get_bool("VisualVehicleDetectorCapture")
    self._recording = capture_requested()

  def _handle_mouse_release(self, _):
    super()._handle_mouse_release(_)
    if not self._enabled:
      return
    dialog = VisualVehicleCaptureQrDialog()
    gui_app.push_widget(dialog)
    self._dialog_ref = dialog  # keep alive

  def _render(self, rect: rl.Rectangle) -> None:
    cx = int(self._rect.x + self._rect.width // 2)
    cy = int(self._rect.y + self._rect.height // 2)
    self._white.a = 180 if self.is_pressed else 255

    rl.draw_circle(cx, cy, self._rect.width / 2, self._black_bg)

    label = "CAP"
    font_size = 56
    text_w = measure_text_cached(self._font, label, font_size).x
    rl.draw_text_ex(self._font, label,
                    rl.Vector2(cx - text_w / 2, cy - font_size / 2),
                    font_size, 0, self._white)

    if self._recording:
      rl.draw_circle(cx, int(cy + font_size / 2 + 18), 10, self._rec_color)
