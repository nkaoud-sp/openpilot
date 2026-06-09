"""
QR-code-based param-entry dialog for nkaoud_nav. Generic over which param
gets set -- the spec (title, hint, example) lives in the ParamWebServer.

Lifecycle:
  show() -> server.start() + generate QR for http://<lan-ip>:8081/
  user scans -> opens form on phone -> pastes value -> POSTs
  server sets a threading.Event
  dialog detects on the next frame, shows "Saved", stops the server, pops itself
  cancel button -> stops the server, pops itself
"""
from __future__ import annotations

import pyray as rl
import qrcode

from openpilot.sunnypilot.nkaoud_nav.token_server import (
  ParamWebServer, mapbox_token_server, share_endpoint_server,
)
from openpilot.sunnypilot.nkaoud_nav.preview_server import visual_vehicle_preview_server
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import Button, ButtonStyle


MARGIN = 100
QR_PADDING_MODULES = 4         # quiet zone on each side of the QR
BG_COLOR = rl.Color(27, 27, 27, 255)
QR_BG_COLOR = rl.Color(255, 255, 255, 255)
QR_FG_COLOR = rl.Color(0, 0, 0, 255)
SUCCESS_COLOR = rl.Color(128, 216, 166, 255)
URL_COLOR = rl.Color(220, 220, 220, 255)
HINT_COLOR = rl.Color(150, 150, 150, 255)
AUTO_CLOSE_FRAMES = 60         # how long to display the success state before popping


class NavParamQrDialog(Widget):
  """QR + spawn-on-demand web server for any nkaoud_nav param. Construct
  with a fully-configured ParamWebServer (use the factories in
  sunnypilot.nkaoud_nav.token_server). `title_text` and `hint_text` show
  on the dialog itself (the web form's own title/example come from the
  spec)."""

  def __init__(self, server: ParamWebServer, title_text: str, hint_text: str) -> None:
    super().__init__()
    self._server = server
    self._server.start()
    self._url = self._server.url
    self._matrix = self._build_matrix(self._url)
    self._title_text = title_text
    self._hint_text = hint_text
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_medium = gui_app.font(FontWeight.MEDIUM)
    self._font_mono_size = 36
    self._title_size = 56
    self._hint_size = 32
    self._success_frames_left = 0
    self._cancel_button = Button(lambda: tr("Cancel"), click_callback=self._on_cancel)
    self._close_button = Button(lambda: tr("Close"), click_callback=self._on_cancel,
                                button_style=ButtonStyle.PRIMARY)

  # ---- lifecycle ----
  def _on_cancel(self) -> None:
    self._teardown_and_pop()

  def _teardown_and_pop(self) -> None:
    self._server.stop()
    gui_app.pop_widget()

  # ---- helpers ----
  @staticmethod
  def _build_matrix(url: str) -> list[list[bool]]:
    qr = qrcode.QRCode(
      version=None,
      error_correction=qrcode.constants.ERROR_CORRECT_M,
      box_size=1,
      border=QR_PADDING_MODULES,
    )
    qr.add_data(url)
    qr.make(fit=True)
    return qr.get_matrix()

  # ---- render ----
  def _render(self, rect: rl.Rectangle) -> None:
    # Detect "saved" once -- arm the close countdown.
    if self._success_frames_left == 0 and self._server.token_saved.is_set():
      self._success_frames_left = AUTO_CLOSE_FRAMES

    dialog_rect = rl.Rectangle(rect.x + MARGIN, rect.y + MARGIN,
                               rect.width - 2 * MARGIN, rect.height - 2 * MARGIN)
    rl.draw_rectangle_rounded(dialog_rect, 0.02, 16, BG_COLOR)

    content_x = dialog_rect.x + MARGIN
    content_w = dialog_rect.width - 2 * MARGIN
    y = dialog_rect.y + MARGIN

    # Title
    rl.draw_text_ex(self._font_bold, self._title_text,
                    rl.Vector2(content_x, y), self._title_size, 0, rl.WHITE)
    y += self._title_size + 24

    # Status block
    if self._success_frames_left > 0:
      self._render_success_state(content_x, y, content_w, dialog_rect)
    else:
      self._render_qr_state(content_x, y, content_w, dialog_rect)

  def _render_qr_state(self, content_x: float, y: float, content_w: float,
                       dialog_rect: rl.Rectangle) -> None:
    rl.draw_text_ex(self._font_medium, self._hint_text,
                    rl.Vector2(content_x, y), self._hint_size, 0, HINT_COLOR)
    y += self._hint_size + 24

    bottom = dialog_rect.y + dialog_rect.height
    cancel_h = 160
    cancel_y = bottom - cancel_h - MARGIN
    url_h = self._font_mono_size + 16
    qr_top = y
    qr_bottom = cancel_y - url_h - 32
    qr_area_h = max(100, qr_bottom - qr_top)
    qr_size = min(content_w, qr_area_h)
    qr_x = content_x + (content_w - qr_size) / 2

    self._draw_qr(qr_x, qr_top, qr_size)

    # URL caption
    url_size = measure_text_cached(self._font_medium, self._url, self._font_mono_size)
    rl.draw_text_ex(self._font_medium, self._url,
                    rl.Vector2(content_x + (content_w - url_size.x) / 2, qr_bottom + 16),
                    self._font_mono_size, 0, URL_COLOR)

    # Cancel button
    cancel_rect = rl.Rectangle(content_x, cancel_y, content_w, cancel_h)
    self._cancel_button.render(cancel_rect)

  def _render_success_state(self, content_x: float, y: float, content_w: float,
                            dialog_rect: rl.Rectangle) -> None:
    msg = tr("Saved. Closing...")
    text_size = measure_text_cached(self._font_bold, msg, 64)
    bottom = dialog_rect.y + dialog_rect.height
    rl.draw_text_ex(self._font_bold, msg,
                    rl.Vector2(content_x + (content_w - text_size.x) / 2,
                               y + (bottom - y) / 2 - text_size.y / 2),
                    64, 0, SUCCESS_COLOR)

    cancel_h = 160
    cancel_y = bottom - cancel_h - MARGIN
    close_rect = rl.Rectangle(content_x, cancel_y, content_w, cancel_h)
    self._close_button.render(close_rect)

    self._success_frames_left -= 1
    if self._success_frames_left <= 0:
      self._teardown_and_pop()

  def _draw_qr(self, qr_x: float, qr_y: float, qr_size: float) -> None:
    modules = len(self._matrix)
    if modules == 0:
      return
    cell = qr_size / modules
    # White background (covers the quiet zone too).
    rl.draw_rectangle(int(qr_x), int(qr_y), int(qr_size), int(qr_size), QR_BG_COLOR)
    for r, row in enumerate(self._matrix):
      for c, bit in enumerate(row):
        if not bit:
          continue
        # +0.5 / -1 trick to avoid hairline gaps between adjacent cells.
        rl.draw_rectangle(
          int(qr_x + c * cell),
          int(qr_y + r * cell),
          int(cell + 1),
          int(cell + 1),
          QR_FG_COLOR,
        )


# ---------- Preconfigured dialog factories ----------

def NavTokenQrDialog() -> NavParamQrDialog:
  return NavParamQrDialog(
    server=mapbox_token_server(),
    title_text=tr("Set Mapbox Token"),
    hint_text=tr("Scan with your phone, then paste your Mapbox token in the form."),
  )


def NavShareEndpointQrDialog() -> NavParamQrDialog:
  return NavParamQrDialog(
    server=share_endpoint_server(),
    title_text=tr("Set Neon Connection String"),
    hint_text=tr("Scan with your phone, then paste your Neon connection string. The page shows the required table layout."),
  )


def VisualVehiclePreviewQrDialog() -> NavParamQrDialog:
  # Reuses NavParamQrDialog: the preview server is duck-typed compatible
  # with ParamWebServer (start/stop/url/token_saved). token_saved is never
  # set, so the dialog stays on QR view until the user taps Cancel.
  return NavParamQrDialog(
    server=visual_vehicle_preview_server(),
    title_text=tr("Detector Live Preview"),
    hint_text=tr("Scan with your phone to view the exact 320x320 RGB tensor the detector sees. Refreshes every second."),
  )
