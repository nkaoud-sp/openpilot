"""
Settings sub-page for nkaoud_nav navigation-maneuver logging and email delivery.

Reachable from Tweaks -> "Email & Logs". Holds the drive-logging toggle, the
auto-email-after-drive toggle, and the SMTP web-form button (which reuses the
QR-dialog ParamWebServer pattern). The email button's description surfaces the
last logging/email status so the user can confirm sends from the device.
"""
import json
from collections.abc import Callable

import pyray as rl
from openpilot.common.params import Params
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.nav_token_qr_dialog import (
  NavEmailConfigQrDialog,
)
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, button_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller


class EmailLogsSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()

    self._params = Params()
    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._drive_logging = toggle_item_sp(
      title=lambda: tr("Log Navigation Maneuvers"),
      description=lambda: tr("While actively navigating a route, record maneuver type, distance, lane " +
                            "recommendations and cross-track error to a per-drive CSV (in " +
                            "/data/media/0/nkaoud_nav_logs) for later analysis and tuning. Requires Navigation " +
                            "to be enabled with a destination set."),
      param="NkaoudNavDriveLogging",
    )
    self._auto_email = toggle_item_sp(
      title=lambda: tr("Email Log After Each Drive"),
      description=lambda: tr("When a drive ends, automatically email the maneuver log as a CSV attachment, then " +
                            "delete the on-device logs once the email is sent. Requires \"Log Navigation " +
                            "Maneuvers\" and configured Email (SMTP) settings below."),
      param="NkaoudNavAutoEmail",
    )
    self._email_config_button = button_item_sp(
      title=lambda: self._email_config_title(),
      button_text=lambda: tr("Configure"),
      description=lambda: self._email_config_description(),
      callback=lambda: self._open_email_config_input(),
    )

    return [
      self._drive_logging,
      self._auto_email,
      self._email_config_button,
    ]

  def _email_config_title(self) -> str:
    cfg = (self._params.get("NkaoudNavEmailConfig") or "").strip()
    if not cfg:
      return tr("Email (SMTP) Settings")
    # Show just the recipient so credentials never appear in the title.
    to = ""
    try:
      to = str(json.loads(cfg).get("to", "")).strip()
    except (ValueError, AttributeError):
      to = ""
    return tr("Email (SMTP) Settings (to {})").format(to) if to else tr("Email (SMTP) Settings (set)")

  def _email_config_description(self) -> str:
    status = (self._params.get("NkaoudNavEmailLastStatus") or "").strip()
    base = tr("Enter your SMTP settings as JSON and send yourself a test email before saving.")
    if status:
      return base + "\n" + tr("Last status: {}").format(status)
    return base

  def _open_email_config_input(self) -> None:
    # QR/web-form workflow; the form's Test button sends a real test email
    # to confirm the SMTP settings before saving.
    gui_app.push_widget(NavEmailConfigQrDialog())

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
