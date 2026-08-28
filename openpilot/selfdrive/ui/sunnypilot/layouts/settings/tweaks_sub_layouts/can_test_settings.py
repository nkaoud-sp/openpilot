"""Developer test panel: fire a hardcoded CAN frame and probe driver monitoring on demand."""
from collections.abc import Callable

import pyray as rl
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import simple_button_item_sp
from openpilot.system.ui.widgets.list_view import text_item
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller

# Face probability threshold used by the DM policy (selfdrive/monitoring/policy.py:_FACE_THRESHOLD).
FACE_THRESHOLD = 0.7


def dmonitoringd_running(sm) -> bool:
  return any(proc.name == "dmonitoringd" and proc.running for proc in sm["managerState"].processes)


class CanTestSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()

    self._params = Params()
    self._back_btn_callback = back_btn_callback
    self._driver_check_active = False
    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(self._on_back)

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._send_button = simple_button_item_sp(
      button_text=lambda: tr("Send Test CAN Frame"),
      button_width=800,
      callback=self._on_send,
    )
    self._driver_button = simple_button_item_sp(
      button_text=lambda: tr("Stop Driver Check") if self._driver_check_active else tr("Start Driver Check"),
      button_width=800,
      callback=self._on_toggle_driver_check,
    )
    return [
      text_item(
        title=lambda: tr("CAN Frame"),
        value=lambda: "0x750 · bus 0",
        description=lambda: tr("Sends 40 05 30 11 00 80 00 00 to address 0x750 on bus 0 once per press. Works " +
                              "offroad: pandad briefly switches the panda to ELM327 diagnostic mode, sends the " +
                              "frame, then reverts. Requires a panda connected to a live CAN bus."),
      ),
      self._send_button,
      text_item(
        title=lambda: tr("Driver Monitoring"),
        value=self._driver_status,
        description=lambda: tr("Starts the driver camera and DM model (via IsDriverViewEnabled) so it works " +
                              "offroad, then reports face detection for both seats (driver = wheel side). The " +
                              "camera takes a few seconds to warm up. Stop the check to shut the camera back off."),
      ),
      self._driver_button,
    ]

  def _on_send(self):
    # pandad reads this trigger in its health loop, sends the frame offroad, and clears it.
    self._params.put_bool("CanTestTrigger", True)

  def _on_toggle_driver_check(self):
    self._set_driver_check(not self._driver_check_active)

  def _set_driver_check(self, active: bool):
    if active == self._driver_check_active:
      return
    self._driver_check_active = active
    # IsDriverViewEnabled makes camerad + dmonitoringmodeld + dmonitoringd run offroad (process_config.py).
    self._params.put_bool("IsDriverViewEnabled", active, block=True)

  def _driver_status(self) -> str:
    if not self._driver_check_active:
      return tr("Press start to check")

    sm = ui_state.sm
    if not dmonitoringd_running(sm) or not sm.alive["driverStateV2"]:
      return tr("Starting driver camera...")

    ds = sm["driverStateV2"]
    # The DM model reports a face probability for both seats; the wheel side is the driver.
    if ds.wheelOnRightProb > 0.5:
      driver_prob, passenger_prob = ds.rightDriverData.faceProb, ds.leftDriverData.faceProb
    else:
      driver_prob, passenger_prob = ds.leftDriverData.faceProb, ds.rightDriverData.faceProb

    def label(name: str, prob: float) -> str:
      seen = tr("yes") if prob > FACE_THRESHOLD else tr("no")
      return f"{name}: {seen} ({prob * 100:.0f}%)"

    return f"{label(tr('Driver'), driver_prob)}   {label(tr('Passenger'), passenger_prob)}"

  def _on_back(self):
    # tweaks.py doesn't call hide_event when navigating away, so shut the camera off here too.
    self._set_driver_check(False)
    self._back_btn_callback()

  def _render(self, rect):
    self._back_button.set_position(self._rect.x, self._rect.y + 20)
    self._back_button.render()

    content_rect = rl.Rectangle(rect.x, rect.y + self._back_button.rect.height + 40,
                                rect.width, rect.height - self._back_button.rect.height - 40)
    self._scroller.render(content_rect)

  def show_event(self):
    self._scroller.show_event()

  def hide_event(self):
    self._set_driver_check(False)
    self._scroller.hide_event()
