"""Developer test panel: fire a hardcoded CAN frame and probe driver monitoring on demand."""
import time
from collections.abc import Callable

import pyray as rl
import openpilot.cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.selfdrive.pandad import can_capnp_to_list
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import simple_button_item_sp
from openpilot.system.ui.widgets.list_view import text_item
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller

# Face probability threshold used by the DM policy (selfdrive/monitoring/policy.py:_FACE_THRESHOLD).
FACE_THRESHOLD = 0.7

# Door status is decoded straight off the raw CAN stream so it works offroad (pandad always
# publishes 'can'). BODY_CONTROL_STATE (addr 1568) on the powertrain bus carries the door signals.
DOOR_DBC = "toyota_nodsu_pt_generated"
DOOR_MSG = "BODY_CONTROL_STATE"
DOOR_BUS = 0
DOOR_SIGNALS = ("DOOR_OPEN_FL", "DOOR_OPEN_FR", "DOOR_OPEN_RL", "DOOR_OPEN_RR")
# Same BODY_CONTROL_STATE message carries the driver seatbelt (the only seat signal in this DBC).
SEATBELT_SIGNAL = "SEATBELT_DRIVER_UNLATCHED"
# Consider the CAN reading stale if we haven't seen the message in this long (seconds).
DOOR_STALE_S = 2.0


def dmonitoringd_running(sm) -> bool:
  return any(proc.name == "dmonitoringd" and proc.running for proc in sm["managerState"].processes)


class CanTestSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()

    self._params = Params()
    self._back_btn_callback = back_btn_callback
    self._driver_check_active = False

    # Door decoding off the raw CAN stream (lazily set up while the panel is visible).
    self._can_sock = None
    self._door_parser = None
    self._door_open = False
    self._seatbelt_unlatched = False
    self._door_last_seen = 0.0

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
      text_item(
        title=lambda: tr("Ignition"),
        value=self._ignition_status,
        description=lambda: tr("Live ignition state from pandaStates (ignitionLine / ignitionCan). Works offroad " +
                              "since pandad always runs."),
      ),
      text_item(
        title=lambda: tr("Door Open"),
        value=self._door_status,
        description=lambda: tr("Decodes BODY_CONTROL_STATE off the raw CAN stream, so it works offroad as long " +
                              "as the powertrain bus is awake. Toyota-specific (toyota_nodsu_pt_generated)."),
      ),
      text_item(
        title=lambda: tr("Driver Seatbelt"),
        value=self._seatbelt_status,
        description=lambda: tr("Driver seatbelt latched state from BODY_CONTROL_STATE (same offroad CAN decode). " +
                              "This DBC exposes no seat-occupancy signal, so only the driver belt is available."),
      ),
    ]

  def _ignition_status(self) -> str:
    return tr("On") if ui_state.ignition else tr("Off")

  def _start_door_parser(self):
    if self._can_sock is not None:
      return
    try:
      from opendbc.can import CANParser
      self._door_parser = CANParser(DOOR_DBC, [(DOOR_MSG, 0)], DOOR_BUS)
      self._can_sock = messaging.sub_sock("can", conflate=False, timeout=0)
    except Exception:
      self._can_sock = None
      self._door_parser = None
    self._door_open = False
    self._door_last_seen = 0.0

  def _stop_door_parser(self):
    self._can_sock = None
    self._door_parser = None

  def _update_door_state(self):
    if self._can_sock is None or self._door_parser is None:
      return
    raw = messaging.drain_sock_raw(self._can_sock)
    if not raw:
      return
    # The parser tracks only BODY_CONTROL_STATE, so any updated address means we saw it this cycle.
    updated = self._door_parser.update(can_capnp_to_list(raw))
    msg = self._door_parser.vl[DOOR_MSG]
    self._door_open = any(msg[s] for s in DOOR_SIGNALS)
    self._seatbelt_unlatched = bool(msg[SEATBELT_SIGNAL])
    if updated:
      self._door_last_seen = time.monotonic()

  def _door_status(self) -> str:
    if self._door_parser is None:
      return tr("Unavailable (no DBC)")
    if self._door_last_seen == 0.0:
      return tr("Waiting for CAN...")
    stale = (time.monotonic() - self._door_last_seen) > DOOR_STALE_S
    state = tr("Open") if self._door_open else tr("Closed")
    return f"{state} ({tr('stale')})" if stale else state

  def _seatbelt_status(self) -> str:
    if self._door_parser is None:
      return tr("Unavailable (no DBC)")
    if self._door_last_seen == 0.0:
      return tr("Waiting for CAN...")
    stale = (time.monotonic() - self._door_last_seen) > DOOR_STALE_S
    state = tr("Unlatched") if self._seatbelt_unlatched else tr("Latched")
    return f"{state} ({tr('stale')})" if stale else state

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
    # tweaks.py doesn't call hide_event when navigating away, so clean up here too.
    self._set_driver_check(False)
    self._stop_door_parser()
    self._back_btn_callback()

  def _update_state(self):
    super()._update_state()
    self._update_door_state()

  def _render(self, rect):
    self._back_button.set_position(self._rect.x, self._rect.y + 20)
    self._back_button.render()

    content_rect = rl.Rectangle(rect.x, rect.y + self._back_button.rect.height + 40,
                                rect.width, rect.height - self._back_button.rect.height - 40)
    self._scroller.render(content_rect)

  def show_event(self):
    self._start_door_parser()
    self._scroller.show_event()

  def hide_event(self):
    self._set_driver_check(False)
    self._stop_door_parser()
    self._scroller.hide_event()
