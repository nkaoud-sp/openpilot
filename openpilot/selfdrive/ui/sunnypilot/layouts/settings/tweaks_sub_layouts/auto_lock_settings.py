"""Auto door lock settings: master toggle plus optional close-windows and fold-mirrors."""
from collections.abc import Callable

import pyray as rl
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller


class AutoLockSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()

    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._auto_door_lock = toggle_item_sp(
      title=lambda: tr("Auto Door Lock"),
      description=lambda: tr("After the car is switched off and openpilot is offroad, wait for the driver to get " +
                            "out, confirm the cabin is empty with the driver camera, then send a door-lock CAN " +
                            "command. Re-checks after each door open/close and only locks once nobody is inside. " +
                            "Toyota-specific and offroad only."),
      param="AutoDoorLock",
    )
    self._auto_close_windows = toggle_item_sp(
      title=lambda: tr("Auto Close Windows"),
      description=lambda: tr("Also close all windows before locking."),
      param="AutoDoorLockCloseWindows",
    )
    self._auto_fold_mirrors = toggle_item_sp(
      title=lambda: tr("Auto Fold Mirrors"),
      description=lambda: tr("Also fold the side mirrors before locking."),
      param="AutoDoorLockFoldMirrors",
    )
    self._auto_hazard = toggle_item_sp(
      title=lambda: tr("Blink Hazards After Locking"),
      description=lambda: tr("Flash the hazard lights once (~1 s) after locking to confirm."),
      param="AutoDoorLockHazard",
    )
    return [
      self._auto_door_lock,
      self._auto_close_windows,
      self._auto_fold_mirrors,
      self._auto_hazard,
    ]

  def _render(self, rect):
    self._back_button.set_position(self._rect.x, self._rect.y + 20)
    self._back_button.render()

    content_rect = rl.Rectangle(rect.x, rect.y + self._back_button.rect.height + 40,
                                rect.width, rect.height - self._back_button.rect.height - 40)
    self._scroller.render(content_rect)

  def _update_state(self):
    super()._update_state()
    # The window/mirror/hazard options only matter when auto door lock is enabled.
    auto_lock_on = self._auto_door_lock.action_item.get_state()
    self._auto_close_windows.action_item.set_enabled(auto_lock_on)
    self._auto_fold_mirrors.action_item.set_enabled(auto_lock_on)
    self._auto_hazard.action_item.set_enabled(auto_lock_on)

  def show_event(self):
    self._scroller.show_event()

  def hide_event(self):
    self._scroller.hide_event()
