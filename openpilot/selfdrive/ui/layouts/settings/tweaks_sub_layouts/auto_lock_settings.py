from openpilot.selfdrive.ui.layouts.settings.tweaks_sub_layouts.base import TweaksSubLayout
from openpilot.system.ui.widgets.list_view import toggle_item
from openpilot.system.ui.lib.multilang import tr


class AutoLockSettingsLayout(TweaksSubLayout):
  def _build_items(self):
    self._auto_lock_on = self._params.get_bool("AutoDoorLock")

    self._auto_door_lock = toggle_item(
      lambda: tr("Auto Door Lock"),
      description=lambda: tr("After the car is switched off and openpilot is offroad, wait for the driver to get " +
                             "out, confirm the cabin is empty with the driver camera, then send a door-lock CAN " +
                             "command. Re-checks after each door open/close and only locks once nobody is inside. " +
                             "Toyota-specific and offroad only."),
      initial_state=self._auto_lock_on,
      callback=self._on_auto_lock,
    )
    self._auto_close_windows = toggle_item(
      lambda: tr("Auto Close Windows"),
      description=lambda: tr("Also close all windows before locking."),
      initial_state=self._params.get_bool("AutoDoorLockCloseWindows"),
      callback=lambda state: self._params.put_bool("AutoDoorLockCloseWindows", state),
      enabled=lambda: self._auto_lock_on,
    )
    self._auto_fold_mirrors = toggle_item(
      lambda: tr("Auto Fold Mirrors"),
      description=lambda: tr("Also fold the side mirrors before locking."),
      initial_state=self._params.get_bool("AutoDoorLockFoldMirrors"),
      callback=lambda state: self._params.put_bool("AutoDoorLockFoldMirrors", state),
      enabled=lambda: self._auto_lock_on,
    )
    return [
      self._auto_door_lock,
      self._auto_close_windows,
      self._auto_fold_mirrors,
    ]

  def _on_auto_lock(self, state: bool):
    self._auto_lock_on = state
    self._params.put_bool("AutoDoorLock", state)

  def show_event(self):
    super().show_event()
    self._auto_lock_on = self._params.get_bool("AutoDoorLock")
    self._auto_door_lock.action_item.set_state(self._auto_lock_on)
