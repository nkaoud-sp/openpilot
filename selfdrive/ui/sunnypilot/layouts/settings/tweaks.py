"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntEnum

from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.auto_lock_settings import AutoLockSettingsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.dynamic_follow_settings import DynamicFollowSettingsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.jerk_settings import JerkSettingsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.launch_assist_settings import LaunchAssistSettingsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.nav_settings import NavSettingsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.park_assist_settings import ParkAssistSettingsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.tweaks_sub_layouts.visual_vehicle_settings import VisualVehicleSettingsLayout
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, simple_button_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller_tici import Scroller


class PanelType(IntEnum):
  TWEAKS = 0
  AUTO_LOCK = 1
  DYNAMIC_FOLLOW = 2
  JERK = 3
  LAUNCH = 4
  PARK = 5
  NAVIGATION = 6
  VISUAL_VEHICLE = 7


class TweaksLayout(Widget):
  def __init__(self):
    super().__init__()

    self._current_panel = PanelType.TWEAKS
    self._auto_lock_layout = AutoLockSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._dynamic_follow_layout = DynamicFollowSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._jerk_layout = JerkSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._launch_layout = LaunchAssistSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._park_layout = ParkAssistSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._navigation_layout = NavSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))
    self._visual_vehicle_layout = VisualVehicleSettingsLayout(lambda: self._set_current_panel(PanelType.TWEAKS))

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    # Secure-on-exit (doorlockd). Toyota/Lexus only.
    # Detailed settings live in a sub-page reachable via the Manage button below.
    self._auto_lock = toggle_item_sp(
      title=lambda: tr("Auto Lock On Exit"),
      description=lambda: tr("Lock the doors (and optionally fold the mirrors / close the windows) once you leave " +
                            "the car. Toyota/Lexus only."),
      param="AutoLockEnabled",
    )
    self._auto_lock_button = simple_button_item_sp(
      button_text=lambda: tr("Manage Auto Lock Settings"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.AUTO_LOCK),
    )

    # Dynamic follow distance (speed-based gap, overrides the personality setting).
    # Detailed settings live in a sub-page reachable via the Manage button below.
    self._dynamic_follow = toggle_item_sp(
      title=lambda: tr("Dynamic Follow Distance"),
      description=lambda: tr("Vary the follow distance with vehicle speed instead of using the fixed driving " +
                            "personality gap. The follow time scales linearly from the low-speed value (at 0 km/h) " +
                            "to the high-speed value (at 130 km/h). Requires openpilot longitudinal control."),
      param="DynamicFollow",
    )
    self._dynamic_follow_button = simple_button_item_sp(
      button_text=lambda: tr("Manage Dynamic Follow Settings"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.DYNAMIC_FOLLOW),
    )

    # Asymmetric jerk (separate accel / decel ramp smoothness).
    # Detailed settings live in a sub-page reachable via the Manage button below.
    self._asymmetric_jerk = toggle_item_sp(
      title=lambda: tr("Asymmetric Accel / Decel Smoothness"),
      description=lambda: tr("Tune how gently acceleration and braking build up independently, so braking can " +
                            "ramp in more smoothly than acceleration (or vice-versa). Requires openpilot " +
                            "longitudinal control."),
      param="AsymmetricJerk",
    )
    self._asymmetric_jerk_button = simple_button_item_sp(
      button_text=lambda: tr("Manage Accel / Decel Smoothness"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.JERK),
    )

    # Lead-departure launch assist (launch sooner when a stopped lead pulls away).
    # Detailed settings live in a sub-page reachable via the Manage button below.
    self._launch_assist = toggle_item_sp(
      title=lambda: tr("Lead Launch Assist"),
      description=lambda: tr("When stopped behind a lead that pulls away, launch sooner instead of waiting for " +
                            "the model. The radar-based planner still enforces the safe gap, and it only acts " +
                            "from a full stop and never overrides the brake or gas. Requires openpilot " +
                            "longitudinal control."),
      param="LaunchAssist",
    )
    self._launch_assist_button = simple_button_item_sp(
      button_text=lambda: tr("Manage Launch Assist Settings"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.LAUNCH),
    )

    # Lead park assist (closer standstill gap behind a stopped lead).
    # Detailed settings live in a sub-page reachable via the Manage button below.
    self._park_assist = toggle_item_sp(
      title=lambda: tr("Lead Halt Assist"),
      description=lambda: tr("When stopped behind a stopped lead, settle at a closer gap than the default. The " +
                            "gap smoothly returns to normal once the lead moves. Only acts near a standstill. " +
                            "Requires openpilot longitudinal control."),
      param="ParkAssist",
    )
    self._park_assist_button = simple_button_item_sp(
      button_text=lambda: tr("Manage Halt Assist Settings"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.PARK),
    )

    # Lane position indicator: small on-screen widget with squares for each lane
    # detected and the current lane filled. Border colour reflects detection
    # confidence (green/amber/red). Estimated from modelV2 road edges.
    self._lane_position_indicator = toggle_item_sp(
      title=lambda: tr("Lane Position Indicator"),
      description=lambda: tr("Small on-screen indicator showing how many lanes are detected and which one you're " +
                            "in (e.g. [□□■□]). The border colour reflects detection " +
                            "confidence: green = high, amber = medium, red = low."),
      param="LanePositionIndicator",
    )

    # Standalone visual adjacent-vehicle detector. UI/debug only; no controls integration.
    self._visual_vehicle_detector = toggle_item_sp(
      title=lambda: tr("Visual Vehicle Detector (test)"),
      description=lambda: tr("Run a standalone camera detector for nearby left/right vehicles and show a large " +
                            "debug readout on the driving view. This is UI/debug only and does not control or " +
                            "block lane changes."),
      param="VisualVehicleDetector",
    )
    self._visual_vehicle_button = simple_button_item_sp(
      button_text=lambda: tr("Manage Visual Detector Settings"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.VISUAL_VEHICLE),
    )

    # Experimental Mapbox-based navigation. Polyline overlay + maneuver banner + optional
    # turn-slowdown. Master gate starts the nkaoud_navd process; submenu has the rest.
    self._navigation = toggle_item_sp(
      title=lambda: tr("Navigation (experimental)"),
      description=lambda: tr("Route to a preset destination with a Mapbox-fetched polyline and maneuver banner. " +
                            "Optionally slow for upcoming turns. Experimental — visual layer first, control " +
                            "is opt-in inside the submenu."),
      param="NkaoudNavEnabled",
    )
    self._navigation_button = simple_button_item_sp(
      button_text=lambda: tr("Manage Navigation Settings"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.NAVIGATION),
    )

    items = [
      self._auto_lock,
      self._auto_lock_button,
      self._dynamic_follow,
      self._dynamic_follow_button,
      self._asymmetric_jerk,
      self._asymmetric_jerk_button,
      self._launch_assist,
      self._launch_assist_button,
      self._park_assist,
      self._park_assist_button,
      self._lane_position_indicator,
      self._visual_vehicle_detector,
      self._visual_vehicle_button,
      self._navigation,
      self._navigation_button,
    ]
    return items

  def _render(self, rect):
    if self._current_panel == PanelType.AUTO_LOCK:
      self._auto_lock_layout.render(rect)
    elif self._current_panel == PanelType.DYNAMIC_FOLLOW:
      self._dynamic_follow_layout.render(rect)
    elif self._current_panel == PanelType.JERK:
      self._jerk_layout.render(rect)
    elif self._current_panel == PanelType.LAUNCH:
      self._launch_layout.render(rect)
    elif self._current_panel == PanelType.PARK:
      self._park_layout.render(rect)
    elif self._current_panel == PanelType.NAVIGATION:
      self._navigation_layout.render(rect)
    elif self._current_panel == PanelType.VISUAL_VEHICLE:
      self._visual_vehicle_layout.render(rect)
    else:
      self._scroller.render(rect)

  def show_event(self):
    self._set_current_panel(PanelType.TWEAKS)
    self._scroller.show_event()

  def _set_current_panel(self, panel: PanelType):
    self._current_panel = panel
    if panel == PanelType.AUTO_LOCK:
      self._auto_lock_layout.show_event()
    elif panel == PanelType.DYNAMIC_FOLLOW:
      self._dynamic_follow_layout.show_event()
    elif panel == PanelType.JERK:
      self._jerk_layout.show_event()
    elif panel == PanelType.LAUNCH:
      self._launch_layout.show_event()
    elif panel == PanelType.PARK:
      self._park_layout.show_event()
    elif panel == PanelType.NAVIGATION:
      self._navigation_layout.show_event()
    elif panel == PanelType.VISUAL_VEHICLE:
      self._visual_vehicle_layout.show_event()

  def _update_state(self):
    super()._update_state()

    # the Manage buttons are only available once their feature is enabled
    self._auto_lock_button.action_item.set_enabled(self._auto_lock.action_item.get_state())
    self._dynamic_follow_button.action_item.set_enabled(self._dynamic_follow.action_item.get_state())
    self._asymmetric_jerk_button.action_item.set_enabled(self._asymmetric_jerk.action_item.get_state())
    self._launch_assist_button.action_item.set_enabled(self._launch_assist.action_item.get_state())
    self._park_assist_button.action_item.set_enabled(self._park_assist.action_item.get_state())
    self._navigation_button.action_item.set_enabled(self._navigation.action_item.get_state())
    self._visual_vehicle_button.action_item.set_enabled(self._visual_vehicle_detector.action_item.get_state())
