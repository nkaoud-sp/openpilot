"""
Lane Center Assist tuning.
"""
from collections.abc import Callable

import pyray as rl
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import multiple_button_item_sp, option_item_sp, toggle_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.network import NavButton
from openpilot.system.ui.widgets.scroller_tici import Scroller


class LaneCenterAssistSettingsLayout(Widget):
  def __init__(self, back_btn_callback: Callable):
    super().__init__()

    self._back_button = NavButton(tr("Back"))
    self._back_button.set_click_callback(back_btn_callback)

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):
    self._method = multiple_button_item_sp(
      title=lambda: tr("Method"),
      description=lambda: tr("Curvature Bias corrects the steering setpoint downstream (cooperative, inherits all " +
                            "safety limits). Camera Offset (experimental) shifts the model's input so the model " +
                            "itself re-plans centred — seamless, but the correction runs through the model and " +
                            "moves its whole view, so it is kept small."),
      buttons=[lambda: tr("Curvature Bias"), lambda: tr("Camera Offset")],
      param="LaneCenterAssistMethod",
      button_width=340,
      inline=False,
    )
    self._mode = multiple_button_item_sp(
      title=lambda: tr("Mode"),
      description=lambda: tr("Readout shows the offset from lane centre and what the assist would do, without " +
                            "steering. On adds a small, capped curvature nudge toward the centre of the ego lane " +
                            "on top of the driving model."),
      buttons=[lambda: tr("Off"), lambda: tr("Readout"), lambda: tr("On")],
      param="LaneCenterAssistMode",
      button_width=240,
      inline=False,
    )
    self._strength = multiple_button_item_sp(
      title=lambda: tr("Strength"),
      description=lambda: tr("How firmly the assist responds to an off-centre position. Higher re-centres sooner; " +
                            "the Max Correction cap still limits the total nudge."),
      buttons=[lambda: tr("Low"), lambda: tr("Medium"), lambda: tr("High")],
      param="LaneCenterAssistStrength",
      button_width=250,
      inline=False,
    )
    self._max_accel = option_item_sp(
      title=lambda: tr("Max Correction"),
      description=lambda: tr("Hard cap on the extra lateral acceleration the assist may add. Keeps the nudge gentle " +
                            "regardless of strength."),
      param="LaneCenterAssistMaxAccel",
      min_value=10,
      max_value=60,
      value_change_step=5,
      label_callback=lambda value: f"{value / 100:.2f} m/s²",
      inline=True,
    )
    self._cam_max = option_item_sp(
      title=lambda: tr("Max Camera Shift"),
      description=lambda: tr("Camera Offset method only: hard cap on the dynamic virtual-camera shift. Kept small " +
                            "because the shift moves the model's whole view, not just steering."),
      param="LaneCenterAssistCamMaxM",
      min_value=3,
      max_value=25,
      value_change_step=1,
      label_callback=lambda value: f"{value / 100:.2f} m",
      inline=True,
    )
    self._cam_damping = option_item_sp(
      title=lambda: tr("Camera Damping"),
      description=lambda: tr("Camera Offset method only: optional derivative damping (off by default). If the car " +
                            "weaves with the camera method, raise this a little at a time; if centering feels " +
                            "sluggish, lower it. 0 = pure proportional."),
      param="LaneCenterAssistCamDamping",
      min_value=0,
      max_value=100,
      value_change_step=5,
      label_callback=lambda value: f"{value / 100:.2f} s",
      inline=True,
    )
    self._min_speed = option_item_sp(
      title=lambda: tr("Minimum Speed"),
      description=lambda: tr("Lane Center Assist is disabled below this speed, where the lane centre is noisy."),
      param="LaneCenterAssistMinKph",
      min_value=5,
      max_value=90,
      value_change_step=5,
      label_callback=lambda value: f"{value} kph",
      inline=True,
    )
    self._confidence = multiple_button_item_sp(
      title=lambda: tr("Confidence Gate"),
      description=lambda: tr("How sure the assist must be of both ego lane lines before it acts. Loose (≥0.2) acts " +
                            "on faint markings, Normal (≥0.4) is balanced, Strict (≥0.6) only acts on strong, " +
                            "clear markings."),
      buttons=[lambda: tr("Loose"), lambda: tr("Normal"), lambda: tr("Strict")],
      param="LaneCenterAssistConfidence",
      button_width=220,
      inline=False,
    )
    self._path_overlay = toggle_item_sp(
      title=lambda: tr("Show Commanded Path"),
      description=lambda: tr("Draw an arc on the driving view showing the commanded curvature (the driving " +
                            "model plus the assist's bias). The model's own predicted path stays green, so the " +
                            "gap between them is what the nudge is adding. The arc tints from cyan toward red as " +
                            "the requested correction approaches your Max Correction setting."),
      param="LaneCenterAssistPathOverlay",
    )

    return [
      self._method,
      self._mode,
      self._strength,
      self._max_accel,
      self._cam_max,
      self._cam_damping,
      self._min_speed,
      self._confidence,
      self._path_overlay,
    ]

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
