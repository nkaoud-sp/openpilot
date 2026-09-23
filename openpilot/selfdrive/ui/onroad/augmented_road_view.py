import time

import numpy as np
import pyray as rl
from openpilot.cereal import log
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.common.hardware import COMMA_HARDWARE
from openpilot.selfdrive.ui import UI_BORDER_SIZE
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.selfdrive.ui.onroad.alert_renderer import AlertRenderer
from openpilot.selfdrive.ui.onroad.driver_state import DriverStateRenderer
from openpilot.selfdrive.ui.onroad.hud_renderer import HudRenderer
from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer
from openpilot.selfdrive.ui.onroad.cameraview import CameraView
from openpilot.selfdrive.ui.onroad.reproject_debug import ReprojectDebug
from openpilot.system.ui.lib.application import gui_app
from openpilot.common.transformations.camera import DEVICE_CAMERAS, DeviceCameraConfig, view_frame_from_device_frame
from openpilot.common.transformations.model import medmodel_intrinsics, sbigmodel_intrinsics, MEDMODEL_INPUT_SIZE
from openpilot.common.transformations.orientation import rot_from_euler

OpState = log.SelfdriveState.OpenpilotState
CALIBRATED = log.ExtrinsicsCalibration.Status.calibrated
NARROW_ROAD_CAM = VisionStreamType.VISION_STREAM_NARROW_ROAD
WIDE_CAM = VisionStreamType.VISION_STREAM_WIDE_ROAD
DEFAULT_DEVICE_CAMERA = DEVICE_CAMERAS["tici", "ar0231"]
CAMERAD = "camerad"
REPROJECT = "reproject"  # reprojectd's comma 4 frames
MODEL_INPUT = "modelinput"  # those frames warped into the model's 512x256 inputs
VIEW_SERVERS = {0: CAMERAD, 1: MODEL_INPUT, 2: REPROJECT}  # ReprojectionView
WHOLE_FRAME = {MODEL_INPUT, REPROJECT}  # shown whole, letterboxed, never panned
DEBUG_FRAMES = {CAMERAD: "device", MODEL_INPUT: "model", REPROJECT: "c4"}  # the debug view's frame names
C4_CAMERA = DEVICE_CAMERAS["mici", "os04c10"]

BORDER_COLORS = {
  UIStatus.DISENGAGED: rl.Color(0x12, 0x28, 0x39, 0xFF),  # Blue for disengaged state
  UIStatus.OVERRIDE: rl.Color(0x89, 0x92, 0x8D, 0xFF),  # Gray for override state
  UIStatus.ENGAGED: rl.Color(0x16, 0x7F, 0x40, 0xFF),  # Green for engaged state
}

# The 3X's UI renders on a free 20 fps timer, so its GPU frame drifts through the camera's 50 ms period and, a few minutes in
# every ten, lands on the reprojection stage (3-9 ms after a frame) or the driver-monitoring model (40-10 ms), stalling them
# by up to 6 and 10 ms. The road view keeps the end of its render this far after the newest frame instead: the quiet gap.
UI_PHASE_TARGET_NS = 22_000_000
UI_PHASE_TOLERANCE_NS = 2_000_000
FRAME_PERIOD_NS = 50_000_000
UI_PHASE_STEP_FPS = range(10, 31)  # one-frame rates the phase hold may pad a frame to: 33..100 ms
WIDE_CAM_MAX_SPEED = 10.0  # m/s (22 mph)
ROAD_CAM_MIN_SPEED = 15.0  # m/s (34 mph)
INF_POINT = np.array([1000.0, 0.0, 0.0])


class AugmentedRoadView(CameraView):
  def __init__(self, stream_type: VisionStreamType = VisionStreamType.VISION_STREAM_NARROW_ROAD):
    super().__init__(CAMERAD, stream_type)
    self._debug_frame = "device_narrow"; self._debug_transform: np.ndarray | None = None  # what is on screen, for the debug view
    self._set_placeholder_color(BORDER_COLORS[UIStatus.DISENGAGED])

    self.device_camera: DeviceCameraConfig | None = None
    self.view_from_calib = view_frame_from_device_frame.copy()
    self.view_from_wide_calib = view_frame_from_device_frame.copy()

    self._phase_fps = 0
    self._matrix_cache_key = (0, 0.0, 0.0, stream_type, CAMERAD)
    self._cached_matrix: np.ndarray | None = None
    self._content_rect = rl.Rectangle()

    self.model_renderer = ModelRenderer()
    self._hud_renderer = HudRenderer()
    self.alert_renderer = AlertRenderer()
    self.driver_state_renderer = DriverStateRenderer()
    self._reproject_debug = ReprojectDebug()

  def hide_event(self):
    super().hide_event()
    self._restore_fps()

  def _render(self, rect):
    self._restore_fps()
    # Only render when system is started to avoid invalid data access
    if not ui_state.started:
      return

    self._switch_stream_if_needed(ui_state.sm)

    # Update calibration before rendering
    self._update_calibration()

    # Create inner content area with border padding
    self._content_rect = rl.Rectangle(
      rect.x + UI_BORDER_SIZE,
      rect.y + UI_BORDER_SIZE,
      rect.width - 2 * UI_BORDER_SIZE,
      rect.height - 2 * UI_BORDER_SIZE,
    )

    # Enable scissor mode to clip all rendering within content rectangle boundaries
    # This creates a rendering viewport that prevents graphics from drawing outside the border
    rl.begin_scissor_mode(
      int(self._content_rect.x),
      int(self._content_rect.y),
      int(self._content_rect.width),
      int(self._content_rect.height)
    )

    # Render the base camera view
    super()._render(self._content_rect)

    # Draw all UI overlays
    self.model_renderer.render(self._content_rect)
    self._hud_renderer.render(self._content_rect)
    self._reproject_debug.render(self._content_rect, self._debug_frame, self._debug_transform)
    self.alert_renderer.render(self._content_rect)
    self.driver_state_renderer.render(self._content_rect)

    # Custom UI extension point - add custom overlays here
    # Use self._content_rect for positioning within camera bounds

    # End clipping region
    rl.end_scissor_mode()

    # Draw colored border based on driving state
    self._draw_border(rect)
    self._hold_ui_phase()

  def _restore_fps(self):
    if self._phase_fps:
      rl.set_target_fps(gui_app.target_fps)
      self._phase_fps = 0

  def _hold_ui_phase(self):
    eof = ui_state.sm['narrowRoadCameraState'].timestampEof  # the stage runs off the narrow frame, whichever camera is shown
    if not COMMA_HARDWARE or not eof:
      return
    err = (UI_PHASE_TARGET_NS - (time.clock_gettime_ns(time.CLOCK_BOOTTIME) - eof)) % FRAME_PERIOD_NS
    if err > FRAME_PERIOD_NS // 2:
      err -= FRAME_PERIOD_NS
    if abs(err) > UI_PHASE_TOLERANCE_NS:
      # raylib pads every frame to its target rate, so one frame at another rate moves its timer by the difference.
      # sleeping here instead lands late by whatever renders after this view, and a late frame costs a full extra period
      self._phase_fps = min(UI_PHASE_STEP_FPS, key=lambda fps: abs(1e9 / fps - FRAME_PERIOD_NS - err))
      rl.set_target_fps(self._phase_fps)

  def _handle_mouse_press(self, _):
    if not self._hud_renderer.user_interacting() and self._click_callback is not None:
      self._click_callback()

  def _handle_mouse_release(self, _):
    # We only call click callback on press if not interacting with HUD
    pass

  def _draw_border(self, rect: rl.Rectangle):
    rl.draw_rectangle_lines_ex(rect, UI_BORDER_SIZE, rl.BLACK)
    border_roundness = 0.12
    border_color = BORDER_COLORS.get(ui_state.status, BORDER_COLORS[UIStatus.DISENGAGED])
    border_rect = rl.Rectangle(rect.x + UI_BORDER_SIZE, rect.y + UI_BORDER_SIZE,
                               rect.width - 2 * UI_BORDER_SIZE, rect.height - 2 * UI_BORDER_SIZE)
    rl.draw_rectangle_rounded_lines_ex(border_rect, border_roundness, 10, UI_BORDER_SIZE, border_color)

  def _switch_stream_if_needed(self, sm):
    stage = sm.seen['reprojectState']  # no eGPU, no stage: the setting is ignored rather than showing a stream that never comes
    name = VIEW_SERVERS.get(ui_state.reproject_view, CAMERAD) if stage else CAMERAD
    if name != self._name:  # change server on the narrow stream; the wide choice follows once its streams are known
      self.switch_stream(NARROW_ROAD_CAM, name)
      return
    if ui_state.reproject_camera:
      target = WIDE_CAM if ui_state.reproject_camera == 2 and WIDE_CAM in self.available_streams else NARROW_ROAD_CAM
    elif sm['selfdriveState'].experimentalMode and WIDE_CAM in self.available_streams:
      v_ego = sm['carState'].vEgo
      if v_ego < WIDE_CAM_MAX_SPEED:
        target = WIDE_CAM
      elif v_ego > ROAD_CAM_MIN_SPEED:
        target = NARROW_ROAD_CAM
      else:
        # Hysteresis zone - keep current stream
        target = self.stream_type
    else:
      target = NARROW_ROAD_CAM

    if self.stream_type != target:
      self.switch_stream(target)

  def _update_calibration(self):
    # Update device camera if not already set
    sm = ui_state.sm
    if not self.device_camera and sm.seen['narrowRoadCameraState'] and sm.seen['deviceState']:
      self.device_camera = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['narrowRoadCameraState'].sensor))]

    # Check if camera calibration data is available and valid
    if not (sm.updated["extrinsicsCalibration"] and sm.valid['extrinsicsCalibration']):
      return

    calib = sm['extrinsicsCalibration']
    if len(calib.rpyCalib) != 3 or calib.calStatus != CALIBRATED:
      return

    # Update view_from_calib matrix
    device_from_calib = rot_from_euler(calib.rpyCalib)
    self.view_from_calib = view_frame_from_device_frame @ device_from_calib

    # Update wide calibration if available
    if hasattr(calib, 'wideFromDeviceEuler') and len(calib.wideFromDeviceEuler) == 3:
      wide_from_device = rot_from_euler(calib.wideFromDeviceEuler)
      self.view_from_wide_calib = view_frame_from_device_frame @ wide_from_device @ device_from_calib

  def _calc_frame_matrix(self, rect: rl.Rectangle) -> np.ndarray:
    # Check if we can use cached matrix
    cache_key = (
      ui_state.sm.recv_frame['extrinsicsCalibration'],
      self._content_rect.width,
      self._content_rect.height,
      self.stream_type,
      self._name,
    )
    if cache_key == self._matrix_cache_key and self._cached_matrix is not None:
      return self._cached_matrix

    # Get camera configuration
    device_camera = self.device_camera or DEFAULT_DEVICE_CAMERA
    is_wide_camera = self.stream_type == WIDE_CAM
    if self._name == MODEL_INPUT:  # the model's frames: calibrated, so the overlays need no calibration rotation
      intrinsic = sbigmodel_intrinsics if is_wide_camera else medmodel_intrinsics
      calibration = view_frame_from_device_frame
      zoom = 0.0
    elif self._name == REPROJECT:  # the comma 4 frames are rendered about the device axes
      intrinsic = C4_CAMERA.wide_road.intrinsics if is_wide_camera else C4_CAMERA.narrow_road.intrinsics
      calibration = self.view_from_calib
      zoom = 0.0
    else:
      intrinsic = device_camera.wide_road.intrinsics if is_wide_camera else device_camera.narrow_road.intrinsics
      calibration = self.view_from_wide_calib if is_wide_camera else self.view_from_calib
      zoom = 2.0 if is_wide_camera else 1.1

    # Calculate transforms for vanishing point
    calib_transform = intrinsic @ calibration
    kep = calib_transform @ INF_POINT

    # Calculate center points and dimensions
    x, y = self._content_rect.x, self._content_rect.y
    w, h = self._content_rect.width, self._content_rect.height
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    if self._name == MODEL_INPUT:  # the model frames' principal point sits up at the horizon row, not at the frame's centre
      cx, cy = MEDMODEL_INPUT_SIZE[0] / 2, MEDMODEL_INPUT_SIZE[1] / 2

    # Ensure zoom views the whole area
    zoom = max(zoom, w / (2 * cx), h / (2 * cy))
    if self._name in WHOLE_FRAME:
      zoom = min(w / (2 * cx), h / (2 * cy))  # the whole frame, letterboxed: a crop would misread as the model's input

    # Calculate max allowed offsets with margins
    margin = 5
    max_x_offset = max(0.0, cx * zoom - w / 2 - margin)
    max_y_offset = max(0.0, cy * zoom - h / 2 - margin)

    # Calculate and clamp offsets to prevent out-of-bounds issues
    try:
      if self._name in WHOLE_FRAME:  # no vanishing-point pan: the frame is shown whole
        x_offset, y_offset = 0, 0
      elif abs(kep[2]) > 1e-6:
        x_offset = np.clip((kep[0] / kep[2] - cx) * zoom, -max_x_offset, max_x_offset)
        y_offset = np.clip((kep[1] / kep[2] - cy) * zoom, -max_y_offset, max_y_offset)
      else:
        x_offset, y_offset = 0, 0
    except (ZeroDivisionError, OverflowError):
      x_offset, y_offset = 0, 0

    # Cache the computed transformation matrix to avoid recalculations
    self._matrix_cache_key = cache_key
    self._cached_matrix = np.array([
      [zoom * 2 * cx / w, 0, -x_offset / w * 2],
      [0, zoom * 2 * cy / h, -y_offset / h * 2],
      [0, 0, 1.0]
    ])

    video_transform = np.array([
      [zoom, 0.0, (w / 2 + x - x_offset) - (cx * zoom)],
      [0.0, zoom, (h / 2 + y - y_offset) - (cy * zoom)],
      [0.0, 0.0, 1.0]
    ])
    self.model_renderer.set_transform(video_transform @ calib_transform)
    self._debug_frame = f"{DEBUG_FRAMES[self._name]}_{'wide' if is_wide_camera else 'narrow'}"
    self._debug_transform = video_transform

    return self._cached_matrix


if __name__ == "__main__":
  gui_app.init_window("OnRoad Camera View")
  road_camera_view = AugmentedRoadView(NARROW_ROAD_CAM)
  gui_app.push_widget(road_camera_view)
  print("***press space to switch camera view***")
  try:
    for _ in gui_app.render():
      ui_state.update()
      if rl.is_key_released(rl.KeyboardKey.KEY_SPACE):
        if WIDE_CAM in road_camera_view.available_streams:
          stream = NARROW_ROAD_CAM if road_camera_view.stream_type == WIDE_CAM else WIDE_CAM
          road_camera_view.switch_stream(stream)
  finally:
    road_camera_view.close()
