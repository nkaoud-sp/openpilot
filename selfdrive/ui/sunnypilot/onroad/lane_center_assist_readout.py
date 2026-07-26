"""
On-road readout for Lane Center Assist.

Recomputes the ego-lane-centre offset independently of the controls process
(same convention as speed_assist_readout) so Readout mode is meaningful before
the assist is allowed to steer.
"""
import numpy as np
import pyray as rl

from cereal import log
from openpilot.common.constants import CV
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached

LaneChangeState = log.LaneChangeState

_STRENGTH_GAIN = (0.4, 0.7, 1.0)
_CONF_MIN_PROB = (0.4, 0.6)
_NEAR_X_LO = 5.0
_NEAR_X_HI = 25.0
_LANE_WIDTH_MIN = 2.6
_LANE_WIDTH_MAX = 4.6
_OFFSET_MAX_VALID = 1.2
_OFFSET_DEADZONE = 0.05

_GREEN = rl.Color(0, 200, 90, 255)
_BLUE = rl.Color(60, 200, 255, 255)
_WHITE = rl.Color(255, 255, 255, 255)
_DIM = rl.Color(180, 180, 180, 255)


def _median_y_near(line):
  xs = np.asarray(line.x)
  ys = np.asarray(line.y)
  if xs.size == 0 or ys.size == 0:
    return None
  mask = (xs >= _NEAR_X_LO) & (xs <= _NEAR_X_HI)
  ys_window = ys[mask]
  if ys_window.size == 0:
    ys_window = ys
  return float(np.median(ys_window))


class LaneCenterAssistReadout:
  def __init__(self):
    self._alpha: float = 0.0
    self._cap_font = gui_app.font(FontWeight.SEMI_BOLD)
    self._val_font = gui_app.font(FontWeight.BOLD)

  def _update_alpha(self, visible: bool):
    if visible:
      self._alpha = min(1.0, self._alpha + 0.1)
    else:
      self._alpha = max(0.0, self._alpha - 0.05)

  @staticmethod
  def _offset(model) -> tuple[bool, float]:
    lane_lines = model.laneLines
    probs = model.laneLineProbs
    if len(lane_lines) < 4 or len(probs) < 4:
      return False, 0.0
    min_prob = _CONF_MIN_PROB[int(max(0, min(len(_CONF_MIN_PROB) - 1, ui_state.lane_center_assist_confidence)))]
    if float(probs[1]) < min_prob or float(probs[2]) < min_prob:
      return False, 0.0
    left_y = _median_y_near(lane_lines[1])
    right_y = _median_y_near(lane_lines[2])
    if left_y is None or right_y is None:
      return False, 0.0
    width = right_y - left_y  # right is +y, left is -y (see ldw.py)
    if not (_LANE_WIDTH_MIN <= width <= _LANE_WIDTH_MAX):
      return False, 0.0
    center = (left_y + right_y) / 2.0
    if abs(center) > _OFFSET_MAX_VALID:
      return False, 0.0
    return True, center

  def _status(self, sm) -> tuple[str, rl.Color, float, float]:
    mode = ui_state.lane_center_assist_mode
    if mode <= 0:
      return "OFF", _DIM, 0.0, 0.0

    CS = sm['carState']
    model = sm['modelV2']
    cc = sm['carControl']
    v_ego_kph = CS.vEgo * CV.MS_TO_KPH

    if not cc.enabled or not cc.latActive:
      return "INACTIVE", _DIM, 0.0, 0.0
    if v_ego_kph < ui_state.lane_center_assist_min_kph:
      return "SPEED", _DIM, 0.0, 0.0
    if model.meta.laneChangeState != LaneChangeState.off:
      return "LANE CHG", _DIM, 0.0, 0.0

    ok, offset = self._offset(model)
    if not ok:
      return "NO LANE", _DIM, 0.0, 0.0
    if abs(offset) < _OFFSET_DEADZONE:
      return "CENTERED", _BLUE, offset, 0.0

    strength = int(max(0, min(len(_STRENGTH_GAIN) - 1, ui_state.lane_center_assist_strength)))
    max_accel = ui_state.lane_center_assist_max_accel / 100.0
    lat_accel = float(np.clip(_STRENGTH_GAIN[strength] * offset, -max_accel, max_accel))
    return ("READY", _BLUE, offset, lat_accel) if mode == 1 else ("ACTIVE", _GREEN, offset, lat_accel)

  def draw(self, sm, rect: rl.Rectangle):
    visible = ui_state.lane_center_assist_mode > 0
    self._update_alpha(visible)
    if self._alpha <= 0.0 or not visible:
      return

    state, state_c, offset, lat_accel = self._status(sm)

    cells = [
      ("LN CTR", state, state_c),
      ("OFF cm", f"{offset * 100:+.0f}", _WHITE),
      ("ACC", f"{lat_accel:+.2f}", _GREEN if state == "ACTIVE" else _DIM),
    ]
    self._render(rect, cells)

  def _render(self, rect: rl.Rectangle, cells):
    a = self._alpha
    cap_size = 22
    val_size = 36
    pad = 18
    cell_gap = 22
    cap_val_gap = 4

    def fade(c: rl.Color) -> rl.Color:
      return rl.Color(c.r, c.g, c.b, int(255 * a))

    cell_w = 0.0
    for cap, val, _c in cells:
      cw = max(measure_text_cached(self._cap_font, cap, cap_size, 0).x,
               measure_text_cached(self._val_font, val, val_size, 0).x)
      cell_w = max(cell_w, cw)

    row_h = cap_size + cap_val_gap + val_size
    content_w = len(cells) * cell_w + (len(cells) - 1) * cell_gap
    panel_w = pad + content_w + pad
    panel_h = pad + row_h + pad

    x = rect.x + (rect.width - panel_w) / 2
    y = rect.y + rect.height * 0.50

    rl.draw_rectangle_rounded(rl.Rectangle(x, y, panel_w, panel_h), 0.18, 10, rl.Color(0, 0, 0, int(120 * a)))

    row_y = y + pad
    for c, (cap, val, color) in enumerate(cells):
      cx = x + pad + c * (cell_w + cell_gap)
      cap_w = measure_text_cached(self._cap_font, cap, cap_size, 0).x
      rl.draw_text_ex(self._cap_font, cap, rl.Vector2(int(cx + (cell_w - cap_w) / 2), int(row_y)),
                      cap_size, 0, fade(_DIM))
      val_w = measure_text_cached(self._val_font, val, val_size, 0).x
      rl.draw_text_ex(self._val_font, val, rl.Vector2(int(cx + (cell_w - val_w) / 2), int(row_y + cap_size + cap_val_gap)),
                      val_size, 0, fade(color))
