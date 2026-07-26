"""
On-road readout for Experimental Speed Assist.
"""
import pyray as rl
from openpilot.common.constants import CV
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached

_MAX_BOOSTS = [0.15, 0.25, 0.35]
_FULL_GAP_KPH = 30.0
_MODEL_PLAN_GAP_MAX = 0.10
_GENTLE_ACCEL_MAX = 0.20
_DECEL_BLOCK = -0.15
_MODEL_LEAD_PROB_MAX = 0.35
_BRAKE_PROB_MAX = 0.06

_GREEN = rl.Color(0, 200, 90, 255)
_BLUE = rl.Color(60, 200, 255, 255)
_YELLOW = rl.Color(255, 180, 0, 255)
_WHITE = rl.Color(255, 255, 255, 255)
_DIM = rl.Color(180, 180, 180, 255)


class SpeedAssistReadout:
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
  def _list_value(values, idx: int, default: float = 0.0) -> float:
    return float(values[idx]) if len(values) > idx else default

  @staticmethod
  def _model_lead_prob(model_msg) -> float:
    if len(model_msg.leadsV3) == 0:
      return 0.0
    return max(float(lead.prob) for lead in model_msg.leadsV3)

  def _status(self, sm) -> tuple[str, rl.Color, float, float]:
    mode = ui_state.experimental_speed_assist_mode
    if mode <= 0:
      return "OFF", _DIM, 0.0, 0.0

    CS = sm['carState']
    LP = sm['longitudinalPlan']
    model = sm['modelV2']
    radar = sm['radarState']
    v_ego_kph = CS.vEgo * CV.MS_TO_KPH
    speed_gap_kph = float(CS.vCruise - v_ego_kph)

    if not sm['selfdriveState'].experimentalMode:
      return "NO EXP", _DIM, speed_gap_kph, 0.0
    if not sm['carControl'].enabled:
      return "DISABLED", _DIM, speed_gap_kph, 0.0
    if str(LP.longitudinalPlanSource).split(".")[-1] != "e2e":
      return "NOT E2E", _DIM, speed_gap_kph, 0.0
    if CS.gasPressed or CS.brakePressed:
      return "DRIVER", _DIM, speed_gap_kph, 0.0
    if CS.vCruise == V_CRUISE_UNSET or CS.vCruise >= V_CRUISE_UNSET:
      return "NO CRUISE", _DIM, speed_gap_kph, 0.0

    min_kph = min(ui_state.experimental_speed_assist_min_kph, ui_state.experimental_speed_assist_max_kph)
    max_kph = max(ui_state.experimental_speed_assist_min_kph, ui_state.experimental_speed_assist_max_kph)
    if v_ego_kph < min_kph or v_ego_kph > max_kph:
      return "SPEED", _DIM, speed_gap_kph, 0.0
    if speed_gap_kph < ui_state.experimental_speed_assist_start_gap_kph:
      return "GAP", _DIM, speed_gap_kph, 0.0
    if ui_state.experimental_speed_assist_lead_mode == 0 and (radar.leadOne.status or self._model_lead_prob(model) > _MODEL_LEAD_PROB_MAX):
      return "LEAD", _DIM, speed_gap_kph, 0.0
    if model.action.shouldStop or LP.shouldStop:
      return "STOP", _DIM, speed_gap_kph, 0.0

    preds = model.meta.disengagePredictions
    brake_press = self._list_value(preds.brakePressProbs, 0)
    brake_disengage = self._list_value(preds.brakeDisengageProbs, 0)
    if brake_press > _BRAKE_PROB_MAX or brake_disengage > _BRAKE_PROB_MAX:
      return "BRAKE P", _DIM, speed_gap_kph, 0.0

    model_accel = float(model.action.desiredAcceleration)
    plan_accel = float(LP.aTarget)
    if plan_accel < _DECEL_BLOCK:
      return "DECEL", _DIM, speed_gap_kph, 0.0
    if abs(model_accel - plan_accel) > _MODEL_PLAN_GAP_MAX:
      return ("MPC LOW" if model_accel > plan_accel else "E2E LOW"), _DIM, speed_gap_kph, 0.0
    if model_accel > _GENTLE_ACCEL_MAX or plan_accel > _GENTLE_ACCEL_MAX:
      return "ACCEL", _DIM, speed_gap_kph, 0.0

    strength = int(max(0, min(len(_MAX_BOOSTS) - 1, ui_state.experimental_speed_assist_strength)))
    full_gap_kph = max(float(ui_state.experimental_speed_assist_start_gap_kph) + 1.0, _FULL_GAP_KPH)
    boost = (speed_gap_kph - ui_state.experimental_speed_assist_start_gap_kph) / (full_gap_kph - ui_state.experimental_speed_assist_start_gap_kph)
    boost = max(0.0, min(1.0, boost))
    boost = 0.05 + boost * (_MAX_BOOSTS[strength] - 0.05)
    return ("READY", _BLUE, speed_gap_kph, boost) if mode == 1 else ("ACTIVE", _GREEN, speed_gap_kph, boost)

  def draw(self, sm, rect: rl.Rectangle):
    visible = ui_state.has_longitudinal_control and ui_state.experimental_speed_assist_mode > 0
    self._update_alpha(visible)
    if self._alpha <= 0.0 or not visible:
      return

    state, state_c, speed_gap_kph, boost = self._status(sm)

    cells = [
      ("SPD AST", state, state_c),
      ("GAP kph", f"{speed_gap_kph:.0f}", _WHITE),
      ("BOOST", f"+{boost:.2f}", _GREEN if state == "ACTIVE" else _DIM),
      ("CMD", f"{float(sm['longitudinalPlan'].aTarget):+.2f}", _WHITE),
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
    y = rect.y + rect.height * 0.62

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
