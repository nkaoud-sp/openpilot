from __future__ import annotations

import time

import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
# Reuse the exact gate thresholds so the reason pill can't drift from the
# actual desire_helper lane-change gate.
from openpilot.selfdrive.controls.lib.desire_helper import (
  VISUAL_CONF_BLOCK_THRESHOLD, VISUAL_STALE_TIME,
)
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached


PARAM_REFRESH_S = 0.5
FLASH_PERIOD_S = 0.7
OVERLAY_SCREEN_FRACTION = 0.5
MIN_OVERLAY_SIZE = 280
TURN_OVERLAY_DISTANCE_M = 150.0
# Lane-change cue window is wider than the turn window: highway exits/forks
# want the lane move started well before a surface-street turn cue.
ADVISORY_OVERLAY_DISTANCE_M = 500.0
OVERLAY_TINT_MIN_ALPHA = 96
OVERLAY_TINT_MAX_ALPHA = 255

# Reason pill (shown under the flashing arrow only while a nav lane change is
# actually blocked by BSM / camera / speed).
PILL_FONT_SIZE = 40
PILL_BLOCK_COLOR = rl.Color(200, 45, 45, 225)
PILL_TEXT_COLOR = rl.Color(255, 255, 255, 255)


class NavTurnArrow:
  def __init__(self) -> None:
    self._enabled = False
    self._show_banner = False
    self._next_param_check = 0.0
    self._textures = {
      "turn_right": gui_app.texture("../../sunnypilot/selfdrive/assets/nav_turn_arrows/arrow_ct_r.png"),
      "turn_left": gui_app.texture("../../sunnypilot/selfdrive/assets/nav_turn_arrows/arrow_ct_r.png", flip_x=True),
      "roundabout_right": gui_app.texture("../../sunnypilot/selfdrive/assets/nav_turn_arrows/arrow_ct_rnd.png"),
      "roundabout_left": gui_app.texture("../../sunnypilot/selfdrive/assets/nav_turn_arrows/arrow_ct_rnd.png", flip_x=True),
      "lane_change_right": gui_app.texture("../../sunnypilot/selfdrive/assets/nav_turn_arrows/arrow_ct_sr.png"),
      "lane_change_left": gui_app.texture("../../sunnypilot/selfdrive/assets/nav_turn_arrows/arrow_ct_sr.png", flip_x=True),
      "straight_right": gui_app.texture("../../sunnypilot/selfdrive/assets/nav_turn_arrows/arrow_ct_st.png"),
      "straight_left": gui_app.texture("../../sunnypilot/selfdrive/assets/nav_turn_arrows/arrow_ct_st.png", flip_x=True),
      "uturn_right": gui_app.texture("../../sunnypilot/selfdrive/assets/nav_turn_arrows/arrow_ct_u.png"),
      "uturn_left": gui_app.texture("../../sunnypilot/selfdrive/assets/nav_turn_arrows/arrow_ct_u.png", flip_x=True),
      "sharp_right": gui_app.texture("../../sunnypilot/selfdrive/assets/nav_turn_arrows/arrow_ct_v.png"),
      "sharp_left": gui_app.texture("../../sunnypilot/selfdrive/assets/nav_turn_arrows/arrow_ct_v.png", flip_x=True),
    }
    self._pill_font = gui_app.font(FontWeight.BOLD)

  def _refresh_params(self) -> None:
    now = time.monotonic()
    if now < self._next_param_check:
      return
    self._next_param_check = now + PARAM_REFRESH_S
    self._enabled = ui_state.params.get_bool("NkaoudNavEnabled")
    self._show_banner = ui_state.params.get_bool("NkaoudNavShowBanner")

  @staticmethod
  def _normalize(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "").replace("_", "")

  @staticmethod
  def _side_from_modifier(modifier: str) -> str | None:
    """Left/right for a normalized maneuver modifier, or None when it carries
    no side (straight / arrive / depart / unknown)."""
    if modifier in ("left", "sharpleft", "slightleft", "uturn"):
      return "left"
    if modifier in ("right", "sharpright", "slightright"):
      return "right"
    return None

  def _select_key(self, nav_sp, inst, dist_to_maneuver: float) -> str | None:
    desire = self._normalize(nav_sp.recommendedDesire)
    maneuver_type = self._normalize(inst.maneuverType or nav_sp.maneuverType)
    modifier = self._normalize(inst.maneuverModifier or nav_sp.maneuverModifier)

    # Turn cues are driven by the upcoming maneuver itself -- the same
    # navInstruction data the maneuver banner uses -- so the arrow shows on a
    # normal drive regardless of the steering toggles.
    turn_key = self._turn_key(desire, maneuver_type, modifier, dist_to_maneuver)
    if turn_key is not None:
      return turn_key

    # Lane-change cue: navd's advisoryLaneChange is pure display intent, never
    # gated by NkaoudNavControlSteer / AutoLaneChangeTimer, so the flashing
    # arrow appears even when nav isn't allowed to make the move itself. The
    # turn arrow keeps precedence inside its own window.
    advisory = self._normalize(nav_sp.advisoryLaneChange)
    if advisory in ("left", "right") and dist_to_maneuver <= ADVISORY_OVERLAY_DISTANCE_M:
      return f"lane_change_{advisory}"
    return None

  def _turn_key(self, desire: str, maneuver_type: str, modifier: str, dist_to_maneuver: float) -> str | None:
    if dist_to_maneuver > TURN_OVERLAY_DISTANCE_M:
      return None

    # Prefer the steering desire's side when present, else the maneuver modifier.
    if desire in ("turnleft", "turnright"):
      side = "left" if desire.endswith("left") else "right"
    else:
      side = self._side_from_modifier(modifier)
    if side is None:
      return None

    if "roundabout" in maneuver_type or "rotary" in maneuver_type:
      return f"roundabout_{side}"
    if "uturn" in (modifier, maneuver_type):
      return f"uturn_{side}"
    if modifier in ("sharpleft", "sharpright"):
      return f"sharp_{side}"
    if modifier in ("slightleft", "slightright"):
      return f"lane_change_{side}"
    if modifier == "straight" or "continue" in maneuver_type:
      return f"straight_{side}"
    return f"turn_{side}"

  @staticmethod
  def _flash_alpha() -> int:
    phase = time.monotonic() % FLASH_PERIOD_S
    if phase < FLASH_PERIOD_S / 2:
      return OVERLAY_TINT_MAX_ALPHA
    return OVERLAY_TINT_MIN_ALPHA

  @staticmethod
  def _visual_side_max_prob(side: str) -> float | None:
    """Worst (highest) car-probability the visual detector reports on `side`,
    or None when there's no fresh per-side signal. Mirrors the aggregation in
    desire_helper._visual_side_clear (probability only, ignoring block flags)."""
    sm = ui_state.sm
    if sm.recv_frame.get("visualVehicleDetectorStateSP", 0) <= 0:
      return None
    vs = sm["visualVehicleDetectorStateSP"]
    if (time.monotonic() - float(vs.monotonicTime)) > VISUAL_STALE_TIME:
      return None
    worst = None
    for zones in (vs.classifier.zones, vs.wideZones, vs.driverZones):
      for z in zones:
        if str(z.name) == side and bool(z.hasProbability):
          p = float(z.probability)
          worst = p if worst is None else max(worst, p)
    return worst

  def _lane_change_block_reason(self, side: str) -> str | None:
    """Short reason a wanted nav lane change toward `side` is currently blocked,
    or None when the side is clear (the keep* bias is free to proceed). Mirrors
    the desire_helper keep* gate: blind spot, or visual car above threshold."""
    sm = ui_state.sm
    lcs = self._normalize(sm["modelV2"].meta.laneChangeState)
    if lcs in ("lanechangestarting", "lanechangefinishing"):
      return None  # a state-machine change is under way, not blocked
    cs = sm["carState"]
    if (cs.leftBlindspot if side == "left" else cs.rightBlindspot):
      return "Blind spot"
    vp = self._visual_side_max_prob(side)
    if vp is not None and vp >= VISUAL_CONF_BLOCK_THRESHOLD:
      return f"Camera {vp * 100:.0f}%"
    return None

  def _draw_pill(self, center_x: float, top_y: float, text: str, bg: rl.Color) -> None:
    pad_x, pad_y = 30.0, 14.0
    text_w = measure_text_cached(self._pill_font, text, PILL_FONT_SIZE, 0).x
    w = text_w + 2 * pad_x
    h = PILL_FONT_SIZE + 2 * pad_y
    x = center_x - w / 2
    rl.draw_rectangle_rounded(rl.Rectangle(x, top_y, w, h), 0.5, 16, bg)
    rl.draw_text_ex(self._pill_font, text, rl.Vector2(int(x + pad_x), int(top_y + pad_y)),
                    PILL_FONT_SIZE, 0, PILL_TEXT_COLOR)

  def render(self, rect: rl.Rectangle) -> None:
    self._refresh_params()
    if not (self._enabled and self._show_banner):
      return

    sm = ui_state.sm
    nav_sp = sm["nkaoudNavigationSP"]
    if not nav_sp.active:
      return

    inst = sm["navInstruction"]
    dist_to_maneuver = float(inst.maneuverDistance or nav_sp.distanceToManeuver)
    if dist_to_maneuver <= 0.0:
      return

    texture_key = self._select_key(nav_sp, inst, dist_to_maneuver)
    if texture_key is None:
      return

    texture = self._textures[texture_key]
    target_size = max(MIN_OVERLAY_SIZE, min(rect.width, rect.height) * OVERLAY_SCREEN_FRACTION)
    scale = target_size / max(texture.width, texture.height)
    draw_w = texture.width * scale
    draw_h = texture.height * scale
    pos_x = rect.x + (rect.width - draw_w) / 2
    pos_y = rect.y + (rect.height - draw_h) / 2
    tint = rl.Color(255, 255, 255, self._flash_alpha())
    rl.draw_texture_ex(texture, rl.Vector2(pos_x, pos_y), 0.0, scale, tint)

    # nkaoud_nav: reason pill under the arrow while the route wants a lane
    # move and that side is currently blocked / unsafe.
    advisory = self._normalize(nav_sp.advisoryLaneChange)
    if advisory in ("left", "right"):
      reason = self._lane_change_block_reason(advisory)
      if reason is not None:
        self._draw_pill(rect.x + rect.width / 2, pos_y + draw_h + 24, reason, PILL_BLOCK_COLOR)
