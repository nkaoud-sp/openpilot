from __future__ import annotations

import time

import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app


PARAM_REFRESH_S = 0.5
FLASH_PERIOD_S = 0.7
OVERLAY_SCREEN_FRACTION = 0.5
MIN_OVERLAY_SIZE = 280
TURN_OVERLAY_DISTANCE_M = 150.0
LANE_CHANGE_OVERLAY_DISTANCE_M = 180.0
LANE_PREP_OVERLAY_DISTANCE_M = 100.0
OVERLAY_TINT_MIN_ALPHA = 96
OVERLAY_TINT_MAX_ALPHA = 255


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

    # Lane-change / lane-prep cues have no maneuver-modifier equivalent, so they
    # can only come from the steering desire (which navd populates when
    # NkaoudNavControlSteer is on).
    if desire in ("lanechangeleft", "lanechangeright"):
      if dist_to_maneuver <= LANE_CHANGE_OVERLAY_DISTANCE_M:
        return "lane_change_left" if desire.endswith("left") else "lane_change_right"
      return None

    if desire in ("keepleft", "keepright"):
      if dist_to_maneuver <= LANE_PREP_OVERLAY_DISTANCE_M:
        return "lane_change_left" if desire.endswith("left") else "lane_change_right"
      return None

    # Turn cues are driven by the upcoming maneuver itself -- the same
    # navInstruction data the maneuver banner uses -- so the arrow shows on a
    # normal drive. Previously this required recommendedDesire to be
    # turnLeft/turnRight, but navd only emits that when NkaoudNavControlSteer is
    # enabled; with steering control off (the default) the arrow never appeared
    # even though the banner did.
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
