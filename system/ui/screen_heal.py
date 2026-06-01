#!/usr/bin/env python3
"""
Screen healing utility for openpilot devices.

Image retention ("pixel burn") happens when static UI elements - like the
driver monitoring circle - stay in the same place for a long time. Cycling
the affected pixels through full-range colors and sweeping motion helps even
out their state and reduce the ghosting.

This can be launched from the developer settings toggle, or run standalone:

  ./screen_heal.py                 # default 120 minute session
  ./screen_heal.py --duration 60   # run for 60 minutes

It runs at full brightness, then restores the previous brightness, and stops
when the timer expires or the screen is tapped.
"""
import argparse
import time

import pyray as rl

from openpilot.system.hardware import HARDWARE, PC
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

# Default session length in minutes.
DEFAULT_DURATION_MIN = 120.0
# Seconds each solid color is held during the color-cycle phase.
DEFAULT_INTERVAL_S = 2.0
# Ignore taps briefly after launch so the tap that opened it doesn't stop it.
TAP_GRACE_S = 0.75
STATUS_FONT_SIZE = 32

# Primary colors plus white/black exercise every sub-pixel through its full range.
HEAL_COLORS = [
  rl.Color(255, 255, 255, 255),  # white  - drives all sub-pixels hard
  rl.Color(255, 0, 0, 255),      # red
  rl.Color(0, 255, 0, 255),      # green
  rl.Color(0, 0, 255, 255),      # blue
  rl.Color(0, 255, 255, 255),    # cyan
  rl.Color(255, 0, 255, 255),    # magenta
  rl.Color(255, 255, 0, 255),    # yellow
  rl.Color(0, 0, 0, 255),        # black  - lets pixels relax
]

# A bright bar sweeps across the screen so every pixel transitions repeatedly,
# which targets static-image retention better than solid fills alone.
SWEEP_COLOR = rl.Color(255, 255, 255, 255)
SWEEP_SPEED = 600.0  # pixels per second


def _wobble(t: float, freq: float) -> float:
  # Cheap triangle wave in [-1, 1], no per-frame trig needed.
  phase = (t * freq) % 1.0
  return 4.0 * abs(phase - 0.5) - 1.0


class ScreenHeal(Widget):
  def __init__(self, duration_min: float = DEFAULT_DURATION_MIN, interval_s: float = DEFAULT_INTERVAL_S,
               manage_brightness: bool = True, on_finish=None):
    super().__init__()
    self._duration_s = duration_min * 60.0
    self._interval_s = interval_s
    self._manage_brightness = manage_brightness and not PC
    self._on_finish = on_finish
    self._start = 0.0
    self._prev_brightness: int | None = None
    self._finished = False

  def show_event(self):
    super().show_event()
    self._start = time.monotonic()
    self._finished = False
    if self._manage_brightness:
      try:
        current = HARDWARE.get_screen_brightness()
        if current and current > 0:
          self._prev_brightness = current
        HARDWARE.set_screen_brightness(100)
      except Exception:
        self._prev_brightness = None

  def hide_event(self):
    super().hide_event()
    if self._prev_brightness is not None:
      try:
        HARDWARE.set_screen_brightness(self._prev_brightness)
      except Exception:
        pass
      self._prev_brightness = None

  def _finish(self):
    if self._finished:
      return
    self._finished = True
    self.dismiss(self._on_finish)

  def _handle_mouse_release(self, mouse_pos):
    super()._handle_mouse_release(mouse_pos)
    # Tap anywhere to stop, after a short grace period.
    if time.monotonic() - self._start > TAP_GRACE_S:
      self._finish()

  def _draw_status(self, remaining_s: float, phase: str) -> None:
    # Draw the status text in a slowly drifting position so the text itself does
    # not contribute to retention. Mid-grey so it reads on any background color.
    mins, secs = divmod(int(remaining_s), 60)
    label = f"healing screen - {phase} - {mins:02d}:{secs:02d} left - tap to stop"
    font = gui_app.font(FontWeight.NORMAL)
    text_w = measure_text_cached(font, label, STATUS_FONT_SIZE).x

    t = time.monotonic()
    x = (gui_app.width - text_w) / 2 * (1.0 + 0.8 * _wobble(t, 0.13))
    y = (gui_app.height - STATUS_FONT_SIZE) / 2 * (1.0 + 0.8 * _wobble(t, 0.19))
    rl.draw_text_ex(font, label, rl.Vector2(x, y), STATUS_FONT_SIZE, 0.0, rl.Color(128, 128, 128, 180))

  def _render(self, rect: rl.Rectangle):
    elapsed = time.monotonic() - self._start
    if elapsed >= self._duration_s:
      self._finish()
      return

    # Alternate between a solid-color cycle and a sweep pass every ~2 cycles so
    # pixels see both steady states and fast transitions.
    cycle_len = self._interval_s * len(HEAL_COLORS)
    block = int(elapsed // cycle_len)

    if block % 2 == 0:
      idx = int((elapsed % cycle_len) // self._interval_s) % len(HEAL_COLORS)
      rl.clear_background(HEAL_COLORS[idx])
      phase = "color cycle"
    else:
      rl.clear_background(rl.BLACK)
      bar_w = max(40, int(rect.width) // 8)
      span = int(rect.width) + bar_w
      pos = (elapsed * SWEEP_SPEED) % (2 * span)
      x = pos if pos <= span else 2 * span - pos
      rl.draw_rectangle(int(x - bar_w), 0, bar_w, int(rect.height), SWEEP_COLOR)
      phase = "sweep"

    self._draw_status(self._duration_s - elapsed, phase)


def main() -> None:
  parser = argparse.ArgumentParser(description="Heal screen image retention by cycling pixels.")
  parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_MIN, help="session length in minutes (default: 120)")
  parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S, help="seconds per solid color (default: 2.0)")
  parser.add_argument("--no-brightness", action="store_true", help="do not force the screen to full brightness")
  args = parser.parse_args()

  gui_app.init_window("Screen Heal")
  heal = ScreenHeal(duration_min=args.duration, interval_s=args.interval,
                    manage_brightness=not args.no_brightness, on_finish=gui_app.request_close)
  gui_app.push_widget(heal)
  for _ in gui_app.render():
    pass


if __name__ == "__main__":
  main()
