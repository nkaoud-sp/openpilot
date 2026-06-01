#!/usr/bin/env python3
"""
Screen healing utility for openpilot devices.

Image retention ("pixel burn") happens when static UI elements - like the
driver monitoring circle - stay in the same place for a long time. Cycling
the affected pixels through full-range colors and sweeping motion helps even
out their state and reduce the ghosting.

Run it and leave the device on until it finishes. The screen is driven at
full brightness for the duration, then the previous brightness is restored.

  ./screen_heal.py                 # default 30 minute session
  ./screen_heal.py --duration 60   # run for 60 minutes
  ./screen_heal.py --interval 1.5  # 1.5s per solid color in the cycle

Tap the screen / close the window to stop early.
"""
import argparse
import time

import pyray as rl

from openpilot.system.hardware import HARDWARE, PC
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached

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
STATUS_FONT_SIZE = 32


def _wobble(t: float, freq: float) -> float:
  # Cheap triangle wave in [-1, 1], no per-frame trig needed.
  phase = (t * freq) % 1.0
  return 4.0 * abs(phase - 0.5) - 1.0


def _draw_status(remaining_s: float, phase: str) -> None:
  # Draw the status text in a slowly drifting position so the text itself does
  # not contribute to retention. Mid-grey so it reads on any background color.
  mins, secs = divmod(int(remaining_s), 60)
  label = f"healing screen - {phase} - {mins:02d}:{secs:02d} left"
  font = gui_app.font(FontWeight.NORMAL)
  text_w = measure_text_cached(font, label, STATUS_FONT_SIZE).x

  t = time.monotonic()
  x = (gui_app.width - text_w) / 2 * (1.0 + 0.8 * _wobble(t, 0.13))
  y = (gui_app.height - STATUS_FONT_SIZE) / 2 * (1.0 + 0.8 * _wobble(t, 0.19))
  rl.draw_text_ex(font, label, rl.Vector2(x, y), STATUS_FONT_SIZE, 0.0, rl.Color(128, 128, 128, 180))


def run(duration_s: float, interval_s: float) -> None:
  start = time.monotonic()
  end = start + duration_s

  for _ in gui_app.render():
    now = time.monotonic()
    if now >= end:
      break
    remaining = end - now
    elapsed = now - start

    # Alternate between a solid-color cycle and a sweep pass every ~2 cycles so
    # pixels see both steady states and fast transitions.
    cycle_len = interval_s * len(HEAL_COLORS)
    block = int(elapsed // cycle_len)

    if block % 2 == 0:
      # Solid color cycle phase.
      idx = int((elapsed % cycle_len) // interval_s) % len(HEAL_COLORS)
      rl.clear_background(HEAL_COLORS[idx])
      phase = "color cycle"
    else:
      # Sweep phase: black background with a bright bar bouncing across.
      rl.clear_background(rl.BLACK)
      bar_w = max(40, gui_app.width // 8)
      span = gui_app.width + bar_w
      pos = (elapsed * SWEEP_SPEED) % (2 * span)
      x = pos if pos <= span else 2 * span - pos
      rl.draw_rectangle(int(x - bar_w), 0, bar_w, gui_app.height, SWEEP_COLOR)
      phase = "sweep"

    _draw_status(remaining, phase)


def main() -> None:
  parser = argparse.ArgumentParser(description="Heal screen image retention by cycling pixels.")
  parser.add_argument("--duration", type=float, default=30.0, help="session length in minutes (default: 30)")
  parser.add_argument("--interval", type=float, default=2.0, help="seconds per solid color (default: 2.0)")
  parser.add_argument("--no-brightness", action="store_true", help="do not force the screen to full brightness")
  args = parser.parse_args()

  # Force full brightness for the most effective healing, restoring afterwards.
  prev_brightness = None
  if not args.no_brightness and not PC:
    try:
      current = HARDWARE.get_screen_brightness()
      if current and current > 0:
        prev_brightness = current
      HARDWARE.set_screen_brightness(100)
    except Exception:
      prev_brightness = None

  gui_app.init_window("Screen Heal")
  try:
    run(args.duration * 60.0, args.interval)
  finally:
    if prev_brightness is not None:
      try:
        HARDWARE.set_screen_brightness(prev_brightness)
      except Exception:
        pass


if __name__ == "__main__":
  main()
