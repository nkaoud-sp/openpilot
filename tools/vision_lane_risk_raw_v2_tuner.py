#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from openpilot.sunnypilot.selfdrive.vision_lane_change_risk.common_frame_tracker import (  # noqa: E402
  RAW_STRIP_PANEL_W,
  RAW_V2_LEFT_PANEL_CENTER,
  RAW_V2_LEFT_ROTATION_DEG,
  RAW_V2_RIGHT_PANEL_CENTER,
  RAW_V2_RIGHT_ROTATION_DEG,
  _paste_rotated_panel,
  debug_frame_rgb,
)


def parse_center(value: str) -> tuple[int, int]:
  parts = value.replace("x", ",").split(",")
  if len(parts) != 2:
    raise argparse.ArgumentTypeError("center must look like 407,425")
  try:
    return int(parts[0]), int(parts[1])
  except ValueError as exc:
    raise argparse.ArgumentTypeError("center must use integer pixels, like 407,425") from exc


def values_block(
  left_angle: float,
  right_angle: float,
  left_center: tuple[int, int],
  right_center: tuple[int, int],
) -> str:
  return "\n".join((
    f"RAW_V2_LEFT_PANEL_CENTER = {left_center}",
    f"RAW_V2_RIGHT_PANEL_CENTER = {right_center}",
    f"RAW_V2_LEFT_ROTATION_DEG = {left_angle:.1f}",
    f"RAW_V2_RIGHT_ROTATION_DEG = {right_angle:.1f}",
  ))


def render_raw_v2(
  raw_strip: np.ndarray,
  left_angle: float,
  right_angle: float,
  left_center: tuple[int, int],
  right_center: tuple[int, int],
) -> np.ndarray:
  raw_rgb = debug_frame_rgb(raw_strip, False, False, 0.0, 0.0)
  left_dm = raw_rgb[:, :RAW_STRIP_PANEL_W]
  front = raw_rgb[:, RAW_STRIP_PANEL_W:RAW_STRIP_PANEL_W * 3]
  right_dm = raw_rgb[:, RAW_STRIP_PANEL_W * 3:]

  canvas = np.full(raw_rgb.shape, 255, dtype=np.uint8)
  _paste_rotated_panel(canvas, right_dm, left_center[0], left_center[1], left_angle)
  _paste_rotated_panel(canvas, left_dm, right_center[0], right_center[1], right_angle)
  canvas[:, RAW_STRIP_PANEL_W:RAW_STRIP_PANEL_W * 3] = front
  return canvas


def load_raw(path: Path) -> np.ndarray:
  return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def write_single(args: argparse.Namespace) -> Path:
  raw = load_raw(args.raw)
  image = render_raw_v2(raw, args.left_angle, args.right_angle, args.left_center, args.right_center)
  output = args.output or args.raw.with_name(f"{args.raw.stem}_tuned.png")
  Image.fromarray(image).save(output)
  return output


def write_contact_sheet(args: argparse.Namespace) -> Path:
  raw = load_raw(args.raw)
  base = (
    args.left_angle,
    args.right_angle,
    args.left_center,
    args.right_center,
  )
  cx = args.center_step
  cy = args.center_step
  angle = args.angle_step
  variants = (
    ("A current", base[0], base[1], base[2], base[3]),
    ("B less angle", base[0] + angle, base[1] - angle, base[2], base[3]),
    ("C more angle", base[0] - angle, base[1] + angle, base[2], base[3]),
    ("D outward", base[0], base[1], (base[2][0] - cx, base[2][1]), (base[3][0] + cx, base[3][1])),
    ("E inward", base[0], base[1], (base[2][0] + cx, base[2][1]), (base[3][0] - cx, base[3][1])),
    ("F lower", base[0], base[1], (base[2][0], base[2][1] + cy), (base[3][0], base[3][1] + cy)),
    ("G higher", base[0], base[1], (base[2][0], base[2][1] - cy), (base[3][0], base[3][1] - cy)),
    ("H less outward", base[0] + angle, base[1] - angle, (base[2][0] - cx, base[2][1]), (base[3][0] + cx, base[3][1])),
    ("I less inward", base[0] + angle, base[1] - angle, (base[2][0] + cx, base[2][1]), (base[3][0] - cx, base[3][1])),
  )

  output_dir = args.output_dir or args.raw.with_name(f"{args.raw.stem}_tuning")
  output_dir.mkdir(parents=True, exist_ok=True)

  tiles: list[Image.Image] = []
  for name, left_angle, right_angle, left_center, right_center in variants:
    image = render_raw_v2(raw, left_angle, right_angle, left_center, right_center)
    image_path = output_dir / f"{name[0]}_{args.raw.stem}_tuned.png"
    Image.fromarray(image).save(image_path)

    tile = Image.new("RGB", (512, 158), "white")
    tile.paste(Image.fromarray(image).resize((512, 128)), (0, 30))
    draw = ImageDraw.Draw(tile)
    label = f"{name}: L{left_angle:.1f}/R{right_angle:.1f} LC{left_center} RC{right_center}"
    draw.text((8, 8), label, fill=(0, 0, 0))
    tiles.append(tile)

  sheet = Image.new("RGB", (512 * 3, 158 * 3), "white")
  for i, tile in enumerate(tiles):
    sheet.paste(tile, ((i % 3) * 512, (i // 3) * 158))

  sheet_path = output_dir / f"{args.raw.stem}_contact_sheet.png"
  sheet.save(sheet_path)

  values_path = output_dir / "base_values.txt"
  values_path.write_text(values_block(args.left_angle, args.right_angle, args.left_center, args.right_center) + "\n")
  return sheet_path


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Tune the vision lane risk raw_v2 debug PNG transform from a *_raw.png strip.",
  )
  parser.add_argument("raw", type=Path, help="Input raw strip PNG, for example C:/commaai/tmp/vid/372/..._raw.png")
  parser.add_argument("--left-angle", type=float, default=RAW_V2_LEFT_ROTATION_DEG)
  parser.add_argument("--right-angle", type=float, default=RAW_V2_RIGHT_ROTATION_DEG)
  parser.add_argument("--left-center", type=parse_center, default=RAW_V2_LEFT_PANEL_CENTER)
  parser.add_argument("--right-center", type=parse_center, default=RAW_V2_RIGHT_PANEL_CENTER)
  parser.add_argument("--output", type=Path, help="Single-preview output PNG path")
  parser.add_argument("--sheet", action="store_true", help="Write a 3x3 contact sheet of nearby tuning variants")
  parser.add_argument("--output-dir", type=Path, help="Contact-sheet output directory")
  parser.add_argument("--angle-step", type=float, default=5.0, help="Angle delta used by --sheet")
  parser.add_argument("--center-step", type=int, default=40, help="Pixel center delta used by --sheet")
  return parser


def main() -> None:
  args = build_parser().parse_args()
  args.raw = args.raw.expanduser().resolve()
  if args.output is not None:
    args.output = args.output.expanduser().resolve()
  if args.output_dir is not None:
    args.output_dir = args.output_dir.expanduser().resolve()

  if args.sheet:
    output = write_contact_sheet(args)
    print(f"Wrote contact sheet: {output}")
  else:
    output = write_single(args)
    print(f"Wrote preview: {output}")

  print()
  print("Current base values:")
  print(values_block(args.left_angle, args.right_angle, args.left_center, args.right_center))


if __name__ == "__main__":
  main()
