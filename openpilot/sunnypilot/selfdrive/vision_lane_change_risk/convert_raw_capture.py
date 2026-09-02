#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def resolve_raw_path(metadata_path: Path, raw_path: str) -> Path:
  path = Path(raw_path)
  if path.exists():
    return path

  local_path = metadata_path.with_suffix(".y8")
  if local_path.exists():
    return local_path

  sibling_path = metadata_path.parent / Path(raw_path).name
  if sibling_path.exists():
    return sibling_path

  raise FileNotFoundError(f"raw capture not found: {raw_path}")


def convert_capture(metadata_path: Path, output_path: Path | None) -> Path:
  with metadata_path.open() as f:
    metadata = json.load(f)

  width = int(metadata["width"])
  height = int(metadata["height"])
  fps = float(metadata["fps"])
  raw_path = resolve_raw_path(metadata_path, metadata["raw_path"])
  output_path = output_path or raw_path.with_suffix(".mp4")

  frame_size = width * height
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), True)
  if not writer.isOpened():
    raise RuntimeError(f"failed to open video writer for {output_path}")

  frames_written = 0
  with raw_path.open("rb") as f:
    while True:
      data = f.read(frame_size)
      if not data:
        break
      if len(data) != frame_size:
        print(f"stopping at partial frame of {len(data)} bytes")
        break
      gray = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
      writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
      frames_written += 1

  writer.release()
  print(f"wrote {frames_written} frames to {output_path}")
  return output_path


def main() -> None:
  parser = argparse.ArgumentParser(description="Convert a VLCR raw .y8 capture to MP4.")
  parser.add_argument("metadata", type=Path, help="Path to vlcr_stream_*.json")
  parser.add_argument("-o", "--output", type=Path, help="Output MP4 path")
  args = parser.parse_args()
  convert_capture(args.metadata, args.output)


if __name__ == "__main__":
  main()
