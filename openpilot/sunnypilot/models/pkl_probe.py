"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Report what a downloaded driving pkl actually contains, for when a model fails to load.

The inspection reads only the pickle opcode header and then walks the out-of-band buffer
records, so it never unpickles and cannot be taken down by a tinygrad version mismatch.
The load attempt at the end is the part that reproduces the real failure, so this is meant
to run as its own process - see the Model Pkl Diagnostics button in the tweaks menu.
"""

import argparse
import glob
import os
import struct
import subprocess
import time
import traceback

from openpilot.common.basedir import BASEDIR
from openpilot.common.file_chunker import get_existing_chunks, open_file_chunked
from openpilot.selfdrive.modeld.helpers import MODELS_DIR, load_oob

# Params and the model catalog are only needed to resolve the selected bundle, so they are
# imported where they are used: the inspection half stays importable without them.

READ_BLOCK = 4 << 20
# no single tinygrad buffer is a terabyte, so a length this large means we are not reading lengths
MAX_RECORD = 1 << 40
MAX_ERROR_CHARS = 240
# the device upstream's loader names for the eGPU; the pkl's buffers are bound to it
CHESTNUT_DEV = 'USB+AMD:LLVM'

SCHEMA_KEYS = ('run', 'run_model', 'run_policy', 'input_specs', 'output_specs',
               'metadata', 'input_shapes', 'output_shapes', 'output_slices')


def key_marker(key: str) -> bytes:
  # pickle protocol 5 stores short dict keys as SHORT_BINUNICODE: 0x8c <len> <utf8>
  return b'\x8c' + bytes([len(key)]) + key.encode()


def schema_of(keys) -> str:
  """Which of the loader paths in modeld_v2 this pkl is shaped for."""
  keys = set(keys)
  if {'run', 'input_specs'} <= keys:
    return "upstream run-schema (needs an external driving_warp pkl)"
  if 'run_model' in keys:
    return "sunnypilot run_model (warp baked in)"
  if 'run_policy' in keys:
    return "sunnypilot vision/policy split"
  return "unknown"


def total_size(path: str) -> int:
  """Size on disk, summed over the chunks when the artifact was delivered chunked."""
  try:
    return sum(os.path.getsize(chunk) for chunk in get_existing_chunks(path))
  except OSError:
    return 0


def walk_buffers(f) -> str:
  """Walk the out-of-band buffer records and report which container format this is.

  dump_oob writes <int64 length><payload> records that end exactly at EOF. tinygrad's newer
  openpilot artifacts (tinygrad #18260) instead append a 256-byte-aligned arena addressed by
  persistent ids, so the same walk reads payload bytes as a length and desyncs at once.
  """
  records = 0
  while True:
    head = f.read(8)
    if not head:
      return f"length-prefixed records, {records} of them - load_oob can read this"
    if len(head) < 8:
      return f"truncated header after {records} records - file is incomplete"
    (length,) = struct.unpack('<q', head)
    if not 0 <= length <= MAX_RECORD:
      return "not length-prefixed, looks like a persistent-id arena - load_oob CANNOT read this"
    remaining = length
    while remaining:
      block = f.read(min(READ_BLOCK, remaining))
      if not block:
        return "record runs past EOF, looks like a persistent-id arena - load_oob CANNOT read this"
      remaining -= len(block)
    records += 1


def probe(path: str, attempt_load: bool = True) -> list[str]:
  lines = [os.path.basename(path)]
  try:
    f = open_file_chunked(path)
    opcodes = f.read(struct.unpack('<q', f.read(8))[0])
  except Exception as e:
    return lines + [f"  - unreadable: {type(e).__name__}: {e}"]

  keys = [key for key in SCHEMA_KEYS if key_marker(key) in opcodes]
  lines.append(f"  - size: {total_size(path)} bytes")
  lines.append(f"  - schema: {schema_of(keys)}")
  lines.append(f"  - keys: {', '.join(keys) if keys else 'none found'}")
  lines.append(f"  - buffers: {walk_buffers(f)}")

  if attempt_load:
    lines += attempt_loads(path)
  return lines


def attempt_loads(path: str) -> list[str]:
  """Open the pkl every way the runtime might, because they do not all behave the same.

  modeld_v2 has its own compatibility unpickler that rewrites tinygrad globals, so a file that
  a straight unpickle reads fine can still fail the load that matters. Report both, then, if the
  straight one failed, retry it in the model's device context the way upstream does - that
  separates a container mismatch, which fails every way, from a device that only resolves under
  Context(DEV=...).
  """
  ok, lines = _try_load(path)
  lines += _try_compat_load(path)
  if ok:
    return lines

  try:
    from tinygrad import Context
  except Exception as e:
    return lines + [f"  - load_oob on {CHESTNUT_DEV}: tinygrad unavailable ({e})"]

  try:
    with Context(DEV=CHESTNUT_DEV):
      return lines + _try_load(path, label=f"load_oob on {CHESTNUT_DEV}")[1]
  except Exception as e:
    return lines + [f"  - load_oob on {CHESTNUT_DEV}: {type(e).__name__}: {str(e)[:MAX_ERROR_CHARS]}"]


def _try_compat_load(path: str) -> list[str]:
  try:
    from openpilot.sunnypilot.modeld_v2.helpers import load_oob as compat_load_oob
  except Exception as e:
    return [f"  - compat unpickle: unavailable ({e})"]
  try:
    jits = compat_load_oob(open_file_chunked(path))
  except Exception as e:
    return [f"  - compat unpickle: {type(e).__name__}: {str(e)[:MAX_ERROR_CHARS]}"]
  return [f"  - compat unpickle: ok, top level {', '.join(sorted(str(k) for k in jits))}"]


def _try_load(path: str, label: str = "plain unpickle") -> tuple[bool, list[str]]:
  try:
    jits = load_oob(open_file_chunked(path))
  except Exception as e:
    return False, [f"  - {label}: {type(e).__name__}: {str(e)[:MAX_ERROR_CHARS]}"]
  return True, [f"  - {label}: ok, top level {', '.join(sorted(str(k) for k in jits))}"]


def tinygrad_head() -> str:
  try:
    return subprocess.check_output(['git', '-C', 'tinygrad_repo', 'rev-parse', 'HEAD'],
                                   cwd=BASEDIR, text=True, timeout=20).strip()
  except Exception as e:
    return f"unknown ({e})"


def selected_chestnut_pkl() -> tuple[str | None, list[str]]:
  """The pkl behind the chestnut bundle the model selector currently points at."""
  from openpilot.common.hardware.hw import Paths
  from openpilot.sunnypilot.models.helpers import _bundle_is_valid_locally, get_selected_bundle

  bundle = get_selected_bundle(source="chestnut")
  if bundle is None:
    return None, ["chestnut bundle: none selected"]
  lines = [f"chestnut bundle: {bundle.displayName}",
           f"  - files: {'valid' if _bundle_is_valid_locally(bundle) else 'MISSING or hash mismatch'}"]
  if not bundle.models:
    return None, lines + ["  - bundle lists no models"]
  path = os.path.join(Paths.model_root(), bundle.models[0].artifact.fileName)
  return path, lines


def last_error_lines() -> list[str]:
  """What modeld recorded the last time it tried to bring the big model up this boot."""
  from openpilot.common.params import Params

  error = Params().get("ChestnutLastError")
  if not error:
    return ["modeld's last chestnut error: none recorded this boot"]
  return ["modeld's last chestnut error:"] + [f"  {line}" for line in error.strip().splitlines()]


def warp_lines() -> list[str]:
  lines = ["warp pkls in selfdrive/modeld/models:"]
  warps = sorted(glob.glob(str(MODELS_DIR / '*driving_warp_*_tinygrad.pkl')))
  for warp in warps:
    lines.append(f"  - {os.path.basename(warp)} ({os.path.getsize(warp)} bytes)")
  if not warps:
    lines.append("  - none, so an upstream run-schema model cannot load")
  return lines


def dry_run() -> list[str]:
  """Bring the big model up the way modeld does, but offroad and in this process.

  modeld only runs onroad, so without this the only way to find out whether a model gets past
  the load is to go and drive. This does the same init and the same warmup pass against the
  eGPU, so whatever modeld would hit shows up here instead.
  """
  from openpilot.common.hardware import HARDWARE
  from openpilot.common.params import Params
  from openpilot.common.transformations.camera import _ar_ox_fisheye, _os_fisheye
  from openpilot.selfdrive.modeld.helpers import chestnut_present

  if not chestnut_present():
    return ["dry run: chestnut is not attached"]

  # the same camera the build picks its warp resolution from
  camera = _os_fisheye if HARDWARE.get_device_type() == "mici" else _ar_ox_fisheye
  lines = [f"dry run: {camera.width}x{camera.height} camera on chestnut"]

  started = time.monotonic()
  try:
    from openpilot.sunnypilot.modeld_v2.modeld import ModelState
    state = ModelState(cam_w=camera.width, cam_h=camera.height, chestnut=True)
    schema = "upstream run" if getattr(state, 'is_upstream_run', False) else "sunnypilot"
    lines.append(f"  - init ok in {time.monotonic() - started:.1f}s, {schema} schema")

    # the camera resolution above says nothing about whether the frame stage engaged, so report
    # what the model was actually built around
    lines.append(f"  - frames: {Params().get('ChestnutFrameMode') or 'unknown'}")
    lines.append(f"  - c4 intrinsics: {getattr(state, 'c4_intrinsics', False)}")

    warming = time.monotonic()
    state.warmup()
    lines.append(f"  - warmup ok in {time.monotonic() - warming:.1f}s, the model runs")
  except Exception:
    lines.append(f"  - FAILED after {time.monotonic() - started:.1f}s")
    lines += [f"  {line}" for line in traceback.format_exc().strip().splitlines()[-14:]]
  return lines


def report(paths: list[str] | None = None, attempt_load: bool = True) -> str:
  lines = [f"tinygrad_repo: {tinygrad_head()}"]

  if paths:
    targets = paths
  else:
    target, bundle_lines = selected_chestnut_pkl()
    lines += bundle_lines
    targets = [target] if target else []

  if not targets:
    lines.append("no pkl to inspect")
  for path in targets:
    lines += probe(path, attempt_load)

  return "\n".join(lines + warp_lines() + last_error_lines())


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('paths', nargs='*', help='pkls to inspect; default is the selected chestnut bundle')
  parser.add_argument('--all', action='store_true', help='inspect every downloaded pkl')
  parser.add_argument('--no-load', action='store_true', help='skip the load attempt, inspect only')
  parser.add_argument('--dry-run', action='store_true', help='bring the big model up the way modeld does')
  args = parser.parse_args()

  if args.dry_run:
    _, bundle_lines = selected_chestnut_pkl()
    print("\n".join([f"tinygrad_repo: {tinygrad_head()}"] + bundle_lines + warp_lines() + dry_run()))
    return

  paths = args.paths
  if args.all:
    from openpilot.common.hardware.hw import Paths
    paths = sorted(glob.glob(os.path.join(Paths.model_root(), '*tinygrad.pkl')))
  print(report(paths, attempt_load=not args.no_load))


if __name__ == '__main__':
  main()
