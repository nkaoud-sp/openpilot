import functools
import sys
import time
from pathlib import Path

from openpilot.common.hardware import AGNOS, HARDWARE
from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT, USB_DEVICES_PATH, cable_connected, is_chestnut_usb_id
from openpilot.common.params import Params

MODELS_DIR = Path(__file__).resolve().parent / 'models'


def modeld_pkl_path(chestnut: bool):
  prefix = 'big_' if chestnut else ''
  return MODELS_DIR / f'{prefix}driving_tinygrad.pkl'

def load_oob(path, chestnut=False):
  from tinygrad import Context
  device = 'USB+AMD:LLVM' if chestnut else 'QCOM' if AGNOS else 'METAL' if sys.platform == 'darwin' else 'CPU:LLVM'
  with Context(DEV=device):
    from tinygrad_repo.examples.openpilot.helpers import load_pickle
    return load_pickle(path, out_of_band=True)

def chestnut_present() -> bool:
  for d in USB_DEVICES_PATH.glob("*"):
    try:
      usb_id = (int((d / "idVendor").read_text(), 16), int((d / "idProduct").read_text(), 16))
      product = (d / "product").read_text().strip()
      if is_chestnut_usb_id(*usb_id) and product == CHESTNUT_USB_PRODUCT:
        return True
    except Exception:
      pass
  return False

def wait_for_chestnut(timeout: float = 10.) -> None:
  # the cable is detected before chestnut enumerates
  st = time.monotonic()
  while not chestnut_present():
    if time.monotonic() - st > timeout:
      raise TimeoutError("chestnut did not enumerate")
    time.sleep(0.1)

def chestnut_compiled() -> bool:
  path = modeld_pkl_path(chestnut=True)
  return path.is_file() and all(
    (MODELS_DIR / f'big_driving_warp_{size}_tinygrad.pkl').is_file() for size in ('1344x760', '1928x1208'))


@functools.cache
def reproject_expected() -> bool:
  """reprojectd serves the 3X cameras as a comma 4's for the big model: a 3X with a chestnut and the big model and its warps in place.
  The chestnut test is modeld's, a cable counting before the chestnut enumerates, so the two always agree on the frames. The
  AlwaysReproject developer toggle serves them without one, to the small model. Decided once per process, as modeld
  does: the manager starts reprojectd and reprojectcalibd on it, so a chestnut plugged in later, or the toggle, takes a reboot."""
  if HARDWARE.get_device_type() != "tizi":
    return False
  return Params().get_bool("AlwaysReproject") or (chestnut_compiled() and (chestnut_present() or cable_connected()))
