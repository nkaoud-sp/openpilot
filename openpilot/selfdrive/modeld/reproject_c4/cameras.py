"""camerad's narrow and wide frames paired by start-of-frame time, as modeld pairs them on a two-camera device."""

from msgq.visionipc import VisionIpcClient
from openpilot.common.swaglog import cloudlog


def recv_pair(narrow: VisionIpcClient, wide: VisionIpcClient):
  """The newest narrow frame and the wide frame taken with it, or None. The narrow client conflates; the wide must not,
  so the frames it queued can be drained up to the narrow's."""
  buf_n = narrow.recv()
  if buf_n is None:
    return None
  while True:
    buf_w = wide.recv()
    if buf_w is None or narrow.timestamp_sof < wide.timestamp_sof + 25000000:
      break
  if buf_w is None:
    return None
  if abs(narrow.timestamp_sof - wide.timestamp_sof) > 10000000:
    cloudlog.error(f"reproject: frames out of sync! narrow {narrow.frame_id} ({narrow.timestamp_sof / 1e9:.5f}), wide {wide.frame_id} ({wide.timestamp_sof / 1e9:.5f})")
  return buf_n, buf_w
