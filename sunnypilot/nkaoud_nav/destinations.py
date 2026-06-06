from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Destination:
  key: str
  label: str
  latitude: float
  longitude: float

  def as_dict(self) -> dict:
    return {
      "latitude": self.latitude,
      "longitude": self.longitude,
      "place_name": self.label,
    }


# Ported verbatim from old fork (selfdrive/navd/navd.py:31-35 on fp-new-nkaoud-g4.7-nav).
# Replace these with your own presets at any time.
PRESETS: list[Destination] = [
  Destination("home",   "Navigation test - Home",   24.675764, 46.581478),
  Destination("work",   "Navigation test - Work",   24.714778, 46.683775),
  Destination("school", "Navigation test - School", 24.781423, 46.622246),
]


def preset_by_key(key: str) -> Destination | None:
  for d in PRESETS:
    if d.key == key:
      return d
  return None


# "Share" is not a static preset -- the coordinates come from an HTTP endpoint
# configured via NkaoudNavShareEndpoint at the moment the user picks it. The
# picker shows this label; navd handles the actual fetch.
SHARE_LABEL = "Share (fetch from endpoint)"
