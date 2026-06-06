"""
"Share" destination fetcher for nkaoud_nav.

The user configures NkaoudNavShareEndpoint to point at any HTTP(S) URL that
returns a destination. Supported response shapes (tried in order):

  1. JSON dict:  {"latitude": 24.7, "longitude": 46.6, "place_name": "..."}
  2. JSON list:  [ {"latitude": ..., "longitude": ..., ...}, ... ]   (first item used)
  3. Mapbox /sql-style:  {"rows": [[lat, lon], ...]}                 (first row used)

Anything else -> ShareFetchError. The result type mirrors what the picker
writes to NkaoudNavDestination so navd can just put it straight in.

Tied to fetch_route in route_client.py: same requests dependency, same
timeout pattern. Failure is silent except for cloudlog.warning so the UI
doesn't get a noisy error path -- if the share endpoint is misconfigured,
the user sees no route appear, checks swaglog, fixes the URL, retaps Share.
"""
from __future__ import annotations

from typing import Any

import requests


class ShareFetchError(RuntimeError):
  pass


def _extract(payload: Any) -> dict[str, Any] | None:
  if isinstance(payload, dict):
    if "latitude" in payload and "longitude" in payload:
      return payload
    rows = payload.get("rows")
    if isinstance(rows, list) and rows:
      row = rows[0]
      if isinstance(row, list) and len(row) >= 2:
        return {"latitude": row[0], "longitude": row[1]}
      if isinstance(row, dict) and "latitude" in row and "longitude" in row:
        return row
    return None
  if isinstance(payload, list) and payload:
    return _extract(payload[0])
  return None


def fetch_share_destination(url: str, timeout: float = 5.0) -> dict[str, Any]:
  """GETs the configured share URL, returns a dict with at least
  latitude / longitude (and place_name if provided), raises ShareFetchError
  on any failure."""
  if not url:
    raise ShareFetchError("share endpoint is empty")
  try:
    resp = requests.get(url, timeout=timeout)
  except requests.RequestException as e:
    raise ShareFetchError(f"network error: {e}") from e
  if resp.status_code != 200:
    raise ShareFetchError(f"http {resp.status_code}: {resp.text[:200]}")
  try:
    payload = resp.json()
  except ValueError as e:
    raise ShareFetchError(f"response was not JSON: {e}") from e

  data = _extract(payload)
  if data is None:
    raise ShareFetchError("response missing latitude / longitude")
  try:
    lat = float(data["latitude"])
    lon = float(data["longitude"])
  except (TypeError, ValueError) as e:
    raise ShareFetchError(f"latitude/longitude not numeric: {e}") from e

  result: dict[str, Any] = {"latitude": lat, "longitude": lon}
  name = data.get("place_name") or data.get("name") or ""
  if name:
    result["place_name"] = str(name)
  else:
    result["place_name"] = "Shared destination"
  return result
