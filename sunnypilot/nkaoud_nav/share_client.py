"""
"Share" destination fetcher for nkaoud_nav.

The user configures NkaoudNavShareEndpoint to point at any of:

  1. Plain HTTP(S) URL: nkaoud_nav `GET`s it; the response is parsed for
     latitude/longitude (and optionally place_name).

  2. Neon Postgres connection string (postgres:// or postgresql://):
     nkaoud_nav POSTs to https://<host>/sql with the connection string as
     a Neon-Connection-String header and a fixed query:
        SELECT latitude, longitude, COALESCE(place_name, '') AS place_name
        FROM destinations ORDER BY id DESC LIMIT 1
     Same as the old fork's Neon path -- the destinations table needs id,
     latitude, longitude (place_name optional).

Accepted JSON shapes for the HTTP(S) path:
  - dict:  {"latitude": 24.7, "longitude": 46.6, "place_name": "..."}
  - list:  [ {"latitude": ...}, ... ]                            (first item)
  - rows:  {"rows": [[lat, lon, place_name?], ...]}              (first row)
The Neon path always returns the rows shape.

Anything else -> ShareFetchError. The result dict mirrors what the picker
writes to NkaoudNavDestination so navd can put it straight in.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import requests

from openpilot.common.swaglog import cloudlog


# Pull everything from the row so we don't depend on the user having a
# specific set of columns. We look up latitude / longitude / place_name
# by name (via the Neon response's `fields` metadata) so the column order
# in their CREATE TABLE doesn't matter and extra columns are ignored.
NEON_QUERY = "SELECT * FROM destinations ORDER BY id DESC LIMIT 1"

NEON_SCHEMES = ("postgres", "postgresql")

# Column names we treat as the destination label, in priority order.
NAME_COLUMN_CANDIDATES = ("place_name", "name", "label", "title")


class ShareFetchError(RuntimeError):
  pass


def _is_neon_url(url: str) -> bool:
  scheme = urlparse(url).scheme.lower()
  return scheme in NEON_SCHEMES


def _extract(payload: Any) -> dict[str, Any] | None:
  if isinstance(payload, dict):
    if "latitude" in payload and "longitude" in payload:
      return payload
    rows = payload.get("rows")
    fields = payload.get("fields") or []
    if isinstance(rows, list) and rows:
      row = rows[0]
      if isinstance(row, list):
        # Neon-Array-Mode response. Use the response's `fields` metadata
        # so we look up columns by NAME instead of by position; this works
        # regardless of how the user ordered their CREATE TABLE.
        col_idx = {f.get("name"): i for i, f in enumerate(fields) if isinstance(f, dict)}
        if "latitude" in col_idx and "longitude" in col_idx:
          out: dict[str, Any] = {
            "latitude": row[col_idx["latitude"]],
            "longitude": row[col_idx["longitude"]],
          }
          for cand in NAME_COLUMN_CANDIDATES:
            if cand in col_idx and row[col_idx[cand]]:
              out["place_name"] = row[col_idx[cand]]
              break
          return out
        # If fields metadata is PRESENT but doesn't include latitude /
        # longitude, the user's table really is missing them -- don't
        # silently treat the first two values as coordinates.
        if fields:
          return None
        # No fields metadata at all -- fall back to legacy positional
        # shape ([lat, lon, name?]).
        if len(row) >= 2:
          out = {"latitude": row[0], "longitude": row[1]}
          if len(row) >= 3 and row[2]:
            out["place_name"] = row[2]
          return out
      if isinstance(row, dict) and "latitude" in row and "longitude" in row:
        return row
    return None
  if isinstance(payload, list) and payload:
    return _extract(payload[0])
  return None


def _fetch_neon(connection_string: str, timeout: float) -> Any:
  parsed = urlparse(connection_string)
  host = parsed.hostname
  if not host:
    raise ShareFetchError("Neon connection string is missing a hostname")
  url = f"https://{host}/sql"
  cloudlog.info(f"share_client: Neon POST {url}")
  try:
    resp = requests.post(
      url,
      timeout=timeout,
      headers={
        "Neon-Connection-String": connection_string,
        "Neon-Raw-Text-Output": "true",
        "Neon-Array-Mode": "true",
      },
      json={"query": NEON_QUERY, "params": []},
    )
  except requests.RequestException as e:
    raise ShareFetchError(f"network error: {e}") from e
  cloudlog.info(f"share_client: Neon status={resp.status_code} body={resp.text[:300]!r}")
  if resp.status_code != 200:
    raise ShareFetchError(f"neon http {resp.status_code}: {resp.text[:200]}")
  try:
    return resp.json()
  except ValueError as e:
    raise ShareFetchError(f"neon response was not JSON: {e}") from e


def _fetch_http(url: str, timeout: float) -> Any:
  try:
    resp = requests.get(url, timeout=timeout)
  except requests.RequestException as e:
    raise ShareFetchError(f"network error: {e}") from e
  if resp.status_code != 200:
    raise ShareFetchError(f"http {resp.status_code}: {resp.text[:200]}")
  try:
    return resp.json()
  except ValueError as e:
    raise ShareFetchError(f"response was not JSON: {e}") from e


def fetch_share_destination(url: str, timeout: float = 5.0) -> dict[str, Any]:
  """Returns a dict with at least latitude/longitude (and place_name if
  available). Raises ShareFetchError on any failure. Picks Neon POST or
  plain HTTP GET based on the URL scheme."""
  if not url:
    raise ShareFetchError("share endpoint is empty")

  payload = _fetch_neon(url, timeout) if _is_neon_url(url) else _fetch_http(url, timeout)

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
