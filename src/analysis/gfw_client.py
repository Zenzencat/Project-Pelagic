"""
Project Pelagic — Real AIS Vessel Attribution via Global Fishing Watch
SWU Prasarnmit AI Engineering Final Project

Replaces the old hardcoded mock_vessels in src/api/main.py's predict() with
real nearby-vessel attribution, for detections that have a real acquisition
datetime (live-fetched scenes only -- see src/data/cdse_fetch.py). Queries
GFW's v3 4Wings Report API for real AIS vessel presence within a radius of
the slick centroid, at the closest available AIS timestamp to acquisition.

Dataset choice, verified live against this project's actual GFW_TOKEN:
`public-global-presence:latest` (resolves to `public-global-presence:v4.0`)
is GFW's real all-vessel AIS presence dataset (category "activity",
subcategory "presence") -- covers every vessel type (cargo, fishing,
passenger, support, etc.), not just fishing effort, which is what
"candidate nearby vessels" needs. `public-global-fishing-effort:latest`
(the dataset most GFW API examples use) was deliberately NOT used here since
it only covers apparent fishing activity. Verified with a real request
against a real Singapore Strait bbox (2026-08-10/12): the API returned real
per-vessel entries (mmsi, shipName, lat, lon, date, entry/exitTimestamp,
flag, vesselType, hours) directly -- no separate vessel-ID-to-MMSI resolution
call needed, since `group-by: MMSI` is a real, working reportGrouping for
this dataset (confirmed via GET /v3/datasets/public-global-presence:latest's
own `reportGroupings` field, which lists vessel_id/flag/mmsi).

The dataset is GFW's own real product, not a per-ping AIS feed: entries are
daily-resolution (temporal-resolution=DAILY is the finest this report
endpoint offers), so "closest available AIS timestamp to acquisition" is
honestly day-level, not sub-day -- disclosed here and in the API response,
not hidden.

Spatial resolution (found investigating a real reported bug: 3 of 5
attributed vessels showing an identical 0.0km distance, and only 2 of 5
markers visible on the map). Verified directly against a real request's raw
`entries` rows, not assumed: with `spatial-resolution: HIGH`, every row's
`lat`/`lon` lands on a 0.01-degree grid (e.g. 1.25/103.79, 1.18/103.78,
1.19/103.88 -- all exact 0.01 multiples). This is the finest resolution the
4Wings Report endpoint offers -- it's a spatial-aggregation product, not a
per-vessel-ping feed, so `lat`/`lon` is the grid cell's reference point, not
each vessel's precise position. Confirmed against the real bug: 3 real,
distinct vessels (different MMSIs, genuinely present) legitimately shared
one grid cell that day, so a straight haversine distance from that identical
cell center is not a calculation bug -- it's a correctly-computed distance
to an honestly-coarse position. `distance_meters` here is left as that real,
correctly-computed value; `position_resolution_m` is added to each vessel so
callers can render the coarseness honestly (e.g. "same ~1.1km AIS cell"
instead of a falsely precise "0.0km") instead of the distance number
silently implying more precision than the underlying data has.
"""

import math

import requests

REPORT_URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"

# See module docstring: the real all-vessel AIS presence dataset, verified
# live against this project's GFW_TOKEN. Re-verify against
# GET /v3/datasets/public-global-presence:latest if GFW bumps the version.
GFW_DATASET = "public-global-presence:latest"

# The "HIGH" spatial-resolution grid cell size for the 4Wings Report
# endpoint (see module docstring) -- 0.01 degrees, verified directly against
# real response coordinates. Requested as a fixed value below (not derived
# from the API's params dict) so it can't silently drift out of sync if the
# request's "spatial-resolution" value is ever changed without updating this.
SPATIAL_RESOLUTION_DEG = 0.01

EARTH_RADIUS_M = 6_371_000.0
DEG_LAT_TO_M = math.radians(1.0) * EARTH_RADIUS_M


def _grid_cell_diagonal_m(lat):
    """Real diagonal size (meters) of one SPATIAL_RESOLUTION_DEG grid cell
    at the given latitude -- the honest "position could be anywhere within
    this radius" figure for a vessel reported at a grid-cell center."""
    lat_m = SPATIAL_RESOLUTION_DEG * DEG_LAT_TO_M
    lon_m = SPATIAL_RESOLUTION_DEG * DEG_LAT_TO_M * max(math.cos(math.radians(lat)), 1e-6)
    return math.hypot(lat_m, lon_m)


def _haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _buffer_polygon(center_lat, center_lon, radius_km, n_points=32):
    """Small circular buffer (GeoJSON Polygon) around the centroid, in
    degrees -- close enough for the small (<=~50km) radii this endpoint is
    meant for; not a geodesic-exact buffer."""
    lat_deg_per_km = 1.0 / 111.32
    lon_deg_per_km = 1.0 / (111.32 * max(math.cos(math.radians(center_lat)), 1e-6))
    coords = []
    for i in range(n_points + 1):
        theta = 2 * math.pi * i / n_points
        dlat = radius_km * lat_deg_per_km * math.sin(theta)
        dlon = radius_km * lon_deg_per_km * math.cos(theta)
        coords.append([round(center_lon + dlon, 6), round(center_lat + dlat, 6)])
    return {"type": "Polygon", "coordinates": [coords]}


def get_nearby_vessels(gfw_token, center_lat, center_lon, acquisition_datetime_iso, radius_km=10.0, top_n=5, timeout=30):
    """
    Real GFW AIS vessel-presence query within radius_km of (center_lat,
    center_lon), at the closest available (daily) AIS data to
    acquisition_datetime_iso.

    Returns {"status": "ok"|"empty"|"skipped_no_credentials"|"error",
             "vessels": [...], "detail": str}. Never fabricates a vessel on
    error/empty/no-credentials -- returns the status instead, so the caller
    (src/api/main.py) can store and the UI can render an honest reason.
    Each vessel dict: mmsi, vessel_name, latitude, longitude, timestamp,
    distance_meters, position_resolution_m -- explicitly "candidate nearby
    vessels," not confirmed source. Deduplicated by MMSI (keeping the
    closest-to-centroid entry) so one real vessel can't fill two of the
    returned slots.
    """
    if not gfw_token:
        return {"status": "skipped_no_credentials", "vessels": [], "detail": "GFW_TOKEN not configured."}

    try:
        acq_date = acquisition_datetime_iso[:10]  # YYYY-MM-DD
        from datetime import datetime, timedelta
        day = datetime.strptime(acq_date, "%Y-%m-%d")
        date_from = day.strftime("%Y-%m-%d")
        date_to = (day + timedelta(days=1)).strftime("%Y-%m-%d")

        body = {
            "geojson": _buffer_polygon(center_lat, center_lon, radius_km),
            "group-by": "MMSI",
        }
        params = {
            "spatial-resolution": "HIGH",
            "temporal-resolution": "DAILY",
            "datasets[0]": GFW_DATASET,
            "date-range": f"{date_from},{date_to}",
            "format": "JSON",
        }
        r = requests.post(
            REPORT_URL,
            headers={"Authorization": f"Bearer {gfw_token}", "Content-Type": "application/json"},
            params=params,
            json=body,
            timeout=timeout,
        )
    except requests.RequestException as e:
        return {"status": "error", "vessels": [], "detail": f"GFW request failed: {e}"}

    if r.status_code == 401 or r.status_code == 403:
        return {"status": "error", "vessels": [], "detail": f"GFW rejected the token: HTTP {r.status_code}."}
    if r.status_code != 200:
        return {"status": "error", "vessels": [], "detail": f"GFW report request failed: HTTP {r.status_code} — {r.text[:300]}"}

    try:
        data = r.json()
        raw_entries = []
        for entry_group in data.get("entries", []):
            for rows in entry_group.values():
                # A region with genuinely zero vessel presence for the date
                # returns `null` for its dataset key here (verified live
                # against a real remote Patagonia-fjord bbox/date with no
                # AIS activity), not an empty list -- found via a real crash
                # (`'NoneType' object is not iterable`) on that real query,
                # not assumed defensively.
                if rows:
                    raw_entries.extend(rows)
    except (ValueError, AttributeError) as e:
        return {"status": "error", "vessels": [], "detail": f"Could not parse GFW response: {e}"}

    # `group-by: MMSI` groups each vessel's presence per grid cell per day,
    # not into one row per vessel overall -- a vessel that crossed more than
    # one 0.01deg cell on the acquisition day produces multiple raw rows
    # with the same MMSI. Deduplicate here (keep the row closest to the
    # slick centroid) so one real vessel can't occupy two of the top_n
    # candidate slots under the same MMSI.
    best_by_mmsi = {}
    for row in raw_entries:
        lat, lon = row.get("lat"), row.get("lon")
        mmsi = row.get("mmsi")
        if lat is None or lon is None or mmsi is None:
            continue
        dist_m = _haversine_m(center_lat, center_lon, lat, lon)
        if dist_m > radius_km * 1000:
            continue
        if mmsi in best_by_mmsi and best_by_mmsi[mmsi]["distance_meters"] <= dist_m:
            continue
        best_by_mmsi[mmsi] = {
            "mmsi": mmsi,
            "vessel_name": row.get("shipName") or None,
            "latitude": lat,
            "longitude": lon,
            "timestamp": row.get("date"),
            "distance_meters": round(dist_m, 1),
            "position_resolution_m": round(_grid_cell_diagonal_m(lat), 1),
        }

    candidates = list(best_by_mmsi.values())
    candidates.sort(key=lambda v: v["distance_meters"])
    top = candidates[:top_n]

    if not top:
        return {"status": "empty", "vessels": [], "detail": f"No AIS vessel presence found within {radius_km}km on {acq_date}."}
    return {"status": "ok", "vessels": top, "detail": f"{len(top)} candidate nearby vessel(s) within {radius_km}km."}
