"""
Project Pelagic — Multi-Temporal SAR Comparison (standalone analysis, NOT wired
into the live pipeline)
SWU Prasarnmit AI Engineering Final Project

Real oil slicks drift and change shape between passes (current/wind advect them);
many look-alikes (biogenic films pinned to a source, wind-shadow zones, current
fronts) hold a static shape and location. This module searches the Copernicus Data
Space Ecosystem (CDSE) catalog for a same-area Sentinel-1 acquisition near a scene's
reference date, and — once a second pass is actually downloaded and run through the
model — compares the two detected polygons to flag `static_across_passes` as a
look-alike-likely signal.

STATUS:
  - CDSE's OData catalog *search* is public, no auth required (verified directly
    against https://catalogue.dataspace.copernicus.eu/odata/v1/Products). Actually
    *downloading* a matched product does require a free CDSE account + OAuth2
    client credentials (register at https://dataspace.copernicus.eu), which are
    not configured in this environment.
  - More fundamentally, same as src/analysis/era5_wind_check.py: this dataset has
    no acquisition timestamp for any of the 30 holdout scenes (verified against
    all three Zenodo deposits directly), so there is no reference date to search
    "before/after" in the first place. search_same_track_passes() below works and
    was validated with real coordinates, but only answers "how much Sentinel-1
    coverage exists at this location in general" (see
    docs/sentinel1_coverage_density.json), not "does THIS scene have a matched
    second pass" -- that needs a real timestamp per scene.
  - Sentinel-1 revisit cadence: verified via dataspace.copernicus.eu -- the
    2026 constellation reconfiguration (Sentinel-1C + Sentinel-1D, completed
    2026-06-24) restored a ~6-day nominal same-track revisit, not the ~12-day
    figure that held during the single-satellite gap. Do not assume 12 days.

Shape-comparison choice: centroid-drift + area-change, not polygon IoU. IoU
between two independently-thresholded detections is sensitive to small boundary
noise even for a genuinely static object (see the whole-image-vs-tiled inference
investigation in a prior session, where ~25% of a tiny false-positive blob's
pixels disagreed between two runs of the *same* scene for boundary-noise reasons
alone) -- centroid position + total area are coarser but much more robust to that
noise, and are what the underlying hypothesis (drift vs static) actually needs.
"""

import time

import requests

CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

# Nominal same-track repeat cycle post-2026 reconfiguration (Sentinel-1C + 1D).
# Source: https://dataspace.copernicus.eu/news/2026-5-28-sentinel-1-orbital-reconfiguration-dates
NOMINAL_REVISIT_DAYS = 6


def search_same_track_passes(min_lat, max_lat, min_lon, max_lon, reference_date_iso=None,
                               window_days=7, product_type="GRD"):
    """
    Searches CDSE for Sentinel-1 products intersecting a bbox.

    If reference_date_iso is given, restricts to +/- window_days around it (the
    actual "find a nearby pass" use case). If None, searches a full trailing
    12-month window instead, which only answers a coverage-density question
    ("how often does S1 pass here at all"), not "is there a pass near THIS
    scene's specific acquisition" -- used by the coverage pre-check since no
    reference date exists for this dataset.

    Returns a list of dicts: [{"id", "name", "start", "end", "footprint"}, ...],
    real API results, no auth needed for search. "id" is the OData product
    UUID -- required to actually download a product (see src/data/cdse_fetch.py),
    not needed by this module's own revisit-comparison use case, but harmless
    to include for callers that reuse this search function for that purpose.
    """
    bbox_wkt = (
        f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
        f"{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
    )

    if reference_date_iso:
        from datetime import datetime, timedelta
        ref = datetime.fromisoformat(reference_date_iso)
        start = (ref - timedelta(days=window_days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end = (ref + timedelta(days=window_days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    else:
        # No reference date available for this dataset -- fall back to a
        # trailing 12-month window purely to gauge coverage density.
        from datetime import datetime, timedelta
        now = datetime.utcnow()
        start = (now - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    filt = (
        f"Collection/Name eq 'SENTINEL-1' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{bbox_wkt}') and "
        f"contains(Name,'{product_type}') and "
        f"ContentDate/Start gt {start} and ContentDate/Start lt {end}"
    )

    r = requests.get(CDSE_ODATA_URL, params={"$filter": filt, "$top": 50}, timeout=30)
    r.raise_for_status()
    data = r.json()

    return [
        {
            "id": p["Id"],
            "name": p["Name"],
            "start": p["ContentDate"]["Start"],
            "end": p["ContentDate"].get("End"),
            "footprint": p.get("Footprint"),
        }
        for p in data.get("value", [])
    ]


def coverage_density_precheck(scene_coords, window_days_year=365):
    """
    Real, runnable-now check: for each scene's actual coordinates, count
    Sentinel-1 GRD passes over the last 12 months. Doesn't need a reference
    date or any credentials. Answers "is there any hope of a second pass ever
    existing here" as a necessary (not sufficient) precondition, per-region,
    since coverage density varies a lot by latitude/location (see module
    docstring and docs/sentinel1_coverage_density.json for real measured
    results across all 30 holdout scenes).
    """
    results = {}
    for scene_id, geo in scene_coords.items():
        pad = 0.01  # ~1km box around the center point
        try:
            passes = search_same_track_passes(
                geo["center_lat"] - pad, geo["center_lat"] + pad,
                geo["center_lon"] - pad, geo["center_lon"] + pad,
                reference_date_iso=None,
            )
            results[scene_id] = len(passes)
        except Exception as e:
            results[scene_id] = f"ERROR: {e}"
        time.sleep(0.2)
    return results


def compare_shapes(polygon_a_px, polygon_b_px, pixel_scale_deg):
    """
    Coarse centroid-drift + area-change comparison between two detected
    polygons (as pixel-coordinate point lists from the same-shaped scene).
    Returns dict with centroid_drift_m and area_change_pct. Flags
    'static_across_passes' if drift is small and area is stable -- thresholds
    are NOT validated against real data yet (no second-pass data exists to
    validate against) and should be treated as a starting point.
    """
    import numpy as np

    def centroid_and_area(poly):
        pts = np.array(poly)
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        cx, cy = pts.mean(axis=0)
        return (cx, cy), area

    (cx_a, cy_a), area_a = centroid_and_area(polygon_a_px)
    (cx_b, cy_b), area_b = centroid_and_area(polygon_b_px)

    drift_px = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
    drift_m = drift_px * pixel_scale_deg * 111_000  # deg -> m at equator, approximate

    area_change_pct = abs(area_a - area_b) / max(area_a, 1) * 100

    # Placeholder thresholds -- flagged as unvalidated in the module docstring.
    is_static = drift_m < 50 and area_change_pct < 20

    return {
        "centroid_drift_m": drift_m,
        "area_change_pct": area_change_pct,
        "flag": "static_across_passes" if is_static else None,
    }
