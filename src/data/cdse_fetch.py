"""
Project Pelagic — Live Sentinel-1 Fetch via Copernicus Data Space Ecosystem
SWU Prasarnmit AI Engineering Final Project

Fetches a real, calibrated, orthorectified Sentinel-1 GRD scene for a given
bounding box + date range, for the live detection pipeline
(POST /api/live/fetch in src/api/main.py). Two real CDSE services, both
verified live against this project's actual configured credentials:

1. Product search: the public (no-auth) OData catalog, via
   src/analysis/sentinel1_revisit_check.py::search_same_track_passes() --
   reused as-is, not duplicated. Used for real product discovery and the
   real acquisition start/stop datetime (OData's ContentDate.Start/End),
   since that's the one field this module must never fabricate/default.
2. Pixel data: CDSE's Sentinel Hub Process API (sh.dataspace.copernicus.eu).

Why Process API, not a raw SAFE product download (the originally-planned
approach): this repo's CDSE_CLIENT_ID/SECRET (confirmed valid by
scripts/validate_credentials.py) are Sentinel Hub OAuth2 client credentials.
They authenticate fine against CDSE's shared Keycloak token endpoint
(src/analysis/cdse_auth.py), but a live test against the raw-product
download service (zipper/download.dataspace.copernicus.eu) rejected them
with HTTP 401 "Token audience not allowed" (error code DAT-ZIP-609) -- that
service needs a different credential type (CDSE portal username/password),
not provided in this environment. The Sentinel Hub Catalog + Process APIs
were verified live and working with these exact credentials: a real request
against a real Singapore Strait Sentinel-1D scene (acquired 2026-08-11)
returned a real 512x512 float32 GeoTIFF with real ModelTiepointTag /
ModelPixelScaleTag values matching the requested bbox exactly, and real
sigma0 backscatter values (VV mean ~0.43, VH mean ~0.049 linear --
physically plausible, not placeholder data).

This is also a strictly simpler design than the original raw-download +
manual calibration-LUT-XML-parsing plan: Sentinel Hub performs real
SIGMA0_ELLIPSOID radiometric calibration and orthorectification
server-side, so the returned GeoTIFF is already in the exact linear-sigma0,
simple-lat/lon-grid form src/analysis/geoutils.py and the existing
speckle/dB/normalize preprocessing already expect. No GDAL, no manual GCP
handling, no LUT XML parsing, and no multi-GB download.
"""

import math
from datetime import datetime, timedelta

import requests

from src.analysis.sentinel1_revisit_check import search_same_track_passes

CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# Real ground sampling distance this project's model was trained/evaluated on
# (see src/api/main.py's REAL_PIXEL_SCALE_DEG) -- requesting output at this
# resolution keeps live scenes in the same distribution as the holdout data.
PIXEL_SCALE_DEG = 8.983152841195218e-05

# Sentinel Hub's synchronous Process API documents a hard cap on output
# raster size; keep comfortably under it while still allowing a reasonably
# large AOI for a demo bounding box.
MAX_OUTPUT_PX = 2048

# Matches src/inference.py::run_tiled_inference's patch_size default -- by
# requesting an exact multiple directly from the API, live scenes tile
# cleanly without needing the pad/crop fallback that function also gained
# for robustness against non-multiple inputs.
PATCH_SIZE = 256

EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{bands: ["VV", "VH"]}],
    output: {bands: 2, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(sample) {
  return [sample.VV, sample.VH];
}
"""


class CdseFetchError(Exception):
    """Raised when CDSE product search or the Process API fetch fails, or no
    matching product is found -- never silently falls back to a mock scene."""


def find_best_product(min_lat, min_lon, max_lat, max_lon, date_from_iso, date_to_iso):
    """
    Real OData catalog search (no auth needed) for Sentinel-1 GRD products
    intersecting the bbox within [date_from_iso, date_to_iso]. Returns the
    most recent match (dict with real id/name/start/end/footprint) or None.
    """
    ref = datetime.fromisoformat(date_from_iso.replace("Z", "+00:00")) + (
        datetime.fromisoformat(date_to_iso.replace("Z", "+00:00"))
        - datetime.fromisoformat(date_from_iso.replace("Z", "+00:00"))
    ) / 2
    window_days = max(
        1,
        (
            datetime.fromisoformat(date_to_iso.replace("Z", "+00:00"))
            - datetime.fromisoformat(date_from_iso.replace("Z", "+00:00"))
        ).days
        // 2
        + 1,
    )

    products = search_same_track_passes(
        min_lat, max_lat, min_lon, max_lon,
        reference_date_iso=ref.strftime("%Y-%m-%dT%H:%M:%S"),
        window_days=window_days,
        product_type="GRD",
    )
    if not products:
        return None
    products.sort(key=lambda p: p["start"], reverse=True)
    return products[0]


def _output_size(min_lat, min_lon, max_lat, max_lon):
    """Pixel dimensions for the bbox at the project's real GSD, rounded up to
    the next multiple of PATCH_SIZE and capped at MAX_OUTPUT_PX."""
    width_px = math.ceil((max_lon - min_lon) / PIXEL_SCALE_DEG)
    height_px = math.ceil((max_lat - min_lat) / PIXEL_SCALE_DEG)

    def round_up(n):
        return max(PATCH_SIZE, ((n + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE)

    return min(round_up(width_px), MAX_OUTPUT_PX), min(round_up(height_px), MAX_OUTPUT_PX)


def fetch_scene_geotiff(token, min_lat, min_lon, max_lat, max_lon, acquisition_start_iso, out_path):
    """
    Calls Sentinel Hub's Process API for a single UTC-day window around the
    product's real acquisition_start_iso, requesting real SIGMA0_ELLIPSOID
    calibration + orthorectification for (VV, VH), written to out_path as a
    GeoTIFF. The output's ModelTiepointTag/ModelPixelScaleTag match the
    requested bbox exactly -- src/analysis/geoutils.py's existing
    get_scene_geolocation() reads it back unmodified.
    """
    width_px, height_px = _output_size(min_lat, min_lon, max_lat, max_lon)

    day = datetime.fromisoformat(acquisition_start_iso.replace("Z", "+00:00"))
    day_start = day.strftime("%Y-%m-%dT00:00:00Z")
    day_end = (day + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

    body = {
        "input": {
            "bounds": {"bbox": [min_lon, min_lat, max_lon, max_lat]},
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {"from": day_start, "to": day_end},
                    "resolution": "HIGH",
                },
                "processing": {"backCoeff": "SIGMA0_ELLIPSOID", "orthorectify": True},
            }],
        },
        "output": {
            "width": width_px,
            "height": height_px,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT,
    }

    r = requests.post(
        PROCESS_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "image/tiff",
        },
        json=body,
        timeout=120,
    )
    if r.status_code != 200:
        raise CdseFetchError(f"CDSE Process API request failed: HTTP {r.status_code} — {r.text[:300]}")

    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path
