"""Live Sentinel-1 GRD discovery and calibrated GeoTIFF retrieval.

Reuses public CDSE OData search, shared OAuth authentication (at the caller),
and shared Sentinel Hub Process transport. Requests sigma0 VV/VH at the chosen
catalog acquisition interval, then validates source product names, dates,
valid-pixel coverage and GeoTIFF bounds before accepting the pixels.

The original day-window client was externally tested in earlier sessions.
The strict source/coverage path added on 2026-09-05 is locally tested only;
see docs/status.md for real external verification blockers.
"""

import math
from datetime import datetime, timedelta

from src.analysis.sentinel1_revisit_check import search_same_track_passes

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

class CdseFetchError(Exception):
    """Raised when CDSE product search or the Process API fetch fails, or no
    matching product is found -- never silently falls back to a mock scene."""


def find_best_product(min_lat, min_lon, max_lat, max_lon, date_from_iso, date_to_iso):
    """
    Real OData catalog search (no auth needed) for Sentinel-1 GRD products
    intersecting the bbox within [date_from_iso, date_to_iso]. Returns the
    most recent match (dict with real id/name/start/end/footprint) or None.
    """
    from datetime import timezone
    from src.data.observation_catalog import acquisition_time, valid_product, covers_bbox
    start = datetime.fromisoformat(date_from_iso.replace('Z', '+00:00'))
    end = datetime.fromisoformat(date_to_iso.replace('Z', '+00:00'))
    start = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
    end = end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end
    if len(date_to_iso) == 10:
        end += timedelta(days=1)
    ref = start + (end - start) / 2
    window_days = max(1, math.ceil((end - start).total_seconds() / 172800))
    bbox = (min_lat, min_lon, max_lat, max_lon)
    products = search_same_track_passes(min_lat, max_lat, min_lon, max_lon,
                                       ref.isoformat(), window_days=window_days)
    products = [p for p in products if valid_product(p) and
                start <= acquisition_time(p['start']) < end and
                '_IW_GRDH_1SDV_' in p['name'] and covers_bbox(p, bbox)]
    return max(products, key=lambda p: acquisition_time(p['start']), default=None)


def _output_size(min_lat, min_lon, max_lat, max_lon):
    """Pixel dimensions for the bbox at the project's real GSD, rounded up to
    the next multiple of PATCH_SIZE and capped at MAX_OUTPUT_PX."""
    width_px = math.ceil((max_lon - min_lon) / PIXEL_SCALE_DEG)
    height_px = math.ceil((max_lat - min_lat) / PIXEL_SCALE_DEG)

    def round_up(n):
        return max(PATCH_SIZE, ((n + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE)

    return min(round_up(width_px), MAX_OUTPUT_PX), min(round_up(height_px), MAX_OUTPUT_PX)


def fetch_scene_geotiff(token, min_lat, min_lon, max_lat, max_lon, acquisition_start_iso, out_path,
                        *, product=None):
    """Fetch only the selected acquisition; validate product metadata and data coverage.

    S1 supports ORBIT mosaicking. The returned orbit's source tiles must ALL
    identify the selected product; ambiguous mosaics fail instead of being
    assigned a fabricated single-scene identity. Real end metadata is required.
    """
    import io
    import numpy as np
    import tifffile
    from src.data.sentinel_process import process_request, verify_sources
    if product is None or product.get('start') != acquisition_start_iso or not product.get('end'):
        raise CdseFetchError('Real selected product identity and acquisition interval required')
    width, height = _output_size(min_lat, min_lon, max_lat, max_lon)
    body = {
        'input': {'bounds': {'bbox': [min_lon, min_lat, max_lon, max_lat]}, 'data': [{
            'type': 'sentinel-1-grd',
            'dataFilter': {'timeRange': {'from': product['start'], 'to': product['end']},
                           'resolution': 'HIGH', 'acquisitionMode': 'IW', 'polarization': 'DV'},
            'processing': {'backCoeff': 'SIGMA0_ELLIPSOID', 'orthorectify': True}}]},
        'output': {'width': width, 'height': height, 'responses': [
            {'identifier': 'default', 'format': {'type': 'image/tiff'}},
            {'identifier': 'validity', 'format': {'type': 'image/tiff'}},
            {'identifier': 'userdata', 'format': {'type': 'application/json'}}]},
        'evalscript': """
//VERSION=3
function setup() {
  return {input: ["VV", "VH", "dataMask"], mosaicking: "ORBIT", output: [
    {id: "default", bands: 2, sampleType: "FLOAT32"},
    {id: "validity", bands: 1, sampleType: "UINT8"}]};
}
function evaluatePixel(samples) {
  if (!samples.length) return {default: [0, 0], validity: [0]};
  return {default: [samples[0].VV, samples[0].VH], validity: [samples[0].dataMask]};
}
function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
  outputMetadata.userData = {tiles: [].concat.apply([], scenes.orbits.map(o => o.tiles))};
}
"""}
    try:
        files, metadata = process_request(token, body)
        provenance = verify_sources(metadata, product, 'S1')
        pixels = tifffile.imread(io.BytesIO(files['default.tif']))
        valid = tifffile.imread(io.BytesIO(files['validity.tif']))
        if pixels.shape != (height, width, 2) or not np.isfinite(pixels).all() or (pixels < 0).any():
            raise ValueError('Invalid calibrated VV/VH raster')
        if valid.shape != (height, width) or not (valid == 1).all():
            raise ValueError('Selected product does not provide valid data throughout the requested footprint')
        from src.analysis.geoutils import get_scene_geolocation
        geo = get_scene_geolocation(io.BytesIO(files['default.tif']))
        returned_bbox = [geo['min_lat'], geo['min_lon'], geo['max_lat'], geo['max_lon']]
        if not np.allclose(returned_bbox, [min_lat, min_lon, max_lat, max_lon], rtol=0, atol=1e-7):
            raise ValueError('Returned GeoTIFF footprint differs from requested region')
        provenance['bbox'] = returned_bbox
        with open(out_path, 'wb') as stream:
            stream.write(files['default.tif'])
        return provenance
    except Exception as exc:
        raise CdseFetchError(f'Sentinel-1 fetch/validation failed ({type(exc).__name__}): {exc}' if isinstance(exc, ValueError)
                             else f'Sentinel-1 fetch/validation failed ({type(exc).__name__}).') from exc
