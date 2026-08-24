"""
Land-sea masking for predicted oil-slick pixels.

SAR dark-backscatter oil-slick detectors are known to false-positive on land
(certain land surfaces -- calm bare soil, some vegetation, shadowed terrain --
produce the same low-backscatter "dark patch" signature the model is trained
to flag as a slick). Verified directly against a real /api/live/fetch run
over the Singapore Strait (bbox spanning both open water and the mainland/
Jurong coastline): 27.1% of the raw predicted "oil" pixels (715,798 of
2,641,623) fell on land, matching the visually-reported bug of the slick
polygon covering Jurong/Bukit Batok/Queenstown.

Dataset choice: `global-land-mask` (PyPI), a ~1km-resolution (21600x43200
global grid, 1/120 deg per cell -- verified directly against its bundled
.npz, not just the package description) land/ocean raster derived from the
GSHHG coastline database. Chosen over a live OSM coastline fetch or a
Natural Earth shapefile because it needs no network call and no additional
heavy geospatial stack (shapely/geopandas/fiona) beyond numpy, which this
project already depends on -- a single ~2.5MB compressed array bundled with
the package, queried with simple nearest-grid-cell lookup
(`global_land_mask.globe.is_land`, fully vectorized over numpy arrays).

Known limitation, found and verified while sanity-checking against known
Singapore coordinates before wiring this in: land reclaimed more recently
than the underlying GSHHG data (e.g. Marina Bay) is still classified as
water by this dataset. Not relevant to the original reported bug
(Jurong/Bukit Batok/Queenstown are long-established, non-reclaimed land and
are correctly classified), but worth knowing for any future coastal scene
near very recent reclamation.

Round 16 -- resolution gap found and fixed. A real live-fetch screenshot
showed the mask still covering Pulau Semakau (a real island) despite the
above fix. Root cause, verified directly: `global_land_mask`'s raster is a
fixed ~930m-spaced sample grid (21600x43200 over the whole Earth) -- a fine
test grid over Semakau's own real bounding box found only 32.8% of sample
points reading as land, meaning real island area genuinely falls between
grid points at this resolution. `src/analysis/osm_coastline.py` adds a
real-vector refinement on top (see that module's docstring for the full
investigation, including why the task's suggested Natural Earth 10m vector
dataset was tried and measurably rejected -- it captures *less* real land
here than the raster, and doesn't contain Pulau Semakau at all): closed
island coastline rings from live OSM data, reconstructed from raw way
segments and rasterized at the scene's own full pixel resolution rather
than a fixed global grid. `strip_land_pixels` below unions this with the
raster classification -- OSM refines islands the raster handles poorly;
the raster remains the fallback for continental coastline and for when
OSM/Overpass is unreachable.
"""
import numpy as np
from global_land_mask import globe

from src.analysis.osm_coastline import get_osm_land_mask


def strip_land_pixels(preds, center_lat, center_lon, scale, bbox=None, osm_timeout=25):
    """
    Zero out any predicted "oil" pixel (value 255) in `preds` that falls on
    land, so land never reaches contour extraction / area calculation /
    the saved mask PNG. Applied post-inference, pre-polygon-extraction, at
    the raster/pixel level -- not a frontend-only visual clip -- so the
    reported confidence score and slick area (km^2) are also corrected, not
    just the map drawing.

    Only meaningful for a scene with REAL georeferencing (real lat/lon on
    Earth); callers must not use this for the placeholder-coordinate
    fallback path (synthetic scenes with no embedded geolocation), since
    there's no real land/sea to check for a fictitious center point.

    `bbox`, if given, is (min_lat, min_lon, max_lat, max_lon) for the scene
    -- enables the real-OSM island refinement (Round 16) on top of the
    ~930m raster (Round 15). Omitted (or if the live Overpass call fails --
    never raises) falls back to raster-only, same as before Round 16.

    Returns (masked_preds, stats). `preds` is not mutated in place.
    """
    H, W = preds.shape
    ys, xs = np.indices((H, W))
    lats = center_lat + (H // 2 - ys) * scale
    lons = center_lon + (xs - W // 2) * scale
    is_land_raster = globe.is_land(lats, lons)

    is_land_osm = np.zeros((H, W), dtype=bool)
    osm_status = {"status": "skipped", "detail": "No bbox given; OSM refinement not attempted."}
    if bbox is not None:
        min_lat, min_lon, max_lat, max_lon = bbox
        is_land_osm, osm_status = get_osm_land_mask(
            min_lat, min_lon, max_lat, max_lon, center_lat, center_lon, scale, H, W, timeout=osm_timeout
        )

    is_land = is_land_raster | is_land_osm

    oil_mask = preds == 255
    oil_px_total_before = int(oil_mask.sum())
    oil_px_removed_on_land = int((oil_mask & is_land).sum())
    oil_px_removed_by_osm_only = int((oil_mask & is_land_osm & ~is_land_raster).sum())

    masked = preds.copy()
    masked[is_land] = 0

    stats = {
        "oil_px_total_before": oil_px_total_before,
        "oil_px_removed_on_land": oil_px_removed_on_land,
        "oil_px_removed_by_osm_only": oil_px_removed_by_osm_only,
        "land_px_fraction_in_scene": float(is_land.mean()),
        "osm_status": osm_status["status"],
        "osm_detail": osm_status["detail"],
    }
    return masked, stats
