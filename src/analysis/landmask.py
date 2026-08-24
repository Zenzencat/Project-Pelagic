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
water by this dataset. Not relevant to the reported bug (Jurong/Bukit
Batok/Queenstown are long-established, non-reclaimed land and are correctly
classified), but worth knowing for any future coastal scene near very
recent reclamation.
"""
import numpy as np
from global_land_mask import globe


def strip_land_pixels(preds, center_lat, center_lon, scale):
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

    Returns (masked_preds, stats). `preds` is not mutated in place.
    """
    H, W = preds.shape
    ys, xs = np.indices((H, W))
    lats = center_lat + (H // 2 - ys) * scale
    lons = center_lon + (xs - W // 2) * scale
    is_land = globe.is_land(lats, lons)

    oil_mask = preds == 255
    oil_px_total_before = int(oil_mask.sum())
    oil_px_removed_on_land = int((oil_mask & is_land).sum())

    masked = preds.copy()
    masked[is_land] = 0

    stats = {
        "oil_px_total_before": oil_px_total_before,
        "oil_px_removed_on_land": oil_px_removed_on_land,
        "land_px_fraction_in_scene": float(is_land.mean()),
    }
    return masked, stats
