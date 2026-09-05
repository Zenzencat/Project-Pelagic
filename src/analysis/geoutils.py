"""
Project Pelagic — GeoTIFF Geolocation Extraction
SWU Prasarnmit AI Engineering Final Project

The Trujillo-Acatitla et al. Sentinel-1 GeoTIFFs (Zenodo 8346860 / 8253899 /
13761290) carry real embedded georeferencing (ModelTiepointTag + ModelPixelScaleTag,
WGS84) even though the app's demo API currently ignores it entirely and shows every
scene at a fixed Gulf-of-Thailand mock coordinate (src/api/main.py). This module reads
the real per-scene location, needed as an input to both the ERA5 wind check and the
Sentinel-1 revisit check.

Verified directly against the actual holdout files: e.g. oil_00000.tif's tiepoint is
(34.693 N, 35.622 E) with a pixel scale of 8.983153e-05 deg/pixel (~10 m/pixel at this
latitude) -- consistent with the Eastern Mediterranean coverage area (30-36 E, 31-34.7 N)
documented for this class of dataset. No acquisition timestamp is present in these
files or anywhere else in the Zenodo deposits (confirmed via the Zenodo API's file
listing for all three records) -- see docs/status.md for the full trail.
"""

import tifffile


def get_scene_geolocation(tif_path):
    """
    Reads real WGS84 georeferencing from a Trujillo-Acatitla-style GeoTIFF.

    Returns a dict with the scene's four corner coordinates and center point,
    computed from the GeoTIFF's origin tiepoint + pixel scale + raster size.
    Raises ValueError if the file has no embedded georeferencing (e.g. a
    synthetic/mock scene) rather than silently returning a fabricated location.
    """
    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        tags = {t.name: t.value for t in page.tags}

    if "ModelTiepointTag" not in tags or "ModelPixelScaleTag" not in tags:
        raise ValueError(
            f"{tif_path} has no embedded GeoTIFF georeferencing "
            "(ModelTiepointTag/ModelPixelScaleTag missing) -- likely a synthetic "
            "or non-georeferenced scene."
        )

    # ModelTiepointTag: (I, J, K, X, Y, Z) -- raster point (I, J) maps to model
    # space (X, Y). For these files (I, J) = (0, 0), i.e. the top-left pixel.
    tiepoint = tags["ModelTiepointTag"]
    origin_lon, origin_lat = tiepoint[3], tiepoint[4]

    # ModelPixelScaleTag: (ScaleX, ScaleY, ScaleZ) in degrees/pixel.
    scale_x, scale_y, _ = tags["ModelPixelScaleTag"]

    width = tags["ImageWidth"]
    height = tags["ImageLength"]

    # GeoTIFF pixel-is-area convention: Y (latitude) decreases as raster row
    # increases (origin is top-left / north-west corner).
    min_lon = origin_lon
    max_lat = origin_lat
    max_lon = origin_lon + width * scale_x
    min_lat = origin_lat - height * scale_y

    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0

    return {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lon": min_lon,
        "max_lon": max_lon,
        # "pixel_scale_deg" is ScaleX (longitude). It is NOT safe to reuse it
        # for latitude: the two are equal on every Trujillo-Acatitla holdout
        # scene, but a live Sentinel Hub Process scene is resampled to a fixed
        # width/height, so ScaleY differs whenever the requested bbox is not
        # square in degrees. Callers that convert rows to latitude want
        # "pixel_scale_y_deg" -- see src/analysis/landmask.py::strip_land_pixels
        # and src/analysis/contour.py::mask_to_polygons.
        "pixel_scale_deg": scale_x,
        "pixel_scale_y_deg": scale_y,
    }
