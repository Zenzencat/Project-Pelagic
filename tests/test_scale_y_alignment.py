"""The land mask and the contours it filters must share one latitude grid.

`get_scene_geolocation()` only reports the GeoTIFF's ScaleX as
`pixel_scale_deg`. Live Process scenes are resampled to a fixed width/height,
so ScaleY differs from ScaleX for any bbox that isn't square in degrees.
`mask_to_polygons` takes a separate `scale_y`; `strip_land_pixels` (and the OSM
refinement under it) must be given the same one, or predicted pixels get
removed from one latitude grid while the surviving polygon is drawn on another.
Measured at up to ~11 km apart before this was wired through.
"""
import numpy as np
import pytest
from global_land_mask import globe

from src.analysis.contour import mask_to_polygons
from src.analysis.landmask import strip_land_pixels
from src.analysis.osm_coastline import rasterize_closed_rings

# The two real bounding boxes from the QC report, with the raster sizes
# src/data/cdse_fetch.py::_output_size produces for them.
RECTANGULAR_SCENES = [
    ((1.0, 103.0, 1.05, 103.2), 2048, 768),
    ((59.0, 18.0, 59.2, 18.4), 2048, 2048),
]


def grids(bbox, width, height):
    min_lat, min_lon, max_lat, max_lon = bbox
    return {
        "center_lat": (min_lat + max_lat) / 2,
        "center_lon": (min_lon + max_lon) / 2,
        "scale": (max_lon - min_lon) / width,
        "scale_y": (max_lat - min_lat) / height,
    }


def bbox_grid(bbox, width, height):
    """Per-pixel lat/lon derived only from the scene's real corner bounds --
    an oracle independent of the formula inside strip_land_pixels."""
    min_lat, min_lon, max_lat, max_lon = bbox
    lats = max_lat - np.arange(height) * ((max_lat - min_lat) / height)
    lons = min_lon + np.arange(width) * ((max_lon - min_lon) / width)
    return np.meshgrid(lons, lats)


@pytest.mark.parametrize("bbox,width,height", RECTANGULAR_SCENES)
def test_land_mask_matches_the_scenes_real_bbox_grid(bbox, width, height):
    geo = grids(bbox, width, height)
    assert geo["scale_y"] != geo["scale"], "this bbox must be non-square to be a real test"

    # Claim every pixel, so the surviving mask *is* the sea mask.
    mask = np.full((height, width), 255, dtype=np.uint8)
    kept, _ = strip_land_pixels(mask, geo["center_lat"], geo["center_lon"], geo["scale"],
                                scale_y=geo["scale_y"])

    lon_grid, lat_grid = bbox_grid(bbox, width, height)
    expected_sea = ~globe.is_land(lat_grid, lon_grid)
    assert expected_sea.any() and not expected_sea.all(), "scene must straddle a coastline"
    np.testing.assert_array_equal(kept == 255, expected_sea)

    # And the pre-fix behaviour (longitude scale used for latitude) really was wrong.
    square_pixel, _ = strip_land_pixels(mask, geo["center_lat"], geo["center_lon"], geo["scale"])
    assert not np.array_equal(square_pixel == 255, expected_sea)


@pytest.mark.parametrize("bbox,width,height", RECTANGULAR_SCENES)
def test_polygon_vertices_stay_inside_the_scene_bounds(bbox, width, height):
    geo = grids(bbox, width, height)
    min_lat, _, max_lat, _ = bbox

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[0:16, 0:16] = 255  # top-left corner, where a latitude-scale error is largest

    rings = mask_to_polygons(mask, geo["center_lat"], geo["center_lon"], geo["scale"],
                             scale_y=geo["scale_y"])
    lats = [point[1] for ring in rings for point in ring]
    assert rings and min_lat - 1e-6 <= min(lats) and max(lats) <= max_lat + 1e-6

    # The land mask now judges that same top row at that same latitude.
    land_mask_top_lat = geo["center_lat"] + (height // 2) * geo["scale_y"]
    assert land_mask_top_lat == pytest.approx(max(lats), abs=1e-4)


@pytest.mark.parametrize("bbox,width,height", RECTANGULAR_SCENES)
def test_osm_rasterization_uses_the_latitude_scale(bbox, width, height):
    geo = grids(bbox, width, height)
    min_lat, min_lon, max_lat, max_lon = bbox

    # A ring covering the northern half of the scene in real degrees.
    half_lat = (min_lat + max_lat) / 2
    ring = [(min_lon, half_lat), (max_lon, half_lat), (max_lon, max_lat), (min_lon, max_lat)]

    correct = rasterize_closed_rings([ring], geo["center_lat"], geo["center_lon"],
                                     geo["scale"], height, width, scale_y=geo["scale_y"])
    square_pixel = rasterize_closed_rings([ring], geo["center_lat"], geo["center_lon"],
                                          geo["scale"], height, width)

    # Northern half of the raster, within a row of rounding.
    assert correct[: height // 2 - 1, :].all()
    assert not correct[height // 2 + 1:, :].any()
    assert not np.array_equal(correct, square_pixel), "scale_y must actually change the result"


def test_scale_y_defaults_to_scale_for_square_pixel_callers():
    """Every cached holdout scene has ScaleY == ScaleX; those callers pass no
    scale_y at all and must stay byte-identical."""
    scale = 8.983153e-05
    center_lat, center_lon = 34.0, 35.5
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:24, 8:24] = 255

    assert mask_to_polygons(mask, center_lat, center_lon, scale) == \
        mask_to_polygons(mask, center_lat, center_lon, scale, scale_y=scale)

    ring = [(35.4, 33.9), (35.6, 33.9), (35.6, 34.1), (35.4, 34.1)]
    np.testing.assert_array_equal(
        rasterize_closed_rings([ring], center_lat, center_lon, scale, 64, 64),
        rasterize_closed_rings([ring], center_lat, center_lon, scale, 64, 64, scale_y=scale),
    )

    default, _ = strip_land_pixels(mask, center_lat, center_lon, scale)
    explicit, _ = strip_land_pixels(mask, center_lat, center_lon, scale, scale_y=scale)
    np.testing.assert_array_equal(default, explicit)
