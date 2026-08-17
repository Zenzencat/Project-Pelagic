"""
Binary segmentation mask -> GeoJSON polygon conversion.

Split out of src/api/main.py's predict() so the same mask->polygon logic can
be reused by a one-off DB-regeneration script (re-deriving geojson_mask for
already-inferred scenes from their saved mask PNGs) without duplicating the
contour tracing / simplification / coordinate-transform math.
"""
import cv2
import numpy as np


def chaikin_smooth(points, iterations):
    """Chaikin's corner-cutting algorithm on a closed vertex loop (points
    not yet duplicated-closed). Each iteration replaces every edge (p0, p1)
    with two points at 1/4 and 3/4 along it, rounding off corners without
    moving the curve outside its original edges -- it only cuts them, so it
    can't invent boundary detail that wasn't already implied by the polygon
    it's given. Doubles the vertex count per iteration.
    """
    pts = np.asarray(points, dtype=float)
    for _ in range(iterations):
        n = len(pts)
        new_pts = np.empty((n * 2, 2), dtype=float)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            new_pts[2 * i] = 0.75 * p0 + 0.25 * p1
            new_pts[2 * i + 1] = 0.25 * p0 + 0.75 * p1
        pts = new_pts
    return pts


# Simplification tolerance as a fraction of each contour's own perimeter
# (cv2.arcLength), so it scales with fragment size rather than using one
# fixed pixel epsilon for both a 2000px slick and a 5px speckle fragment.
#
# Previous value (0.01 = 1% of perimeter) was measured directly against the
# raw cv2.findContours output for the 4 live demo scenes: it collapsed a
# ~4750-point raw contour down to as few as 14 vertices, which is why the
# rendered slicks looked like angular crystals/lightning bolts instead of
# organic slick shapes. 0.002 (0.2%) was chosen by comparing rendered
# contours at several candidate values against the raw boundary -- it keeps
# 3-6x more real vertices (e.g. 14 -> 64 for oil_00000's main fragment, 16 ->
# 88 for oil_00004's), which is enough to make the actual detected curves and
# concavities visible, while still cutting the ~5000-point raw contour down
# to a payload-reasonable size (a fixed-fraction epsilon still discards raw
# per-pixel staircase noise, it just discards less of the real shape than
# 0.01 did).
DEFAULT_EPSILON_FRAC = 0.002

# One Chaikin pass to round off the pixel-grid staircase that
# cv2.approxPolyDP's straight-line segments leave behind (a rasterization
# artifact, not real slick detail) -- applied AFTER simplification, not
# instead of it, so it smooths real vertices rather than inventing curve
# detail approxPolyDP had already thrown away. Doubles vertex count once;
# 2 iterations (4x) was visually indistinguishable from 1 on the largest
# fragments but meaningfully bloated payload size, so 1 was kept.
DEFAULT_CHAIKIN_ITERATIONS = 1

# 6 decimal degrees =~ 11cm at the equator, far finer than the ~10m/pixel
# real ground sampling distance of the source imagery -- rounding to this
# many decimals doesn't discard any real precision, it just stops each
# vertex from serializing ~17 significant figures of float noise. Matters
# more now that simplification keeps more vertices per fragment.
DEFAULT_ROUND_DECIMALS = 6


def mask_to_polygons(
    mask,
    center_lat,
    center_lon,
    scale,
    epsilon_frac=DEFAULT_EPSILON_FRAC,
    chaikin_iterations=DEFAULT_CHAIKIN_ITERATIONS,
    round_decimals=DEFAULT_ROUND_DECIMALS,
):
    """Trace a binary (0/255) mask into closed [lon, lat] polygon rings.

    Mirrors the pixel->lat/lon transform previously inlined in
    src/api/main.py's predict(): lat grows upward from the mask's vertical
    center, lon grows rightward from its horizontal center, both scaled by
    the scene's real per-pixel degree size.
    """
    H, W = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        epsilon = epsilon_frac * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = approx.reshape(-1, 2).astype(float)

        if len(pts) < 3:
            continue

        if chaikin_iterations > 0:
            pts = chaikin_smooth(pts, chaikin_iterations)

        poly_pts = []
        for x, y in pts:
            lat = center_lat + (H // 2 - y) * scale
            lon = center_lon + (x - W // 2) * scale
            poly_pts.append([round(float(lon), round_decimals), round(float(lat), round_decimals)])

        if len(poly_pts) >= 3:
            poly_pts.append(poly_pts[0])  # Close polygon
            polygons.append(poly_pts)

    return polygons
