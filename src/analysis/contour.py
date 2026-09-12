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
    scale_y=None,
):
    """Trace a binary (0/255) mask into closed [lon, lat] polygons, holes included.

    Mirrors the pixel->lat/lon transform previously inlined in
    src/api/main.py's predict(): lat grows upward from the mask's vertical
    center, lon grows rightward from its horizontal center, both scaled by
    the scene's real per-pixel degree size.

    Returns a list of polygons; each polygon is a list of closed [lon, lat]
    rings, ring 0 the exterior boundary and any further rings holes cut out
    of it (standard GeoJSON Polygon interior-ring convention) -- one nesting
    level deeper than this function used to return, since a flat list of
    rings had no way to represent a hole at all.

    Previously used cv2.RETR_EXTERNAL, which only traces outer boundaries.
    Any 0-valued region fully enclosed by 255-valued pixels -- most often
    land a real land-sea mask correctly zeroed out, but a model-predicted
    gap inside a slick works the same way -- has no outer boundary of its
    own, so RETR_EXTERNAL silently dropped it and the surrounding fill
    swallowed it whole. Confirmed against a real live-fetch scene (Stockholm
    Archipelago, 2026-09-12 investigation): the land mask had correctly
    zeroed Tynningo island's pixels, but the rendered polygon still covered
    it, because the island's exclusion was topologically a hole with no
    outer contour of its own. RETR_TREE (full hierarchy, not just the
    2-level RETR_CCOMP) is used so a positive region nested inside a hole
    inside another region -- a lake on an island in a strip of open water,
    however unlikely in practice -- still comes out as its own polygon
    rather than being silently merged one level up.
    """
    H, W = mask.shape
    scale_y = scale if scale_y is None else scale_y
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]  # per-contour [next, previous, first_child, parent]

    def transform_ring(cnt):
        epsilon = epsilon_frac * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = approx.reshape(-1, 2).astype(float)

        if len(pts) < 3:
            return None

        if chaikin_iterations > 0:
            pts = chaikin_smooth(pts, chaikin_iterations)

        ring = []
        for x, y in pts:
            lat = center_lat + (H // 2 - y) * scale_y
            lon = center_lon + (x - W // 2) * scale
            ring.append([round(float(lon), round_decimals), round(float(lat), round_decimals)])

        if len(ring) < 3:
            return None
        ring.append(ring[0])  # Close ring
        return ring

    polygons = []

    def add_polygon_rooted_at(ext_idx):
        """`ext_idx` is a contour that is itself a fill region (even depth,
        e.g. an exterior boundary or a positive region re-emerging inside a
        hole). Its direct children are holes cut into it; each of THOSE
        children (grandchildren of `ext_idx`) is fill again and becomes its
        own top-level polygon via a fresh call to this function."""
        exterior = transform_ring(contours[ext_idx])
        if exterior is None:
            return
        rings = [exterior]

        child = hierarchy[ext_idx][2]
        while child != -1:
            hole_ring = transform_ring(contours[child])
            if hole_ring is not None:
                rings.append(hole_ring)

            grandchild = hierarchy[child][2]
            while grandchild != -1:
                add_polygon_rooted_at(grandchild)
                grandchild = hierarchy[grandchild][0]

            child = hierarchy[child][0]

        polygons.append(rings)

    for i in range(len(contours)):
        if hierarchy[i][3] == -1:  # no parent -> top-level exterior
            add_polygon_rooted_at(i)

    return polygons
