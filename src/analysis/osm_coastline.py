"""
Real OpenStreetMap coastline geometry for land-sea mask refinement.

Round 15's land mask (src/analysis/landmask.py) used `global-land-mask`, a
~1km-resolution pre-rasterized grid. Round 16 found that resolution too
coarse for small islands and complex local coastline: a real live-fetch
screenshot over the Singapore Strait showed Pulau Semakau (a real island,
~2.5km across) still partly covered by the predicted "oil" mask. Verified
directly: a fine test grid over Semakau's real bounding box showed only
32.8% of sample points reading as land under `global_land_mask` -- some of
the island's real area genuinely falls between the raster's ~930m sample
points.

Investigated switching to a static vector dataset first (Natural Earth
10m land polygons) as the task's primary suggestion -- measured, not
assumed: clipped to the same real Singapore Strait scene bbox, Natural
Earth's "land" + "minor islands" layers together only account for 7.62%
of the oil pixels the raster method flags as land (vs. the raster's own
27.10%), and neither Natural Earth layer contains Pulau Semakau at all.
Natural Earth's "10m" designation is a cartographic *map scale*
(1:10,000,000), not a ground resolution guarantee -- its coastline
generalization is coarser than useful at a single ~20km SAR scene's scale,
and it's actually a regression from the existing raster method here, not
an improvement. Full comparison numbers: see landmask.py's docstring.

This module instead queries the real OSM coastline dataset live via the
public Overpass API, scoped to just the scene's own bbox (a small, one-off
query per live-fetch, not a bulk download) -- OSM's coastline mapping for
a well-mapped area like Singapore is actively maintained and, verified
directly, does include Pulau Semakau (OSM relation 9964061), reconstructed
here from 11 separate raw way segments into one real 361-point closed ring
matching its known real bounding box (Nominatim: 1.1890651-1.2172813N,
103.7577792-103.7813209E).

Scope, chosen deliberately: this module only reconstructs and rasterizes
CLOSED rings (islands whose coastline forms a complete loop within the
query area). Continental/mainland coastline, which exits the query bbox
without closing, is deliberately left to the existing raster method in
landmask.py rather than attempting bbox-edge ring closure -- that requires
knowing the coastline's land-side winding convention, which was tested
empirically against known real coordinates and found unreliable to derive
from a small local sample (see landmask.py). Net effect: this module
strictly ADDS precision for islands the raster handles poorly; it never
removes raster-covered land elsewhere. If Overpass is unreachable or
returns nothing, callers fall back to the raster-only mask (never fails
the request over a missing refinement).

Known simplification: an island's real outer coastline ring is rasterized
as fully "land," including any genuinely separate inner water body (e.g. a
lagoon) unless that inner shore is itself tagged `natural=coastline` (most
enclosed lagoons/reservoirs are tagged `natural=water` instead, which this
module does not query). For Pulau Semakau specifically this means its
internal reservoir cells are treated as land too -- a minor over-masking
in exchange for correctly excluding the much larger real landmass, not a
regression from the pre-Round-16 state (which covered the whole island,
inner lagoon included, in "oil" instead).
"""
from collections import defaultdict

import cv2
import numpy as np
import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Overpass rejects requests with no descriptive User-Agent (HTTP 406) --
# found live, not assumed: the default `requests` library UA works fine
# against CDSE/GFW but Overpass specifically 406'd it until this header was
# added. Per Overpass/OSM API etiquette, identifies the real client.
REQUEST_HEADERS = {"User-Agent": "ProjectPelagic/1.0 (SWU AI Engineering student project; oil-slick land-sea masking)"}

# Extra margin (degrees) around the scene bbox for the Overpass query, so an
# island whose coastline pokes just outside the exact scene bbox (but whose
# body still overlaps it) isn't missed because none of its way's nodes fell
# strictly inside the bbox filter. ~0.03deg is a few km at these latitudes.
QUERY_MARGIN_DEG = 0.03


def fetch_coastline_ways(min_lat, min_lon, max_lat, max_lon, timeout=25):
    """Real natural=coastline ways from OSM within (bbox + margin).

    Returns {"status": "ok"|"empty"|"error", "elements": [...], "detail": str}.
    Never raises -- a live-fetch request must not fail just because this
    optional refinement's network call failed or Overpass is rate-limiting.
    """
    q_min_lat = min_lat - QUERY_MARGIN_DEG
    q_min_lon = min_lon - QUERY_MARGIN_DEG
    q_max_lat = max_lat + QUERY_MARGIN_DEG
    q_max_lon = max_lon + QUERY_MARGIN_DEG
    query = (
        f'[out:json][timeout:{max(int(timeout) - 5, 5)}];'
        f'way["natural"="coastline"]({q_min_lat},{q_min_lon},{q_max_lat},{q_max_lon});'
        f'out geom;'
    )
    try:
        r = requests.post(OVERPASS_URL, data={"data": query}, headers=REQUEST_HEADERS, timeout=timeout)
    except requests.RequestException as e:
        return {"status": "error", "elements": [], "detail": f"Overpass request failed: {e}"}

    if r.status_code != 200:
        return {"status": "error", "elements": [], "detail": f"Overpass HTTP {r.status_code}"}

    try:
        data = r.json()
    except ValueError as e:
        return {"status": "error", "elements": [], "detail": f"Could not parse Overpass response: {e}"}

    elements = [e for e in data.get("elements", []) if len(e.get("geometry", []) or []) >= 2]
    if not elements:
        return {"status": "empty", "elements": [], "detail": "No natural=coastline ways found near this bbox."}
    return {"status": "ok", "elements": elements, "detail": f"{len(elements)} real OSM coastline way(s) found."}


def _pt_key(lon, lat, ndigits=7):
    # OSM ways sharing a node have bit-identical lat/lon at that node;
    # rounding just guards against float round-tripping through JSON.
    return (round(lon, ndigits), round(lat, ndigits))


def assemble_closed_rings(elements):
    """Chain raw coastline way segments into closed rings by matching
    shared endpoint coordinates. A single island's real coastline is often
    split across several separate OSM ways (ways are cut at data-management
    boundaries, junctions, etc.) -- using each way in isolation would miss
    this. Direction-agnostic and doesn't need to know which side is "land":
    a ring is just "closed" if walking connected segments returns to the
    start point.

    Returns (closed_rings, open_paths) -- each a list of [(lon, lat), ...].
    Open paths (didn't close within the query's ways -- e.g. continental
    coastline extending beyond the query area) are returned for visibility
    but intentionally not rasterized as land by this module (see module
    docstring).
    """
    chains = [[(p["lon"], p["lat"]) for p in e["geometry"]] for e in elements]
    used = [False] * len(chains)

    endpoint_index = defaultdict(list)
    for i, c in enumerate(chains):
        endpoint_index[_pt_key(*c[0])].append((i, True))
        endpoint_index[_pt_key(*c[-1])].append((i, False))

    closed_rings, open_paths = [], []
    for i in range(len(chains)):
        if used[i]:
            continue
        used[i] = True
        path = list(chains[i])

        while True:
            candidates = [(j, s) for (j, s) in endpoint_index[_pt_key(*path[-1])] if not used[j]]
            if not candidates:
                break
            j, is_start = candidates[0]
            used[j] = True
            nxt = chains[j] if is_start else list(reversed(chains[j]))
            path.extend(nxt[1:])

        while True:
            candidates = [(j, s) for (j, s) in endpoint_index[_pt_key(*path[0])] if not used[j]]
            if not candidates:
                break
            j, is_start = candidates[0]
            used[j] = True
            prev = list(reversed(chains[j])) if is_start else chains[j]
            path = prev[:-1] + path

        if len(path) >= 4 and _pt_key(*path[0]) == _pt_key(*path[-1]):
            closed_rings.append(path)
        else:
            open_paths.append(path)

    return closed_rings, open_paths


def rasterize_closed_rings(closed_rings, center_lat, center_lon, scale, H, W):
    """Even-odd rasterize closed coastline rings onto an HxW canvas at the
    scene's own pixel resolution (not a fixed global sample grid), so
    precision is limited only by the real vector geometry, not by a
    pre-baked lookup spacing. `cv2.fillPoly` given multiple contours in one
    call uses the even-odd rule, so a genuinely separate inner ring (e.g. a
    hole actually tagged as its own coastline) comes out correct without
    this function needing to classify which ring is an outer boundary vs.
    a hole.

    Returns an HxW boolean array (True = land).
    """
    canvas = np.zeros((H, W), dtype=np.uint8)
    if not closed_rings:
        return canvas == 255

    contours = []
    for ring in closed_rings:
        pts = np.array(
            [
                ((lon - center_lon) / scale + W / 2, H / 2 - (lat - center_lat) / scale)
                for lon, lat in ring
            ],
            dtype=np.int32,
        )
        contours.append(pts)
    cv2.fillPoly(canvas, contours, 255)
    return canvas == 255


def get_osm_land_mask(min_lat, min_lon, max_lat, max_lon, center_lat, center_lon, scale, H, W, timeout=25):
    """End-to-end: fetch real coastline ways for this scene, assemble closed
    rings, rasterize them. Returns (is_land_bool_array, status_dict).
    `is_land_bool_array` is all-False (not an error) if OSM has no closed
    coastline data here or the request failed -- callers should treat that
    as "no refinement available," not "confirmed no land," and keep relying
    on the raster mask underneath.
    """
    result = fetch_coastline_ways(min_lat, min_lon, max_lat, max_lon, timeout=timeout)
    if result["status"] != "ok":
        return np.zeros((H, W), dtype=bool), result

    closed_rings, open_paths = assemble_closed_rings(result["elements"])
    is_land = rasterize_closed_rings(closed_rings, center_lat, center_lon, scale, H, W)
    status = {
        "status": "ok",
        "detail": f"{len(closed_rings)} closed island ring(s), {len(open_paths)} open (mainland) path(s) from OSM.",
        "n_closed_rings": len(closed_rings),
        "n_open_paths": len(open_paths),
    }
    return is_land, status
