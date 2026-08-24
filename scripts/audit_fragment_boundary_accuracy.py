"""
Round 17: fragment boundary accuracy audit.

Rounds 15/16 confirmed the land-sea mask correctly EXCLUDES land at the pixel
level (measured removal rates) and that entire islands (Pulau Semakau) are
no longer covered by the predicted "oil" mask. This script answers a
narrower, previously-unverified question: for the individual polygon
fragment BOUNDARIES themselves, how closely do they trace the real
coastline -- pixel-accurate, or off by some visible margin?

Method: re-fetch the real OSM coastline for the same Stockholm Archipelago
and Chonos Archipelago bboxes used in scripts/capture_live_variety_screenshots.js
(Round 16's live-fetch test locations), assemble closed island rings (same
src/analysis/osm_coastline.py code the live pipeline uses), rasterize a
handful of small-to-medium real islands at the exact pixel scale
src/data/cdse_fetch.py::PIXEL_SCALE_DEG uses for live fetches, extract the
rasterized boundary via cv2.findContours, and measure each boundary pixel's
nearest-segment distance back to the original (un-rasterized) OSM vector
ring, in real meters.

This is a read-only measurement -- no production code is touched.
"""
import json
import math
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analysis.osm_coastline import assemble_closed_rings, rasterize_closed_rings

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "ProjectPelagic/1.0 (SWU AI Engineering student project; oil-slick land-sea masking)"}
PIXEL_SCALE_DEG = 8.983152841195218e-05  # src/data/cdse_fetch.py::PIXEL_SCALE_DEG (real live-fetch GSD)

LOCATIONS = {
    "stockholm": {"bbox": (59.20, 18.30, 59.45, 18.75), "island_idx_by_span": [118, 897, 1106]},
    "chonos": {"bbox": (-45.30, -74.10, -45.05, -73.65), "island_idx_by_span": [66, 16, 9]},
}


def fetch_closed_rings(min_lat, min_lon, max_lat, max_lon, cache_path=None, retries=3):
    if cache_path and Path(cache_path).exists():
        elements = json.loads(Path(cache_path).read_text())
    else:
        q = f'[out:json][timeout:60];way["natural"="coastline"]({min_lat},{min_lon},{max_lat},{max_lon});out geom;'
        for attempt in range(retries):
            try:
                r = requests.post(OVERPASS_URL, data={"data": q}, headers=HEADERS, timeout=90)
                if r.status_code == 200:
                    elements = [e for e in r.json().get("elements", []) if len(e.get("geometry", []) or []) >= 2]
                    if cache_path:
                        Path(cache_path).write_text(json.dumps(elements))
                    break
                print(f"  attempt {attempt}: HTTP {r.status_code}, retrying...")
            except requests.RequestException as e:
                print(f"  attempt {attempt}: {e}, retrying...")
            time.sleep(10)
        else:
            raise RuntimeError("Overpass fetch failed after retries")
    closed_rings, open_paths = assemble_closed_rings(elements)
    return closed_rings, open_paths


def bbox_size_m(ring):
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    clat = sum(lats) / len(lats)
    dlat_m = (max(lats) - min(lats)) * 111320
    dlon_m = (max(lons) - min(lons)) * 111320 * math.cos(math.radians(clat))
    return max(dlon_m, dlat_m), clat


def _point_seg_dist(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    t = 0.0 if ab2 == 0 else max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def audit_island(ring, pad_px=40):
    """Rasterize one real closed OSM ring at the live-fetch pixel scale, then
    measure the rasterized boundary's real-meter deviation from the original
    vector ring it was rasterized from."""
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    center_lon = (min(lons) + max(lons)) / 2
    center_lat = (min(lats) + max(lats)) / 2
    mlat = 111320.0
    mlon = 111320.0 * math.cos(math.radians(center_lat))

    half_w_px = int((max(lons) - min(lons)) / PIXEL_SCALE_DEG / 2) + pad_px
    half_h_px = int((max(lats) - min(lats)) / PIXEL_SCALE_DEG / 2) + pad_px
    W, H = max(2 * half_w_px, 20), max(2 * half_h_px, 20)

    is_land = rasterize_closed_rings([ring], center_lat, center_lon, PIXEL_SCALE_DEG, H, W)
    mask_u8 = is_land.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)

    ring_xy_m = [((lo - center_lon) * mlon, (la - center_lat) * mlat) for lo, la in ring]

    dists = []
    for x, y in contour:
        lon = (x - W / 2) * PIXEL_SCALE_DEG + center_lon
        lat = center_lat - (y - H / 2) * PIXEL_SCALE_DEG
        px_m, py_m = (lon - center_lon) * mlon, (lat - center_lat) * mlat
        best = min(
            _point_seg_dist(px_m, py_m, *ring_xy_m[i], *ring_xy_m[i + 1])
            for i in range(len(ring_xy_m) - 1)
        )
        dists.append(best)
    dists = np.array(dists)

    span_m, clat = bbox_size_m(ring)
    return {
        "span_m": span_m, "center_lat": clat,
        "n_vector_pts": len(ring), "n_raster_boundary_pts": len(contour),
        "mean_dev_m": float(dists.mean()), "median_dev_m": float(np.median(dists)),
        "max_dev_m": float(dists.max()), "p90_dev_m": float(np.percentile(dists, 90)),
        "pixel_scale_m_lat": PIXEL_SCALE_DEG * mlat, "pixel_scale_m_lon": PIXEL_SCALE_DEG * mlon,
    }


if __name__ == "__main__":
    for name, cfg in LOCATIONS.items():
        print(f"=== {name} ===")
        closed_rings, open_paths = fetch_closed_rings(*cfg["bbox"], cache_path=f"output/round17_{name}_coastline.json")
        print(f"  {len(closed_rings)} closed rings, {len(open_paths)} open paths")
        for idx in cfg["island_idx_by_span"]:
            result = audit_island(closed_rings[idx])
            print(f"  idx={idx}: {result}")
