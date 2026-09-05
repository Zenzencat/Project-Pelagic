"""Sentinel-1 catalog search and different-date comparison candidate selection.

Live comparisons reuse the primary fetch/inference pipeline. The older standalone
shape heuristic remains for reproducibility and is not used by the live UI/API.
Historical holdout scenes still have no acquisition timestamps.
"""

import time

# Nominal same-track repeat cycle post-2026 reconfiguration (Sentinel-1C + 1D).
# Source: https://dataspace.copernicus.eu/news/2026-5-28-sentinel-1-orbital-reconfiguration-dates
NOMINAL_REVISIT_DAYS = 6


def search_same_track_passes(min_lat, max_lat, min_lon, max_lon, reference_date_iso=None,
                               window_days=7, product_type="GRD"):
    """
    Searches CDSE for Sentinel-1 products intersecting a bbox.

    If reference_date_iso is given, restricts to +/- window_days around it (the
    actual "find a nearby pass" use case). If None, searches a full trailing
    12-month window instead, which only answers a coverage-density question
    ("how often does S1 pass here at all"), not "is there a pass near THIS
    scene's specific acquisition" -- used by the coverage pre-check since no
    reference date exists for this dataset.

    Returns real catalog identity, start/end, footprints and attributes. Search
    does not itself guarantee same track; select_comparison_products ranks
    compatible alternate-date observations using the returned attributes.
    Network or incomplete catalog responses raise; no scenes are fabricated.
    """
    from datetime import datetime, timedelta, timezone
    from src.data.observation_catalog import query_products, acquisition_time
    ref = acquisition_time(reference_date_iso) if reference_date_iso else datetime.now(timezone.utc)
    start = ref - timedelta(days=window_days if reference_date_iso else 365)
    end = ref + timedelta(days=window_days) if reference_date_iso else ref
    if product_type != "GRD":
        raise ValueError("Only Sentinel-1 GRD is supported")
    return query_products((min_lat, min_lon, max_lat, max_lon),
                          start.isoformat(), end.isoformat(), 'SENTINEL-1', "contains(Name,'GRD')")


def select_comparison_products(products, original, bbox, window_days=30):
    """Different UTC date, full footprint, compatible VV/VH IW GRD; no verdict."""
    from src.data.observation_catalog import acquisition_time, valid_product, covers_bbox
    ref = acquisition_time(original["start"])
    candidates = []
    seen = {original["id"]}
    for product in products:
        if not valid_product(product) or product['id'] in seen or product['name'] == original['name']:
            continue
        time = acquisition_time(product['start'])
        if time.date() == ref.date() or abs((time - ref).total_seconds()) > window_days * 86400:
            continue
        if '_IW_GRDH_1SDV_' not in product['name'] or not covers_bbox(product, bbox):
            continue
        seen.add(product['id'])
        candidates.append(product)
    def rank(product):
        attrs, ref_attrs = product.get('attributes', {}), original.get('attributes', {})
        same_track = (ref_attrs.get('relativeOrbitNumber') is not None and
                      attrs.get('relativeOrbitNumber') == ref_attrs['relativeOrbitNumber'])
        same_direction = (ref_attrs.get('orbitDirection') is not None and
                          attrs.get('orbitDirection') == ref_attrs['orbitDirection'])
        return (not same_track, not same_direction, abs((acquisition_time(product['start']) - ref).total_seconds()))
    # CDSE lists one sensing time under more than one product (a COG and a
    # non-COG row carry different ids and names for the same acquisition), so
    # deduplicating on id/name alone made four real revisits look like eight
    # candidates and left the caller's bounded retry trying the same
    # acquisition twice instead of falling back to another date. Full-bbox
    # coverage is already required above, so rows sharing a sensing time are
    # the same observation; keep the best-ranked representation of each.
    # Ties between representations resolve by catalog order (sorted() is
    # stable), so the choice is deterministic for a given catalog response.
    deduplicated, seen_times = [], set()
    for product in sorted(candidates, key=rank):
        time = acquisition_time(product['start'])
        if time in seen_times:
            continue
        seen_times.add(time)
        deduplicated.append(product)
    return deduplicated


def coverage_density_precheck(scene_coords, window_days_year=365):
    """
    Real, runnable-now check: for each scene's actual coordinates, count
    Sentinel-1 GRD passes over the last 12 months. Doesn't need a reference
    date or any credentials. Answers "is there any hope of a second pass ever
    existing here" as a necessary (not sufficient) precondition, per-region,
    since coverage density varies a lot by latitude/location (see module
    docstring and docs/sentinel1_coverage_density.json for real measured
    results across all 30 holdout scenes).
    """
    results = {}
    for scene_id, geo in scene_coords.items():
        pad = 0.01  # ~1km box around the center point
        try:
            passes = search_same_track_passes(
                geo["center_lat"] - pad, geo["center_lat"] + pad,
                geo["center_lon"] - pad, geo["center_lon"] + pad,
                reference_date_iso=None,
            )
            results[scene_id] = len(passes)
        except Exception as e:
            results[scene_id] = f"ERROR: {e}"
        time.sleep(0.2)
    return results


def compare_shapes(polygon_a_px, polygon_b_px, pixel_scale_deg):
    """
    Coarse centroid-drift + area-change comparison between two detected
    polygons (as pixel-coordinate point lists from the same-shaped scene).
    Returns dict with centroid_drift_m and area_change_pct. Flags
    'static_across_passes' if drift is small and area is stable -- thresholds
    are NOT validated against real data yet (no second-pass data exists to
    validate against) and should be treated as a starting point.
    """
    import numpy as np

    def centroid_and_area(poly):
        pts = np.array(poly)
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        cx, cy = pts.mean(axis=0)
        return (cx, cy), area

    (cx_a, cy_a), area_a = centroid_and_area(polygon_a_px)
    (cx_b, cy_b), area_b = centroid_and_area(polygon_b_px)

    drift_px = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
    drift_m = drift_px * pixel_scale_deg * 111_000  # deg -> m at equator, approximate

    area_change_pct = abs(area_a - area_b) / max(area_a, 1) * 100

    # Placeholder thresholds -- flagged as unvalidated in the module docstring.
    is_static = drift_m < 50 and area_change_pct < 20

    return {
        "centroid_drift_m": drift_m,
        "area_change_pct": area_change_pct,
        "flag": "static_across_passes" if is_static else None,
    }
