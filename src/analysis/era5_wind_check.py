"""
Project Pelagic — ERA5 Wind Cross-Check (standalone analysis, NOT wired into the
live pipeline)
SWU Prasarnmit AI Engineering Final Project

Oil slicks dampen capillary waves and only produce a visible dark SAR signature
within roughly 1.5-6 m/s surface wind speed: below that, the sea is already too
calm for wind roughness to contrast against (a "false-dark" look-alike risk);
above it, wind-driven roughening breaks up the slick's dampening signature and
also produces its own dark low-backscatter patches (wind shadow zones, current
fronts) that mimic oil. This module cross-checks each detected polygon's location
against the ERA5 10m wind speed at acquisition time and flags (does not suppress)
detections outside that window as `low_confidence_wind_outlier`, with the actual
wind speed attached.

STATUS: the fetch/classify logic below is real, working code against the
documented CDS API. It has NOT been run against real data yet because:
  1. No CDS API key is configured in this environment (no ~/.cdsapirc, no
     CDSAPI_KEY/CDSAPI_URL env vars, `cdsapi` package not installed). Register
     for free at https://cds.climate.copernicus.eu, accept the ERA5
     single-levels dataset license, then either write
     ~/.cdsapirc as documented at https://cds.climate.copernicus.eu/how-to-api,
     or set CDSAPI_URL / CDSAPI_KEY env vars.
  2. More fundamentally: the 30 holdout scenes have real embedded lat/lon
     (src/analysis/geoutils.py) but NO acquisition timestamp anywhere -- not in
     the GeoTIFF tags, not as a companion file on any of the three Zenodo
     records (verified directly via the Zenodo API's file listing for
     8346860 / 8253899 / 13761290: each deposit is just the image + mask .7z
     archives, nothing else). Without a timestamp there is no way to pick which
     ERA5 timestep to query, for this dataset specifically. See docs/status.md.

Do not fabricate timestamps to make this runnable -- if you have real
acquisition times (e.g. from a source outside this repo), pass them in via
`--timestamps-json` (see `main()` below) rather than editing them into this
file.
"""

import argparse
import json
import sys

try:
    import cdsapi
    CDSAPI_AVAILABLE = True
except ImportError:
    CDSAPI_AVAILABLE = False

# Slick-visibility window in m/s. This is a standard heuristic in SAR oil-spill
# literature, not something calibrated against this specific model or dataset --
# treat it as a starting point, not a validated threshold.
WIND_SPEED_MIN_MS = 1.5
WIND_SPEED_MAX_MS = 6.0


def fetch_era5_wind_speed(lat, lon, timestamp_iso, cds_client=None):
    """
    Fetches ERA5 10m wind speed (u/v components combined) for a given point and
    time via the CDS API.

    Args:
        lat, lon: decimal degrees, WGS84.
        timestamp_iso: ISO 8601 datetime string, e.g. "2024-01-12T03:42:20".
                        ERA5 is hourly, so this is rounded to the nearest hour.
        cds_client: an existing cdsapi.Client(), or None to create one (requires
                    ~/.cdsapirc or CDSAPI_URL/CDSAPI_KEY env vars to already be
                    configured -- raises cdsapi's own error if not).

    Returns:
        float wind speed in m/s at that point/time.

    Raises:
        RuntimeError if the cdsapi package isn't installed.
        Whatever cdsapi/requests raises on auth failure or a bad request.
    """
    if not CDSAPI_AVAILABLE:
        raise RuntimeError(
            "cdsapi is not installed. Run: pip install cdsapi "
            "(see https://github.com/ecmwf/cdsapi)"
        )

    from datetime import datetime
    dt = datetime.fromisoformat(timestamp_iso)

    client = cds_client or cdsapi.Client()

    # CDS subsets by bounding box, not a literal point -- request a small box
    # around the target and read the single nearest grid cell from the result.
    # ERA5 single-levels is ~0.25deg native resolution, so 0.3deg padding on
    # each side comfortably covers one grid cell regardless of point position.
    pad = 0.3
    area = [lat + pad, lon - pad, lat - pad, lon + pad]  # N, W, S, E

    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        target_path = tmp.name

    try:
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                "year": f"{dt.year:04d}",
                "month": f"{dt.month:02d}",
                "day": f"{dt.day:02d}",
                "time": f"{dt.hour:02d}:00",
                "area": area,
                "format": "netcdf",
            },
            target_path,
        )

        import numpy as np
        try:
            import xarray as xr
        except ImportError:
            raise RuntimeError(
                "xarray is required to read the ERA5 NetCDF response. "
                "Run: pip install xarray netCDF4"
            )

        ds = xr.open_dataset(target_path)
        u10 = ds["u10"].sel(latitude=lat, longitude=lon % 360, method="nearest").values
        v10 = ds["v10"].sel(latitude=lat, longitude=lon % 360, method="nearest").values
        speed = float(np.sqrt(u10**2 + v10**2))
        ds.close()
        return speed
    finally:
        if os.path.exists(target_path):
            os.remove(target_path)


def classify_wind_confidence(wind_speed_ms, low=WIND_SPEED_MIN_MS, high=WIND_SPEED_MAX_MS):
    """
    Returns (flag, reason) -- flag is "low_confidence_wind_outlier" or None.
    Never silently suppresses; always returns the actual wind speed for
    inspection alongside the flag.
    """
    if wind_speed_ms < low:
        return "low_confidence_wind_outlier", f"wind {wind_speed_ms:.2f} m/s < {low} m/s (likely dead-calm, not oil dampening)"
    if wind_speed_ms > high:
        return "low_confidence_wind_outlier", f"wind {wind_speed_ms:.2f} m/s > {high} m/s (likely wind-roughened false dark patch)"
    return None, f"wind {wind_speed_ms:.2f} m/s within {low}-{high} m/s slick-visibility window"


def run_analysis(scene_coords, scene_timestamps, ground_truth_outcomes, cds_client=None):
    """
    scene_coords: {scene_id: {"center_lat": .., "center_lon": ..}} (from geoutils)
    scene_timestamps: {scene_id: iso_timestamp_str} -- caller-supplied, real data only.
    ground_truth_outcomes: {scene_id: "correct"|"false_positive"|"partial"|"false_negative"}
                            (from docs/holdout_per_scene_results.json's v2_outcome column)

    Returns a list of per-scene result dicts, and prints a precision/recall-style
    breakdown of how the wind flag lines up with actual v2 false positives.
    """
    results = []
    for scene_id, geo in scene_coords.items():
        if scene_id not in scene_timestamps:
            results.append({"scene_id": scene_id, "status": "no_timestamp_available"})
            continue
        try:
            speed = fetch_era5_wind_speed(
                geo["center_lat"], geo["center_lon"], scene_timestamps[scene_id], cds_client
            )
            flag, reason = classify_wind_confidence(speed)
            results.append({
                "scene_id": scene_id,
                "status": "ok",
                "wind_speed_ms": speed,
                "flag": flag,
                "reason": reason,
                "ground_truth_outcome": ground_truth_outcomes.get(scene_id),
            })
        except Exception as e:
            results.append({"scene_id": scene_id, "status": "fetch_error", "error": str(e)})

    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        tp = sum(1 for r in ok_results if r["flag"] and r["ground_truth_outcome"] == "false_positive")
        fn = sum(1 for r in ok_results if not r["flag"] and r["ground_truth_outcome"] == "false_positive")
        fp = sum(1 for r in ok_results if r["flag"] and r["ground_truth_outcome"] == "correct")
        tn = sum(1 for r in ok_results if not r["flag"] and r["ground_truth_outcome"] == "correct")
        print(f"\nWind-flag vs ground truth (of {len(ok_results)} scenes with a fetched wind speed):")
        print(f"  Caught real false positives (flagged AND was a FP):      {tp}")
        print(f"  Missed real false positives (not flagged, was a FP):     {fn}")
        print(f"  Wrongly flagged correct detections (flagged, was correct): {fp}")
        print(f"  Correctly left correct detections alone:                 {tn}")
    else:
        print("\nNo scenes had both a timestamp and a successful ERA5 fetch -- nothing to report yet.")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coords-json", default="docs/holdout_scene_coordinates.json")
    parser.add_argument("--outcomes-json", default="docs/holdout_per_scene_results.json",
                         help="Uses the v2_outcome field per scene as ground truth.")
    parser.add_argument("--timestamps-json", default=None,
                         help="REQUIRED to actually run: JSON file of {scene_id: iso_timestamp}. "
                              "No default exists because no real timestamps are available for "
                              "this dataset -- see the module docstring.")
    args = parser.parse_args()

    with open(args.coords_json) as f:
        scene_coords = json.load(f)

    with open(args.outcomes_json) as f:
        per_scene = json.load(f)
    ground_truth_outcomes = {r["scene_id"]: r["v2_outcome"] for r in per_scene}

    if args.timestamps_json is None:
        print(
            "[!] No --timestamps-json supplied. This dataset has no acquisition "
            "timestamps available (see module docstring) -- refusing to fabricate "
            "any. Supply real timestamps via --timestamps-json to run this for real.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(args.timestamps_json) as f:
        scene_timestamps = json.load(f)

    if not CDSAPI_AVAILABLE:
        print("[!] cdsapi not installed. Run: pip install cdsapi", file=sys.stderr)
        sys.exit(1)

    run_analysis(scene_coords, scene_timestamps, ground_truth_outcomes)


if __name__ == "__main__":
    main()
