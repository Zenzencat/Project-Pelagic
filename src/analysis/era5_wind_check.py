"""Optional ERA5 10 m wind evidence for live acquisitions.

The live API exposes measured components, speed and provenance only. The older
standalone heuristic below is retained for reproducibility and is NOT used by
the live pipeline. Holdout scenes still need externally supplied timestamps.
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


DATASET = "reanalysis-era5-single-levels"
RESOLUTION_NOTE = "ERA5 hourly reanalysis on a 0.25° grid (~28 km north-south); not local wind at Sentinel-1 pixel resolution."


def wind_magnitude(u10, v10):
    import math
    if not all(math.isfinite(v) for v in (u10, v10)):
        raise ValueError("ERA5 wind components must be finite")
    return math.hypot(u10, v10)


def utc_datetime(value):
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Acquisition datetime must include its UTC offset")
    return dt.astimezone(timezone.utc)


def extract_wind_record(ds, lat, lon, hour):
    """Reject mismatched times, distant cells, missing values and ambiguous arrays."""
    import numpy as np
    time_key = "valid_time" if "valid_time" in ds.coords else "time"
    expected = np.datetime64(hour.replace(tzinfo=None), "ns")
    # Exact time selection: a different hour must never masquerade as requested data.
    point = ds.sel({time_key: expected})
    longitudes = np.asarray(ds.longitude.values)
    query_lon = lon % 360 if np.all(longitudes >= 0) else (lon + 180) % 360 - 180
    point = point.sel(latitude=lat, longitude=query_lon, method="nearest")
    cell_lat, cell_lon = float(point.latitude.values), float(point.longitude.values)
    lon_distance = abs((cell_lon - lon + 180) % 360 - 180)
    if not np.isfinite([cell_lat, cell_lon]).all() or abs(cell_lat - lat) > 0.26 or lon_distance > 0.26:
        raise ValueError("ERA5 returned a grid cell outside the requested location")
    values = []
    for name in ("u10", "v10"):
        variable = point[name]
        if variable.attrs.get("units") not in ("m s**-1", "m s-1", "m/s"):
            raise ValueError("ERA5 wind units missing or unsupported")
        arr = np.asarray(variable.values)
        if arr.size != 1:
            raise ValueError("ERA5 returned ambiguous wind components")
        values.append(float(arr.item()))
    u10, v10 = values
    return {"u10_ms": u10, "v10_ms": v10, "wind_speed_ms": wind_magnitude(u10, v10),
            "valid_time_utc": hour.isoformat(), "grid_lat": cell_lat,
            "grid_lon": (cell_lon + 180) % 360 - 180}


def _retrieve_wind(lat, lon, timestamp_iso, cds_client=None):
    from datetime import timedelta
    import tempfile
    import xarray as xr
    dt = utc_datetime(timestamp_iso)
    hour = (dt + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
    client = cds_client or cdsapi.Client(timeout=15, retry_max=1, quiet=True, progress=False)
    # Clamp at the dateline to keep the CDS bounding box valid.
    west, east = max(-180, lon - 0.3), min(180, lon + 0.3)
    with tempfile.TemporaryDirectory(prefix="pelagic-era5-") as directory:
        from pathlib import Path
        target = str(Path(directory) / "wind.nc")
        client.retrieve(DATASET, {
            "product_type": ["reanalysis"],
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "year": [f"{hour.year:04d}"], "month": [f"{hour.month:02d}"],
            "day": [f"{hour.day:02d}"], "time": [f"{hour.hour:02d}:00"],
            "area": [min(90, lat + 0.3), west, max(-90, lat - 0.3), east],
            "grid": [0.25, 0.25], "data_format": "netcdf", "download_format": "unarchived",
        }, target)
        with xr.open_dataset(target) as ds:
            return extract_wind_record(ds, lat, lon, hour)


def _wind_worker(connection, lat, lon, timestamp):
    try:
        connection.send({"status": "available", **_retrieve_wind(lat, lon, timestamp)})
    except Exception as exc:
        # Service exceptions may contain credentials/URLs. Do not serialize them.
        connection.send({"status": "unavailable", "reason": f"ERA5 retrieval or response validation failed ({type(exc).__name__})."})
    finally:
        connection.close()


def get_wind_evidence(lat, lon, timestamp_iso, *, timeout_seconds=60):
    """Optional bounded retrieval. CDS queue delays cannot hold up inference forever."""
    import math
    import os
    from pathlib import Path
    base = {"source": DATASET, "resolution_note": RESOLUTION_NOTE,
            "requested_lat": lat, "requested_lon": lon, "acquisition_time_utc": timestamp_iso}
    try:
        utc_datetime(timestamp_iso)
        if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Invalid location")
    except (ValueError, TypeError, AttributeError):
        return {**base, "status": "skipped", "reason": "Real acquisition datetime with UTC offset and valid location required."}
    rc = Path(os.environ.get("CDSAPI_RC", str(Path.home() / ".cdsapirc")))
    if not ((os.environ.get("CDSAPI_KEY") and os.environ.get("CDSAPI_URL")) or rc.is_file()):
        return {**base, "status": "not_configured", "reason": "Configure CDSAPI_URL and CDSAPI_KEY (or .cdsapirc)."}
    if not CDSAPI_AVAILABLE:
        return {**base, "status": "unavailable", "reason": "Optional cdsapi dependency is not installed."}
    import multiprocessing
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_wind_worker, args=(child, lat, lon, timestamp_iso), daemon=True)
    try:
        process.start()
        child.close()
        if parent.poll(timeout_seconds):
            return {**base, **parent.recv()}
        return {**base, "status": "unavailable", "reason": f"ERA5 request exceeded the {timeout_seconds}-second retrieval budget; CDS may still be queuing the request."}
    except Exception as exc:
        return {**base, "status": "unavailable", "reason": f"ERA5 worker failed ({type(exc).__name__})."}
    finally:
        if process.is_alive():
            process.terminate()
        if process.pid is not None:
            process.join(timeout=2)
        parent.close()
        child.close()


def fetch_era5_wind_speed(lat, lon, timestamp_iso, cds_client=None):
    """Compatibility adapter for the standalone analysis; no synthetic fallback."""
    return _retrieve_wind(lat, lon, timestamp_iso, cds_client)["wind_speed_ms"]


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
