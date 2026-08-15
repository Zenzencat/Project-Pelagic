"""
Project Pelagic — v1 vs v2 Checkpoint Comparison
SWU Prasarnmit AI Engineering Final Project

Runs both checkpoints over the full 30-scene holdout set using the same shared
tiled-inference path (src/inference.py) and exports two artifacts under docs/:

  1. checkpoint_comparison_summary.{json,csv} — matched, directly-plottable rows
     (metric, v1_value, v2_value, unit) for the capstone comparison chart. Includes
     per-category IoU/Dice AND raw-pixel false-positive area, since IoU/Dice floor
     to exactly 0 on any negative scene (target sum == 0) the instant a single false
     positive pixel exists, which hides the real, non-binary improvement in how much
     area v2 misclassifies vs v1 on lookalike/no-oil scenes.

  2. holdout_per_scene_results.{json,csv} — per-scene confusion-style breakdown
     (tp/fp/fn pixel counts + outcome bucket) for both checkpoints.

Outcome bucketing (a judgment call, not a scientific threshold — flagged as such):
  - Negative scenes (no_oil / lookalike, empty ground truth): "correct" if zero
    false-positive pixels, else "false_positive".
  - Positive scenes (oil): "correct" if IoU >= 0.5, "partial" if some true-positive
    overlap exists but IoU < 0.5, "false_negative" if zero true-positive pixels.
"""

import os
import sys
import glob
import json
import csv
import time

import numpy as np
import tifffile
import torch

sys.path.append(os.path.abspath("."))
from src.models.unet import UNet
from src.data.preprocess import speckle_filter, normalize_image
from src.inference import run_tiled_inference
from src.analysis.geoutils import get_scene_geolocation

KM_PER_DEG_LAT = 111.32


def pixel_area_km2(center_lat_deg, pixel_scale_deg):
    """
    Real ground area per pixel, using the scene's own embedded GeoTIFF pixel
    scale (not a mock display constant) and correcting for the fact that a
    degree of longitude is shorter than a degree of latitude away from the
    equator (factor cos(latitude)). This matters here specifically: the 30
    holdout scenes span roughly -5 to 60 degrees latitude (Sunda Strait to the
    Baltic), where that correction is a ~2x effect, not a rounding error.
    """
    import math
    km_per_deg_lon = KM_PER_DEG_LAT * math.cos(math.radians(center_lat_deg))
    return (pixel_scale_deg * KM_PER_DEG_LAT) * (pixel_scale_deg * km_per_deg_lon)

CHECKPOINTS = {"v1": "model_real_best.pt", "v2": "model_real_v2_best.pt"}
CATEGORIES = ["oil", "no_oil", "lookalike"]


def load_model(checkpoint_name, device):
    model = UNet(in_channels=2, out_channels=1)
    checkpoint = torch.load(os.path.join("checkpoints", checkpoint_name), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def preprocess(image_raw):
    linear = 10.0 ** (image_raw.astype(np.float32) / 10.0)
    filt = speckle_filter(linear, window_size=5)
    db = 10.0 * np.log10(np.clip(filt, 1e-5, None))
    return normalize_image(db)


def scene_category(scene_id):
    for cat in CATEGORIES:
        if scene_id.startswith(cat):
            return cat
    return None


def classify_outcome(category, tp, fp, fn, iou):
    if category != "oil":
        return "correct" if fp == 0 else "false_positive"
    if tp == 0:
        return "false_negative"
    return "correct" if iou >= 0.5 else "partial"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device: {device}")

    models = {v: load_model(name, device) for v, name in CHECKPOINTS.items()}

    img_paths = sorted(glob.glob(os.path.join("data", "holdout", "images", "*.tif")))
    print(f"[*] Found {len(img_paths)} holdout scenes")

    per_scene_rows = []
    t_start = time.time()

    for idx, img_path in enumerate(img_paths):
        scene_id = os.path.splitext(os.path.basename(img_path))[0]
        cat = scene_category(scene_id)
        if cat is None:
            print(f"  [!] skipping {scene_id}: doesn't match any known category prefix")
            continue

        mask_path = os.path.join("data", "holdout", "masks", os.path.basename(img_path))
        image_raw = tifffile.imread(img_path)
        mask_raw = tifffile.imread(mask_path)
        target = (mask_raw > 0).astype(np.uint8)
        norm = preprocess(image_raw)

        geo = get_scene_geolocation(img_path)
        px_area_km2 = pixel_area_km2(geo["center_lat"], geo["pixel_scale_deg"])

        row = {"scene_id": scene_id, "category": cat, "gt_positive_px": int(target.sum()),
               "center_lat": geo["center_lat"], "center_lon": geo["center_lon"],
               "pixel_area_km2_at_scene_latitude": round(px_area_km2, 8)}

        for v, model in models.items():
            t0 = time.time()
            _, pred = run_tiled_inference(model, norm, device)
            elapsed = time.time() - t0

            tp = int(np.sum((pred == 1) & (target == 1)))
            fp = int(np.sum((pred == 1) & (target == 0)))
            fn = int(np.sum((pred == 0) & (target == 1)))

            if target.sum() == 0:
                iou = 1.0 if fp == 0 else 0.0
                dice = 1.0 if fp == 0 else 0.0
            else:
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                dice = 2.0 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

            outcome = classify_outcome(cat, tp, fp, fn, iou)

            row[f"{v}_tp_px"] = tp
            row[f"{v}_fp_px"] = fp
            row[f"{v}_fn_px"] = fn
            row[f"{v}_pred_positive_px"] = tp + fp
            row[f"{v}_iou"] = round(iou, 4)
            row[f"{v}_dice"] = round(dice, 4)
            row[f"{v}_fp_area_km2"] = round(fp * px_area_km2, 5)
            row[f"{v}_outcome"] = outcome

            print(f"  [{idx+1}/{len(img_paths)}] {scene_id:16s} [{v}] iou={iou:.3f} fp_px={fp:>9d} "
                  f"outcome={outcome:15s} ({elapsed:.1f}s)")

        per_scene_rows.append(row)

    print(f"\n[+] Done in {time.time()-t_start:.1f}s total")

    docs_dir = os.path.join("docs")
    os.makedirs(docs_dir, exist_ok=True)

    # --- Export 1: per-scene results ---
    per_scene_json_path = os.path.join(docs_dir, "holdout_per_scene_results.json")
    with open(per_scene_json_path, "w") as f:
        json.dump(per_scene_rows, f, indent=2)

    per_scene_csv_path = os.path.join(docs_dir, "holdout_per_scene_results.csv")
    if per_scene_rows:
        fieldnames = list(per_scene_rows[0].keys())
        with open(per_scene_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_scene_rows)
    print(f"[+] Wrote {per_scene_json_path} and {per_scene_csv_path}")

    # --- Export 2: matched comparison summary (plottable rows) ---
    summary_rows = []

    for cat in CATEGORIES:
        cat_rows = [r for r in per_scene_rows if r["category"] == cat]
        if not cat_rows:
            continue
        for metric, unit in [("iou", "ratio"), ("dice", "ratio")]:
            v1_avg = float(np.mean([r[f"v1_{metric}"] for r in cat_rows]))
            v2_avg = float(np.mean([r[f"v2_{metric}"] for r in cat_rows]))
            summary_rows.append({
                "metric": f"{cat}_{metric}",
                "v1_value": round(v1_avg, 4),
                "v2_value": round(v2_avg, 4),
                "unit": unit,
                "n_scenes": len(cat_rows)
            })

        # Raw-pixel false-positive area — the non-binary-floored view. Uses each
        # scene's own real, latitude-corrected pixel area (pixel_area_km2()
        # above) — no longer the old 0.00015 deg/pixel demo-display mock.
        v1_fp_area_avg = float(np.mean([r["v1_fp_area_km2"] for r in cat_rows]))
        v2_fp_area_avg = float(np.mean([r["v2_fp_area_km2"] for r in cat_rows]))
        summary_rows.append({
            "metric": f"{cat}_avg_false_positive_area_km2",
            "v1_value": round(v1_fp_area_avg, 3),
            "v2_value": round(v2_fp_area_avg, 3),
            "unit": "km2_per_scene (real GSD, latitude-corrected)",
            "n_scenes": len(cat_rows)
        })

        v1_fp_px_avg = float(np.mean([r["v1_fp_px"] for r in cat_rows]))
        v2_fp_px_avg = float(np.mean([r["v2_fp_px"] for r in cat_rows]))
        summary_rows.append({
            "metric": f"{cat}_avg_false_positive_pixels",
            "v1_value": round(v1_fp_px_avg, 1),
            "v2_value": round(v2_fp_px_avg, 1),
            "unit": "pixels_per_scene (scene = 2048x2048px)",
            "n_scenes": len(cat_rows)
        })

        scene_total_px = 2048 * 2048
        v1_pct = float(np.mean([r["v1_fp_px"] / scene_total_px * 100 for r in cat_rows]))
        v2_pct = float(np.mean([r["v2_fp_px"] / scene_total_px * 100 for r in cat_rows]))
        summary_rows.append({
            "metric": f"{cat}_avg_false_positive_pct_of_scene",
            "v1_value": round(v1_pct, 3),
            "v2_value": round(v2_pct, 3),
            "unit": "percent_of_scene_area",
            "n_scenes": len(cat_rows)
        })

    summary_json_path = os.path.join(docs_dir, "checkpoint_comparison_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    summary_csv_path = os.path.join(docs_dir, "checkpoint_comparison_summary.csv")
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "v1_value", "v2_value", "unit", "n_scenes"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[+] Wrote {summary_json_path} and {summary_csv_path}")

    print("\n=== SUMMARY ===")
    for row in summary_rows:
        print(f"  {row['metric']:40s} v1={row['v1_value']:>12} v2={row['v2_value']:>12} ({row['unit']})")


if __name__ == "__main__":
    main()
