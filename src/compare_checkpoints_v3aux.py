"""
Project Pelagic -- Phase 3: v1/v2/v3/v3-aux-0.3/v3-aux-1.0 Checkpoint Comparison

Extends src/compare_checkpoints_v3.py's pattern (itself a fork of
src/compare_checkpoints.py) rather than rewriting from scratch: same
preprocessing, same tiled-inference-based segmentation metrics, same
classify_outcome() bucketing rule, same real 30-scene holdout, same output
file format (with a few extra columns). v1/v2/v3 use the plain UNet +
run_tiled_inference (unchanged, single logits tensor). The two aux models
use JointUNet + run_tiled_inference_joint (returns segmentation AND a
scene-level oil-vs-lookalike classification), since their forward() returns
a (seg_logits, cls_logits) tuple that the plain run_tiled_inference() can't
handle.

Aux classification accuracy is evaluated ONLY on the 20 oil+lookalike
holdout scenes (scene_id prefixes "oil_"/"lookalike_") -- per Phase 3 Step 0,
the aux head's label space is binary (oil vs. lookalike only); no_oil scenes
have no ground-truth label for it and are excluded from that specific metric,
though they still go through segmentation evaluation identically to every
other category.
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
from src.models.joint_unet import JointUNet
from src.data.preprocess import speckle_filter, normalize_image
from src.inference import run_tiled_inference
from src.joint_inference import run_tiled_inference_joint
from src.analysis.geoutils import get_scene_geolocation

KM_PER_DEG_LAT = 111.32

PLAIN_CHECKPOINTS = {"v1": "model_real_best.pt", "v2": "model_real_v2_best.pt", "v3": "model_real_v3_best.pt"}
AUX_CHECKPOINTS = {"v3-aux-0.3": "model_real_v3_aux03_best.pt", "v3-aux-1.0": "model_real_v3_aux10_best.pt"}
ALL_VERSIONS = list(PLAIN_CHECKPOINTS.keys()) + list(AUX_CHECKPOINTS.keys())
CATEGORIES = ["oil", "no_oil", "lookalike"]

# scene_id prefix -> binary aux ground-truth label. no_oil has none (excluded).
CLS_GROUND_TRUTH = {"oil": 1, "lookalike": 0}


def pixel_area_km2(center_lat_deg, pixel_scale_deg):
    import math
    km_per_deg_lon = KM_PER_DEG_LAT * math.cos(math.radians(center_lat_deg))
    return (pixel_scale_deg * KM_PER_DEG_LAT) * (pixel_scale_deg * km_per_deg_lon)


def load_plain_model(checkpoint_name, device):
    model = UNet(in_channels=2, out_channels=1)
    checkpoint = torch.load(os.path.join("checkpoints", checkpoint_name), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def load_joint_model(checkpoint_name, device):
    model = JointUNet(in_channels=2, out_channels=1, num_classes=2)
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

    plain_models = {v: load_plain_model(name, device) for v, name in PLAIN_CHECKPOINTS.items()}
    aux_models = {v: load_joint_model(name, device) for v, name in AUX_CHECKPOINTS.items()}

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
               "pixel_area_km2_at_scene_latitude": round(px_area_km2, 8),
               "cls_ground_truth": CLS_GROUND_TRUTH.get(cat, "")}

        # --- Plain models: v1, v2, v3 ---
        for v, model in plain_models.items():
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

            precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if target.sum() == 0 and fp == 0 else 0.0)
            recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if target.sum() == 0 else 0.0)

            outcome = classify_outcome(cat, tp, fp, fn, iou)

            row[f"{v}_tp_px"] = tp
            row[f"{v}_fp_px"] = fp
            row[f"{v}_fn_px"] = fn
            row[f"{v}_pred_positive_px"] = tp + fp
            row[f"{v}_iou"] = round(iou, 4)
            row[f"{v}_dice"] = round(dice, 4)
            row[f"{v}_precision"] = round(precision, 4)
            row[f"{v}_recall"] = round(recall, 4)
            row[f"{v}_fp_area_km2"] = round(fp * px_area_km2, 5)
            row[f"{v}_outcome"] = outcome

            print(f"  [{idx+1}/{len(img_paths)}] {scene_id:16s} [{v}] iou={iou:.3f} fp_px={fp:>9d} "
                  f"outcome={outcome:15s} ({elapsed:.1f}s)")

        # --- Joint (aux head) models: v3-aux-0.3, v3-aux-1.0 ---
        for v, model in aux_models.items():
            t0 = time.time()
            _, pred, cls_probs, cls_pred = run_tiled_inference_joint(model, norm, device)
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

            precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if target.sum() == 0 and fp == 0 else 0.0)
            recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if target.sum() == 0 else 0.0)

            outcome = classify_outcome(cat, tp, fp, fn, iou)

            safe_v = v.replace(".", "_").replace("-", "_")
            row[f"{safe_v}_tp_px"] = tp
            row[f"{safe_v}_fp_px"] = fp
            row[f"{safe_v}_fn_px"] = fn
            row[f"{safe_v}_pred_positive_px"] = tp + fp
            row[f"{safe_v}_iou"] = round(iou, 4)
            row[f"{safe_v}_dice"] = round(dice, 4)
            row[f"{safe_v}_precision"] = round(precision, 4)
            row[f"{safe_v}_recall"] = round(recall, 4)
            row[f"{safe_v}_fp_area_km2"] = round(fp * px_area_km2, 5)
            row[f"{safe_v}_outcome"] = outcome
            row[f"{safe_v}_cls_pred"] = cls_pred
            row[f"{safe_v}_cls_prob_lookalike"] = round(float(cls_probs[0]), 4)
            row[f"{safe_v}_cls_prob_oil"] = round(float(cls_probs[1]), 4)
            if cat in CLS_GROUND_TRUTH:
                row[f"{safe_v}_cls_correct"] = int(cls_pred == CLS_GROUND_TRUTH[cat])
            else:
                row[f"{safe_v}_cls_correct"] = ""

            print(f"  [{idx+1}/{len(img_paths)}] {scene_id:16s} [{v}] iou={iou:.3f} fp_px={fp:>9d} "
                  f"outcome={outcome:15s} cls_pred={cls_pred} ({elapsed:.1f}s)")

        per_scene_rows.append(row)

    print(f"\n[+] Done in {time.time()-t_start:.1f}s total")

    docs_dir = os.path.join("docs")
    os.makedirs(docs_dir, exist_ok=True)

    per_scene_json_path = os.path.join(docs_dir, "holdout_per_scene_results_v3aux.json")
    with open(per_scene_json_path, "w") as f:
        json.dump(per_scene_rows, f, indent=2)

    per_scene_csv_path = os.path.join(docs_dir, "holdout_per_scene_results_v3aux.csv")
    if per_scene_rows:
        fieldnames = list(per_scene_rows[0].keys())
        with open(per_scene_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_scene_rows)
    print(f"[+] Wrote {per_scene_json_path} and {per_scene_csv_path}")

    # --- Segmentation summary (all 5 versions) ---
    summary_rows = []
    version_keys = {"v1": "v1", "v2": "v2", "v3": "v3",
                     "v3-aux-0.3": "v3_aux_0_3", "v3-aux-1.0": "v3_aux_1_0"}

    for cat in CATEGORIES:
        cat_rows = [r for r in per_scene_rows if r["category"] == cat]
        if not cat_rows:
            continue
        for metric in ("iou", "dice", "precision", "recall"):
            entry = {"metric": f"{cat}_{metric}", "n_scenes": len(cat_rows)}
            for vlabel, vkey in version_keys.items():
                entry[vlabel] = round(float(np.mean([r[f"{vkey}_{metric}"] for r in cat_rows])), 4)
            summary_rows.append(entry)

        entry = {"metric": f"{cat}_avg_false_positive_pixels", "n_scenes": len(cat_rows)}
        for vlabel, vkey in version_keys.items():
            entry[vlabel] = round(float(np.mean([r[f"{vkey}_fp_px"] for r in cat_rows])), 1)
        summary_rows.append(entry)

        for vlabel, vkey in version_keys.items():
            outcomes = [r[f"{vkey}_outcome"] for r in cat_rows]
            counts = {o: outcomes.count(o) for o in set(outcomes)}
            summary_rows.append({
                "metric": f"{cat}_outcome_bucket_counts__{vlabel}",
                "n_scenes": len(cat_rows),
                vlabel: json.dumps(counts),
            })

    # --- Aux classification accuracy (v3-aux-0.3 / v3-aux-1.0 only, oil+lookalike scenes only) ---
    cls_rows = [r for r in per_scene_rows if r["category"] in CLS_GROUND_TRUTH]
    for vlabel in AUX_CHECKPOINTS.keys():
        safe_v = vlabel.replace(".", "_").replace("-", "_")
        n = len(cls_rows)
        n_correct = sum(int(r[f"{safe_v}_cls_correct"]) for r in cls_rows)
        acc = n_correct / n if n > 0 else float("nan")
        summary_rows.append({
            "metric": "aux_head_classification_accuracy_oil_vs_lookalike",
            "n_scenes": n,
            vlabel: round(acc, 4),
        })
        print(f"\n[AUX ACCURACY] {vlabel}: {n_correct}/{n} = {acc:.4f} (oil+lookalike holdout scenes only, n=20)")

    summary_json_path = os.path.join(docs_dir, "checkpoint_comparison_summary_v3aux.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    all_cols = ["metric", "n_scenes", "v1", "v2", "v3", "v3-aux-0.3", "v3-aux-1.0"]
    summary_csv_path = os.path.join(docs_dir, "checkpoint_comparison_summary_v3aux.csv")
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, "") for k in all_cols})
    print(f"[+] Wrote {summary_json_path} and {summary_csv_path}")

    print("\n=== SUMMARY ===")
    for row in summary_rows:
        vals = " ".join(f"{k}={row.get(k, ''):}" for k in ("v1", "v2", "v3", "v3-aux-0.3", "v3-aux-1.0") if k in row)
        print(f"  {row['metric']:55s} {vals}")


if __name__ == "__main__":
    main()
