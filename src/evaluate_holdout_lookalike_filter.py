"""
Project Pelagic — Phase 3/4: Evaluate the post-hoc lookalike classifier on
the real 30-scene holdout set.

This is the actual test the whole investigation has been building toward
(docs/status.md's ESSD/PANGAEA + own-domain-classifier sections). Runs the
real v2 U-Net exactly as src/compare_checkpoints.py already does (same
tiled-inference path, same before-filter outcome bucketing:
  - negative scenes (no_oil/lookalike): "correct" if zero FP pixels, else
    "false_positive"
  - positive scenes (oil): "correct" if IoU>=0.5, "partial" if some TP but
    IoU<0.5, "false_negative" if zero TP
so the "before" numbers here should reproduce docs/confusion_matrix_breakdown.json's
already-documented v2 baseline exactly, not just approximately), then applies
src/analysis/lookalike_filter.py's classify_and_filter() -- the SAME real,
saved code path src/api/main.py's /api/predict calls when a caller opts in
(Phase 4) -- as an ADDITIVE filter on the U-Net's own largest predicted
connected component.

Never touches any training data -- this is read-only evaluation on the 30
holdout scenes reserved for exactly this since Phase 0.
"""
import os
import sys
import glob
import json
import csv
import argparse

import numpy as np
import tifffile
import torch

sys.path.append(os.path.abspath("."))
from src.models.unet import UNet
from src.data.preprocess import speckle_filter, normalize_image
from src.inference import run_tiled_inference
from src.analysis.lookalike_filter import classify_and_filter, GATE_THRESHOLD

CATEGORIES = ["oil", "no_oil", "lookalike"]


def load_unet(device):
    model = UNet(in_channels=2, out_channels=1)
    checkpoint = torch.load(os.path.join("checkpoints", "model_real_v2_best.pt"), map_location=device)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
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


def compute_outcome_row(pred, target, category):
    tp = int(np.sum((pred == 1) & (target == 1)))
    fp = int(np.sum((pred == 1) & (target == 0)))
    fn = int(np.sum((pred == 0) & (target == 1)))
    if target.sum() == 0:
        iou = 1.0 if fp == 0 else 0.0
    else:
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    outcome = classify_outcome(category, tp, fp, fn, iou)
    return {"tp": tp, "fp": fp, "fn": fn, "iou": round(iou, 4), "outcome": outcome}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", default=None,
                         help="Path to a classifier .joblib (default: lookalike_filter.py's own default, "
                              "checkpoints/lookalike_classifier_final.joblib)")
    parser.add_argument("--gate", action="store_true",
                         help="Apply the Phase 3.9 confidence gate (default: off, i.e. classifier verdict always used)")
    parser.add_argument("--gate-threshold", type=float, default=GATE_THRESHOLD,
                         help=f"Confidence gate threshold (default: {GATE_THRESHOLD}, the exact Phase 3.9 value)")
    parser.add_argument("--out-prefix", default="phase3_holdout_with_filter",
                         help="Prefix for docs/ output files")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device: {device}")

    model = load_unet(device)
    print("[+] Loaded model_real_v2_best.pt")
    print(f"[*] Confidence gate: {'ON' if args.gate else 'OFF'}"
          + (f" (threshold={args.gate_threshold})" if args.gate else ""))

    img_paths = sorted(glob.glob(os.path.join("data", "holdout", "images", "*.tif")))
    print(f"[*] Found {len(img_paths)} holdout scenes")

    # A gate threshold of 1.01 (>1.0, impossible for any probability) makes
    # the gate a structural no-op without special-casing the call below --
    # classify_and_filter's own gate condition (`> threshold`) can never fire.
    effective_gate_threshold = args.gate_threshold if args.gate else 1.01

    rows = []
    for idx, img_path in enumerate(img_paths):
        scene_id = os.path.splitext(os.path.basename(img_path))[0]
        cat = scene_category(scene_id)
        if cat is None:
            continue

        mask_path = os.path.join("data", "holdout", "masks", os.path.basename(img_path))
        image_raw = tifffile.imread(img_path).astype(np.float32)
        mask_raw = tifffile.imread(mask_path)
        target = (mask_raw > 0).astype(np.uint8)
        norm = preprocess(image_raw)

        probs_before, pred_before = run_tiled_inference(model, norm, device)
        before = compute_outcome_row(pred_before, target, cat)

        pred_after, info = classify_and_filter(
            image_raw, probs_before, pred_before,
            classifier_path=args.classifier, gate_threshold=effective_gate_threshold)

        after = compute_outcome_row(pred_after, target, cat)

        row = {"scene_id": scene_id, "category": cat,
               "before_outcome": before["outcome"], "before_iou": before["iou"],
               "before_fp_px": before["fp"], "before_tp_px": before["tp"],
               "classifier_verdict": info["verdict"] if info["applied"] else info["reason"],
               "proba_oil": info["proba_oil"], "mean_unet_confidence": info["mean_unet_confidence"],
               "gated": info["gated"],
               "after_outcome": after["outcome"], "after_iou": after["iou"],
               "after_fp_px": after["fp"], "after_tp_px": after["tp"]}
        rows.append(row)
        print(f"  [{idx+1}/{len(img_paths)}] {scene_id:16s} before={before['outcome']:15s} "
              f"verdict={str(row['classifier_verdict']):22s} gated={info['gated']!s:5s} after={after['outcome']:15s}")

    os.makedirs("docs", exist_ok=True)
    with open(os.path.join("docs", f"{args.out_prefix}_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join("docs", f"{args.out_prefix}_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 70)
    print("BEFORE vs. AFTER per-category outcome breakdown")
    print("=" * 70)
    summary = {}
    for cat in CATEGORIES:
        cat_rows = [r for r in rows if r["category"] == cat]
        before_counts = {}
        after_counts = {}
        for r in cat_rows:
            before_counts[r["before_outcome"]] = before_counts.get(r["before_outcome"], 0) + 1
            after_counts[r["after_outcome"]] = after_counts.get(r["after_outcome"], 0) + 1
        summary[cat] = {"before": before_counts, "after": after_counts, "n": len(cat_rows)}
        print(f"\n{cat} (n={len(cat_rows)}):")
        print(f"  before: {before_counts}")
        print(f"  after:  {after_counts}")

    regressions = [r for r in rows if r["category"] == "oil" and r["before_outcome"] != r["after_outcome"]]
    print(f"\n[Regression check] Oil-bucket scenes whose outcome changed after filtering: {len(regressions)}")
    for r in regressions:
        print(f"  {r['scene_id']}: {r['before_outcome']} -> {r['after_outcome']} "
              f"(verdict={r['classifier_verdict']}, gated={r['gated']})")

    with open(os.path.join("docs", f"{args.out_prefix}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Wrote docs/{args.out_prefix}_results.{{json,csv}} and docs/{args.out_prefix}_summary.json")


if __name__ == "__main__":
    main()
