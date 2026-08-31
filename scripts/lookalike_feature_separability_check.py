"""
Cheap feasibility check, run BEFORE touching the ESSD/PANGAEA dataset (per Zen's
explicit instruction): does hand-crafted texture/shape (GLCM, edge density,
compactness) separate real oil from lookalikes in THIS project's own scenes at
all? If yes, Phase 1 (ESSD download + domain alignment) is worth the investment.
If not, the bottleneck is the feature choice, not the ESSD domain gap.

Data used (no ESSD/PANGAEA data touched):
  - 10 holdout oil + 10 holdout lookalike scenes (data/holdout/), fully local already.
  - 15 training oil + 15 training lookalike scenes, sampled evenly across the real
    685-lookalike / 1200-oil training pools and downloaded fresh from the exact
    Kaggle datasets train_kaggle.py itself trains on (shuddhabrotabanerjee/
    oil-spill-dartis-part1, -part2) -- NOT the local repo's Mask_lookalike-only
    checkout, which (per Phase 0 findings in docs/status.md) only has masks, not
    images, for the training-set lookalikes.

Method: extracts the candidate-region crop from EACH scene by running the
project's REAL v2 U-Net checkpoint (checkpoints/model_real_v2_best.pt, via the
exact shared src/data/preprocess.py::preprocess_for_prediction +
src/inference.py::run_tiled_inference path production code uses) and taking
its largest predicted connected component -- this is not a proxy detector,
it's literally the same candidate-region-producing step Phase 2's post-hoc
classifier will sit downstream of. A scene the model predicts nothing on
(empty mask) has no candidate region and is skipped, which is itself
informative (reported as a skip-rate). Ground-truth oil masks are used only
for a sanity-check IoU against the real per-scene results already documented
in docs/status.md, not to pick the crop.

Outputs: output/lookalike_feature_separability.csv (per-scene features),
output/lookalike_feature_separability_pca.png (2D PCA scatter),
prints quantitative separability metrics to stdout.
"""
import os
import glob
import json

import numpy as np
import cv2
import tifffile
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "output")
os.makedirs(OUT_DIR, exist_ok=True)
import sys
sys.path.append(REPO)
from src.models.unet import UNet
from src.data.preprocess import preprocess_for_prediction
from src.inference import run_tiled_inference
from src.analysis.candidate_region_features import (
    largest_component, extract_region_features, FEATURE_COLUMNS)

MIN_BLOB_AREA = 400  # px, ~0.04 km^2 at 10m GSD -- filters pure speckle noise

_MODEL = None
_DEVICE = None


def get_model():
    global _MODEL, _DEVICE
    if _MODEL is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL = UNet(in_channels=2, out_channels=1)
        ckpt = torch.load(os.path.join(REPO, "checkpoints", "model_real_v2_best.pt"), map_location=_DEVICE)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        _MODEL.load_state_dict(state)
        _MODEL.to(_DEVICE)
        _MODEL.eval()
        print(f"[*] Loaded model_real_v2_best.pt on {_DEVICE}")
    return _MODEL, _DEVICE


def load_raw(path):
    """Loads the full 2-channel (VV, VH) raw scene. Holdout/Kaggle-training
    GeoTIFFs here are already real calibrated sigma0 dB (confirmed: values are
    negative, matches docs/status.md's documented [-25,0] dB clip convention)."""
    arr = tifffile.imread(path)
    return arr.astype(np.float32)


def model_candidate_mask(image_raw):
    """Runs the REAL v2 checkpoint (the exact production predict path) and
    returns its binary predicted mask -- this is the real candidate-region
    proposal a post-hoc classifier would sit downstream of, not a proxy."""
    model, device = get_model()
    norm = preprocess_for_prediction(image_raw, already_calibrated=False)
    probs_full, preds_full = run_tiled_inference(model, norm, device)
    return preds_full


def extract_features(image_raw, gt_mask=None, margin=15):
    """image_raw: full (H, W, 2) raw VV/VH dB scene. Candidate region comes
    from the real v2 U-Net's own prediction, not a heuristic proxy."""
    pred_mask = model_candidate_mask(image_raw)
    result = largest_component(pred_mask, min_area=MIN_BLOB_AREA)
    if result is None:
        return None
    blob_mask, (x, y, w, h), area = result

    vv_db = image_raw[..., 0]
    feats = extract_region_features(vv_db, blob_mask, (x, y, w, h), margin=margin)
    if feats is None:
        return None
    feats["blob_area_px"] = area
    feats["pred_total_area_px"] = int(np.count_nonzero(pred_mask))

    if gt_mask is not None:
        inter = np.count_nonzero((pred_mask > 0) & (gt_mask > 0))
        union = np.count_nonzero((pred_mask > 0) | (gt_mask > 0))
        feats["candidate_vs_gt_iou"] = inter / union if union > 0 else 0.0
    else:
        feats["candidate_vs_gt_iou"] = np.nan

    return feats


def main():
    rows = []

    holdout_img_dir = os.path.join(REPO, "data", "holdout", "images")
    holdout_mask_dir = os.path.join(REPO, "data", "holdout", "masks")
    for cat, n in [("oil", 10), ("lookalike", 10)]:
        for i in range(n):
            sid = f"{cat}_{i:05d}"
            img_path = os.path.join(holdout_img_dir, f"{sid}.tif")
            mask_path = os.path.join(holdout_mask_dir, f"{sid}.tif")
            if not os.path.exists(img_path):
                continue
            raw = load_raw(img_path)
            gt = tifffile.imread(mask_path) if os.path.exists(mask_path) else None
            feats = extract_features(raw, gt)
            if feats is None:
                print(f"[!] Model predicted empty mask for {sid} -- no candidate region, skipped")
                continue
            feats["scene_id"] = sid
            feats["label"] = "oil" if cat == "oil" else "lookalike"
            feats["source"] = "holdout"
            rows.append(feats)
            print(f"[+] {sid}: blob_area={feats['blob_area_px']}px "
                  f"circularity={feats.get('circularity', float('nan')):.3f} "
                  f"iou_vs_gt={feats['candidate_vs_gt_iou']:.3f}")

    sample_idx = {
        "oil": [0, 80, 160, 240, 320, 400, 480, 560, 640, 720, 800, 880, 960, 1040, 1120],
        "lookalike": [0, 45, 90, 135, 180, 225, 270, 315, 360, 405, 450, 495, 540, 585, 630],
    }
    kaggle_oil_dir = os.path.expandvars(r"%LOCALAPPDATA%\Temp\kaggle_sample\oil")
    kaggle_lookalike_dir = os.path.expandvars(r"%LOCALAPPDATA%\Temp\kaggle_sample\lookalike")
    mask_oil_dir = os.path.join(REPO, "data", "raw", "Mask_oil")

    for i in sample_idx["oil"]:
        idx = f"{i:05d}"
        img_path = os.path.join(kaggle_oil_dir, f"{idx}.tif")
        mask_path = os.path.join(mask_oil_dir, f"{idx}.tif")
        if not os.path.exists(img_path):
            print(f"[!] Missing training oil sample {idx} -- skipped")
            continue
        raw = load_raw(img_path)
        gt = tifffile.imread(mask_path) if os.path.exists(mask_path) else None
        feats = extract_features(raw, gt)
        if feats is None:
            print(f"[!] Model predicted empty mask for train_oil_{idx} -- no candidate region, skipped")
            continue
        feats["scene_id"] = f"train_oil_{idx}"
        feats["label"] = "oil"
        feats["source"] = "train_sample"
        rows.append(feats)
        print(f"[+] train_oil_{idx}: blob_area={feats['blob_area_px']}px "
              f"circularity={feats.get('circularity', float('nan')):.3f} "
              f"iou_vs_gt={feats['candidate_vs_gt_iou']:.3f}")

    for i in sample_idx["lookalike"]:
        idx = f"{i:05d}"
        img_path = os.path.join(kaggle_lookalike_dir, f"{idx}.tif")
        if not os.path.exists(img_path):
            print(f"[!] Missing training lookalike sample {idx} -- skipped")
            continue
        raw = load_raw(img_path)
        feats = extract_features(raw, gt_mask=None)
        if feats is None:
            print(f"[!] Model predicted empty mask for train_lookalike_{idx} -- no candidate region (correctly suppressed), skipped")
            continue
        feats["scene_id"] = f"train_lookalike_{idx}"
        feats["label"] = "lookalike"
        feats["source"] = "train_sample"
        rows.append(feats)
        print(f"[+] train_lookalike_{idx}: blob_area={feats['blob_area_px']}px "
              f"circularity={feats.get('circularity', float('nan')):.3f}")

    # ---- Save raw feature table ----
    import csv
    csv_path = os.path.join(OUT_DIR, "lookalike_feature_separability.csv")
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[+] Wrote {len(rows)} scenes' features to {csv_path}")

    # ---- Sanity check: does the real model's own prediction land on real oil? ----
    # (should roughly match the IoU numbers already documented in docs/status.md's
    # confusion-matrix/holdout results, since this reruns the same v2 checkpoint.)
    oil_ious = [r["candidate_vs_gt_iou"] for r in rows if r["label"] == "oil" and not np.isnan(r["candidate_vs_gt_iou"])]
    print(f"\n[Sanity check] Real v2-model-prediction vs. ground-truth IoU for {len(oil_ious)} oil scenes: "
          f"mean={np.mean(oil_ious):.3f}, median={np.median(oil_ious):.3f} "
          f"(should be broadly consistent with this project's already-documented holdout IoU numbers -- "
          f"a large mismatch would mean something in this script's inference call diverged from production)")

    # ---- Build feature matrix for separability analysis ----
    feature_cols = FEATURE_COLUMNS
    X, y, meta = [], [], []
    for r in rows:
        if any(k not in r or r[k] is None or (isinstance(r[k], float) and np.isnan(r[k])) for k in feature_cols):
            continue
        X.append([r[k] for k in feature_cols])
        y.append(1 if r["label"] == "oil" else 0)
        meta.append((r["scene_id"], r["source"]))
    X = np.array(X)
    y = np.array(y)
    print(f"\n[+] {X.shape[0]} scenes with complete features ({y.sum()} oil, {(1 - y).sum()} lookalike)")

    Xs = StandardScaler().fit_transform(X)

    # Per-feature separation: Cohen's d + Welch t-test p-value
    print("\n[Per-feature separation: oil vs. lookalike]")
    print(f"{'feature':<22}{'oil mean':>12}{'lookalike mean':>16}{'cohens_d':>12}{'p_value':>12}")
    for j, col in enumerate(feature_cols):
        oil_vals = X[y == 1, j]
        la_vals = X[y == 0, j]
        pooled_std = np.sqrt((oil_vals.var(ddof=1) + la_vals.var(ddof=1)) / 2)
        cohens_d = (oil_vals.mean() - la_vals.mean()) / pooled_std if pooled_std > 0 else 0.0
        _, p = stats.ttest_ind(oil_vals, la_vals, equal_var=False)
        print(f"{col:<22}{oil_vals.mean():>12.4f}{la_vals.mean():>16.4f}{cohens_d:>12.3f}{p:>12.4f}")

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(Xs)
    explained = pca.explained_variance_ratio_
    print(f"\n[PCA] 2 components explain {explained.sum() * 100:.1f}% of variance "
          f"(PC1={explained[0] * 100:.1f}%, PC2={explained[1] * 100:.1f}%)")

    sil = silhouette_score(X_pca, y) if len(set(y)) > 1 else float("nan")
    print(f"[Cluster separability] Silhouette score (oil vs. lookalike, in 2D PCA space): {sil:.3f} "
          f"(range -1..1; >0.25 = modest-but-real separation, ~0 = heavily overlapping, <0 = worse than random)")

    # Leave-one-out logistic regression as a simple, small-n-honest classifier check
    loo = LeaveOneOut()
    correct = 0
    for train_idx, test_idx in loo.split(Xs):
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(Xs[train_idx], y[train_idx])
        pred = clf.predict(Xs[test_idx])
        correct += int(pred[0] == y[test_idx][0])
    loo_acc = correct / len(y)
    baseline = max(y.mean(), 1 - y.mean())
    print(f"[Classifier check] Leave-one-out logistic regression accuracy: {loo_acc:.3f} "
          f"(n={len(y)}; majority-class baseline={baseline:.3f})")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"oil": "#d62728", "lookalike": "#1f77b4"}
    markers = {"holdout": "o", "train_sample": "^"}
    for label_name, label_val in [("oil", 1), ("lookalike", 0)]:
        for source in ["holdout", "train_sample"]:
            idxs = [k for k, (sid, src) in enumerate(meta)
                    if (y[k] == label_val) and src == source]
            if not idxs:
                continue
            ax.scatter(X_pca[idxs, 0], X_pca[idxs, 1],
                       c=colors[label_name], marker=markers[source],
                       s=70, alpha=0.75, edgecolors="k", linewidths=0.5,
                       label=f"{label_name} ({source})")
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title(f"Hand-crafted texture/shape features: oil vs. lookalike\n"
                 f"(silhouette={sil:.3f}, LOO accuracy={loo_acc:.2f}, n={len(y)})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, "lookalike_feature_separability_pca.png")
    fig.savefig(plot_path, dpi=150)
    print(f"\n[+] Saved PCA scatter plot to {plot_path}")

    summary = {
        "n_scenes": int(len(y)),
        "n_oil": int(y.sum()),
        "n_lookalike": int((1 - y).sum()),
        "oil_candidate_vs_gt_iou_mean": float(np.mean(oil_ious)) if oil_ious else None,
        "pca_explained_variance": explained.tolist(),
        "silhouette_score": float(sil),
        "loo_logistic_regression_accuracy": float(loo_acc),
        "majority_class_baseline": float(baseline),
    }
    with open(os.path.join(OUT_DIR, "lookalike_feature_separability_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Summary written to output/lookalike_feature_separability_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
