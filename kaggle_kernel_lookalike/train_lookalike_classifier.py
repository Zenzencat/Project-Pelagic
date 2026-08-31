"""
Project Pelagic — Phase 2: Post-Hoc Lookalike Discrimination Classifier
SWU Prasarnmit AI Engineering Final Project

Trains a lightweight binary classifier (oil vs. lookalike) on hand-crafted
GLCM/edge/shape features extracted from the real v2 U-Net's own candidate
regions -- a post-hoc filter sitting downstream of the existing segmentation
model, not a retrain of it (see docs/status.md's Track D scoping).

REVISED BASIS (see docs/status.md, "Post-Phase-1 correction"): an earlier
check found a classifier trained on the ESSD/PANGAEA look-alike dataset does
NOT transfer to this project's own data (40-55% accuracy, at/below chance) --
almost certainly because ESSD's JPG pre-normalization flips the sign of
several GLCM texture features relative to this project's raw dB imagery, and
because ESSD's own oil-vs-no-oil candidate regions are extracted by two
different methods (annotated-bbox-guided for oil, whole-patch heuristic for
no-oil), a confound in ESSD's own labels. So this script trains ONLY on this
project's own real training-pool scenes (Zenodo Part I oil + Part II
lookalike, the same Kaggle-hosted datasets kaggle_kernel/train_kaggle.py
itself trains the U-Net on) -- NOT on ESSD, and NOT on data/holdout/ (that
30-scene set is reserved for Phase 3's evaluation only; using any of it here,
even for validation, would make Phase 3's numbers dishonest).

WARNING: single-file Kaggle kernel constraint (see train_kaggle.py's own
banner). Duplicates, rather than imports:
  - UNet/DoubleConv           <- src/models/unet.py
  - preprocess_for_prediction <- src/data/preprocess.py
  - run_tiled_inference       <- src/inference.py
  - candidate-region feature extraction <- src/analysis/candidate_region_features.py
Mirror any change to those files here too.

Dataset sources (kernel-metadata.json):
  - shuddhabrotabanerjee/oil-spill-dartis-part1  (Oil/Mask_oil)
  - shuddhabrotabanerjee/oil-spill-dartis-part2  (lookalike/lookalike_mask, no_oil/no_oil_mask -- no_oil unused here)
  - trongzen/project-pelagic-model-best          (model_real_v2_best.pt)
"""
import os
import sys
import glob
import time
import json
import hashlib
import subprocess

try:
    import imagecodecs
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "imagecodecs"], check=True)

import numpy as np
import cv2
import tifffile
import torch
import torch.nn as nn
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import joblib

# NOTE: no hardcoded /kaggle/input/<dataset-slug> paths -- the actual mount
# layout doesn't reliably match that convention (see find_dir_with_tifs()'s
# docstring below). Every input path is resolved by searching /kaggle/input.
OUT_DIR = "/kaggle/working"
SEED = 42
MIN_BLOB_AREA = 400  # px, matches scripts/lookalike_feature_separability_check.py

# NOTE: mean_unet_confidence is Phase 3.8's addition -- deliberately NOT added
# to src/analysis/candidate_region_features.py's FEATURE_COLUMNS, since that
# module is shared with the ESSD/PANGAEA feature extraction (scripts/essd_
# feature_extraction.py), which has no U-Net probability output to draw this
# from at all (JPG patches, not this project's own SAR pipeline). This
# feature list is local to the own-domain lookalike classifier only.
FEATURE_COLUMNS = [
    "circularity", "solidity", "aspect_ratio", "blob_fill_fraction",
    "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
    "glcm_energy", "glcm_correlation", "glcm_ASM", "edge_density",
    "mean_unet_confidence",
]

# -------------------------------------------------------------
# STEP 1: UNet (synced with src/models/unet.py)
# -------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4); x = torch.cat([x, x3], dim=1); x = self.conv_up1(x)
        x = self.up2(x); x = torch.cat([x, x2], dim=1); x = self.conv_up2(x)
        x = self.up3(x); x = torch.cat([x, x1], dim=1); x = self.conv_up3(x)
        return self.outc(x)


# -------------------------------------------------------------
# STEP 2: preprocessing + tiled inference (synced with src/data/preprocess.py,
# src/inference.py -- real-dB branch only, this dataset is always dB-scale)
# -------------------------------------------------------------
def speckle_filter(image, window_size=5):
    if len(image.shape) == 3:
        filtered = np.zeros_like(image)
        for c in range(image.shape[2]):
            filtered[:, :, c] = cv2.blur(image[:, :, c], (window_size, window_size))
        return filtered
    return cv2.blur(image, (window_size, window_size))


def normalize_image(image_db, min_db=-25.0, max_db=0.0):
    clipped = np.clip(image_db, min_db, max_db)
    return (clipped - min_db) / (max_db - min_db)


def preprocess_for_prediction(image_raw):
    if len(image_raw.shape) == 3 and image_raw.shape[0] == 2:
        image_raw = image_raw.transpose(1, 2, 0)
    linear = 10.0 ** (image_raw.astype(np.float32) / 10.0)
    filt = speckle_filter(linear, window_size=5)
    db = 10.0 * np.log10(np.clip(filt, 1e-5, None))
    return normalize_image(db)


def run_tiled_inference(model, norm, device, patch_size=256, threshold=0.5):
    H0, W0, C = norm.shape
    pad_h = (-H0) % patch_size
    pad_w = (-W0) % patch_size
    if pad_h or pad_w:
        norm = np.pad(norm, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    H, W, _ = norm.shape
    patches, coords = [], []
    for y in range(0, H, patch_size):
        for x in range(0, W, patch_size):
            patches.append(norm[y:y + patch_size, x:x + patch_size, :].transpose(2, 0, 1))
            coords.append((y, x))
    patches_tensor = torch.from_numpy(np.array(patches)).float().to(device)
    with torch.no_grad():
        logits = model(patches_tensor)
        probs_tiles = torch.sigmoid(logits).squeeze(1).cpu().numpy()
    probs_full = np.zeros((H, W), dtype=np.float32)
    for p_idx, (y, x) in enumerate(coords):
        probs_full[y:y + patch_size, x:x + patch_size] = probs_tiles[p_idx]
    probs_full = probs_full[:H0, :W0]
    return probs_full, (probs_full > threshold).astype(np.uint8)


# -------------------------------------------------------------
# STEP 3: candidate-region features (synced with
# src/analysis/candidate_region_features.py)
# -------------------------------------------------------------
def largest_component(binary_mask, min_area):
    n_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return None
    areas = stats_cc[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    if areas[best - 1] < min_area:
        return None
    blob_mask = (labels == best).astype(np.uint8)
    x, y, w, h, area = stats_cc[best]
    return blob_mask, (x, y, w, h), int(area)


def compactness_features(blob_mask_bbox, w, h):
    contours, _ = cv2.findContours(blob_mask_bbox, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    if area <= 0 or perim <= 0:
        return None
    circularity = 4 * np.pi * area / (perim ** 2)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else np.nan
    aspect_ratio = max(w, h) / max(1, min(w, h))
    return {"circularity": circularity, "solidity": solidity, "aspect_ratio": aspect_ratio,
            "blob_fill_fraction": area / (w * h) if w * h > 0 else np.nan}


def glcm_features(gray_crop, levels=32):
    v = gray_crop.astype(np.float64).copy()
    finite = np.isfinite(v)
    v[~finite] = np.nanmedian(v[finite]) if finite.any() else 0.0
    lo, hi = np.percentile(v, [1, 99])
    if hi <= lo:
        hi = lo + 1e-3
    q = ((np.clip(v, lo, hi) - lo) / (hi - lo) * (levels - 1)).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                         levels=levels, symmetric=True, normed=True)
    return {f"glcm_{p}": float(np.mean(graycoprops(glcm, p)))
            for p in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]}


def edge_density(gray_crop):
    v = gray_crop.astype(np.float64).copy()
    finite = np.isfinite(v)
    v[~finite] = np.nanmedian(v[finite]) if finite.any() else 0.0
    lo, hi = np.percentile(v, [1, 99])
    if hi <= lo:
        hi = lo + 1e-3
    v8 = np.clip((v - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    edges = cv2.Canny(v8, 50, 150)
    return float(np.count_nonzero(edges)) / edges.size


def extract_region_features(gray_full, blob_mask, bbox, margin=15):
    x, y, w, h = bbox
    H, W = gray_full.shape[:2]
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(W, x + w + margin), min(H, y + h + margin)
    crop = gray_full[y0:y1, x0:x1]
    feats = compactness_features(blob_mask[y:y + h, x:x + w], w, h)
    if feats is None:
        return None
    feats.update(glcm_features(crop))
    feats["edge_density"] = edge_density(crop)
    return feats


# -------------------------------------------------------------
# STEP 4: build the real oil + lookalike scene pair list (dedup logic
# synced with this file's own get_real_dataloaders() / src/data/dataset.py)
# -------------------------------------------------------------
def find_dir_with_tifs(root, dirname):
    """Kaggle's actual /kaggle/input mount path layout doesn't reliably match
    the flat "<root>/oil-spill-dartis-part1/..." convention the original
    train_kaggle.py run assumed (confirmed the hard way: the model-checkpoint
    dataset mounted at .../input/datasets/trongzen/project-pelagic-model-best/...,
    an extra "datasets/" segment vs. what every hardcoded path here expected,
    and PART1_DIR/PART2_DIR turned out to have silently drifted the same way --
    "0 oil scenes found" on a real run, not a hypothetical risk). Search
    for a directory whose name matches AND that directly contains .tif files
    (both Mask_oil/Mask_oil and Oil/Oil nest the name twice, so matching name
    alone isn't enough -- must also confirm real file content one level in)."""
    for r, dirs, files in os.walk(root):
        if os.path.basename(r) == dirname and any(f.endswith(".tif") for f in files):
            return r
    return None


def build_scene_pairs():
    oil_mask_dir = find_dir_with_tifs("/kaggle/input", "Mask_oil")
    oil_img_dir = find_dir_with_tifs("/kaggle/input", "Oil")
    if oil_mask_dir is None or oil_img_dir is None:
        print(f"[!] ERROR: could not locate Mask_oil ({oil_mask_dir}) or Oil ({oil_img_dir}) under /kaggle/input")
        for r, dirs, files in os.walk("/kaggle/input"):
            if files:
                print(r, "->", files[:3], f"({len(files)} files)")
        sys.exit(1)
    print(f"[+] oil_mask_dir={oil_mask_dir}")
    print(f"[+] oil_img_dir={oil_img_dir}")
    oil_masks = sorted(glob.glob(os.path.join(oil_mask_dir, "*.tif")))
    pairs = [{"img_path": os.path.join(oil_img_dir, os.path.basename(p)), "label": "oil",
              "scene_id": f"oil_{os.path.basename(p)}"} for p in oil_masks]
    print(f"[*] {len(pairs)} oil scenes (Part I)")

    lookalike_mask_dir = find_dir_with_tifs("/kaggle/input", "lookalike_mask")
    lookalike_img_dir = find_dir_with_tifs("/kaggle/input", "lookalike")
    no_oil_mask_dir = find_dir_with_tifs("/kaggle/input", "no_oil_mask")
    no_oil_img_dir = find_dir_with_tifs("/kaggle/input", "no_oil")
    if lookalike_mask_dir is None or lookalike_img_dir is None or no_oil_mask_dir is None or no_oil_img_dir is None:
        print(f"[!] ERROR: could not locate one of lookalike_mask ({lookalike_mask_dir}), "
              f"lookalike ({lookalike_img_dir}), no_oil_mask ({no_oil_mask_dir}), no_oil ({no_oil_img_dir})")
        sys.exit(1)
    print(f"[+] lookalike_mask_dir={lookalike_mask_dir}")
    print(f"[+] lookalike_img_dir={lookalike_img_dir}")
    lookalike_masks = sorted(glob.glob(os.path.join(lookalike_mask_dir, "*.tif")))
    no_oil_masks = sorted(glob.glob(os.path.join(no_oil_mask_dir, "*.tif")))

    # Same image-level dedup check as train_kaggle.py / docs/status.md Phase 0:
    # confirms lookalike images are genuinely distinct from no_oil, not reused.
    lookalike_fnames = {os.path.basename(p) for p in lookalike_masks}
    no_oil_fnames = {os.path.basename(p) for p in no_oil_masks}
    common = sorted(lookalike_fnames & no_oil_fnames)
    dup_hashes = set()
    for fname in common:
        p1, p2 = os.path.join(lookalike_img_dir, fname), os.path.join(no_oil_img_dir, fname)
        if os.path.getsize(p1) == os.path.getsize(p2):
            if hashlib.md5(open(p1, "rb").read()).hexdigest() == hashlib.md5(open(p2, "rb").read()).hexdigest():
                dup_hashes.add(fname)
    print(f"[*] {len(dup_hashes)}/{len(common)} lookalike/no_oil image pairs identical (expect 0, per Phase 0 finding)")

    for p in lookalike_masks:
        fname = os.path.basename(p)
        pairs.append({"img_path": os.path.join(lookalike_img_dir, fname), "label": "lookalike",
                      "scene_id": f"lookalike_{fname}"})
    print(f"[*] {len(lookalike_masks)} lookalike scenes (Part II) -- {len(pairs) - len(oil_masks)} kept")
    return pairs


# -------------------------------------------------------------
# STEP 5: feature extraction over the full scene pool
# -------------------------------------------------------------
def extract_all_features(model, device, pairs):
    rows = []
    skipped = {"oil": 0, "lookalike": 0}
    t0 = time.time()
    for i, p in enumerate(pairs):
        if not os.path.exists(p["img_path"]):
            skipped[p["label"]] += 1
            continue
        try:
            raw = tifffile.imread(p["img_path"]).astype(np.float32)
        except Exception as e:
            print(f"[!] Failed to read {p['img_path']}: {e}")
            skipped[p["label"]] += 1
            continue

        norm = preprocess_for_prediction(raw)
        probs_full, pred_mask = run_tiled_inference(model, norm, device)
        result = largest_component(pred_mask, min_area=MIN_BLOB_AREA)
        if result is None:
            skipped[p["label"]] += 1
            continue
        blob_mask, bbox, area = result
        gray = raw[..., 0] if raw.ndim == 3 else raw
        feats = extract_region_features(gray, blob_mask, bbox, margin=15)
        if feats is None:
            skipped[p["label"]] += 1
            continue
        # Phase 3.8: the U-Net's own mean sigmoid confidence within the
        # candidate blob -- a real signal the model already computes
        # internally that the classifier has never seen before this round.
        feats["mean_unet_confidence"] = float(probs_full[blob_mask > 0].mean())
        feats["blob_area_px"] = area
        feats["scene_id"] = p["scene_id"]
        feats["label"] = p["label"]
        rows.append(feats)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{len(pairs)}] {elapsed:.0f}s elapsed, {len(rows)} kept, "
                  f"skipped oil={skipped['oil']} lookalike={skipped['lookalike']}")

    print(f"[+] Feature extraction done in {time.time() - t0:.0f}s: {len(rows)} kept, "
          f"skipped oil={skipped['oil']} lookalike={skipped['lookalike']}")
    return pd.DataFrame(rows), skipped


# -------------------------------------------------------------
# STEP 6: train + evaluate lightweight classifiers
# -------------------------------------------------------------
def train_classifiers(df):
    df = df.dropna(subset=FEATURE_COLUMNS)
    X = df[FEATURE_COLUMNS].values
    y = (df["label"] == "oil").astype(int).values
    scene_ids = df["scene_id"].values

    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X, y, scene_ids, test_size=0.2, random_state=SEED, stratify=y)
    print(f"[*] Train: {len(y_train)} ({y_train.sum()} oil, {(1 - y_train).sum()} lookalike)")
    print(f"[*] Val:   {len(y_val)} ({y_val.sum()} oil, {(1 - y_val).sum()} lookalike)")

    scaler = StandardScaler().fit(X_train)
    Xtr, Xva = scaler.transform(X_train), scaler.transform(X_val)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=SEED),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=SEED),
    }

    results = {}
    best_name, best_auc, best_clf = None, -1, None
    for name, clf in candidates.items():
        clf.fit(Xtr, y_train)
        pred = clf.predict(Xva)
        proba = clf.predict_proba(Xva)[:, 1]
        acc = accuracy_score(y_val, pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, pred, average="binary", zero_division=0)
        auc = roc_auc_score(y_val, proba)
        cm = confusion_matrix(y_val, pred).tolist()
        results[name] = {"val_accuracy": acc, "val_precision": prec, "val_recall": rec,
                          "val_f1": f1, "val_auc": auc, "val_confusion_matrix": cm}
        print(f"[{name}] acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} auc={auc:.3f} cm={cm}")
        if auc > best_auc:
            best_name, best_auc, best_clf = name, auc, clf

    print(f"\n[+] Best model by val AUC: {best_name} (AUC={best_auc:.3f})")

    joblib.dump({"model": best_clf, "scaler": scaler, "feature_columns": FEATURE_COLUMNS,
                 "model_name": best_name}, os.path.join(OUT_DIR, "lookalike_classifier.joblib"))
    with open(os.path.join(OUT_DIR, "train_val_split.json"), "w") as f:
        json.dump({"train_scene_ids": id_train.tolist(), "val_scene_ids": id_val.tolist()}, f)
    with open(os.path.join(OUT_DIR, "classifier_results.json"), "w") as f:
        json.dump({"results": results, "best_model": best_name, "n_train": len(y_train),
                    "n_val": len(y_val)}, f, indent=2)
    return results, best_name


def find_file_recursive(root, filename):
    """Kaggle dataset mount paths aren't always the flat top-level path you'd
    expect from the dataset slug (see kaggle_eval_kernel/eval_kaggle.py, which
    hit this same issue first and fixed it the same way)."""
    for r, dirs, files in os.walk(root):
        if filename in files:
            return os.path.join(r, filename)
    return None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    model_path = find_file_recursive("/kaggle/input", "model_real_v2_best.pt")
    if model_path is None:
        print("[!] ERROR: model_real_v2_best.pt not found anywhere under /kaggle/input!")
        for r, dirs, files in os.walk("/kaggle/input"):
            for f in files:
                print(os.path.join(r, f))
        sys.exit(1)
    print(f"[+] Found model checkpoint at: {model_path}")

    model = UNet(in_channels=2, out_channels=1)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print("[+] Loaded model_real_v2_best.pt")

    pairs = build_scene_pairs()
    df, skipped = extract_all_features(model, device, pairs)

    features_path = os.path.join(OUT_DIR, "own_domain_features.csv")
    df.to_csv(features_path, index=False)
    print(f"[+] Wrote {len(df)} feature rows to {features_path}")

    with open(os.path.join(OUT_DIR, "extraction_summary.json"), "w") as f:
        json.dump({"n_pairs_total": len(pairs), "n_kept": len(df), "n_skipped": skipped}, f, indent=2)

    if len(df) < 20:
        print("[!] Too few examples to train a classifier -- stopping.")
        return

    train_classifiers(df)


if __name__ == "__main__":
    main()
