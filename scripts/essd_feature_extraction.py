"""
Phase 1 (continued): extracts the same hand-crafted texture/shape features
(src/analysis/candidate_region_features.py) from the downloaded ESSD/PANGAEA
patches (scripts/download_essd_pangaea.py), so they can sit alongside this
project's own in-domain feature table for Phase 2 training.

Domain-alignment approach (per docs/status.md's Phase 1 section): ESSD's JPGs
are pre-normalized 8-bit single-channel (VV only, no VH, no physical dB
scale) -- a real, disclosed format gap from this project's own raw calibrated
dB GeoTIFFs. Rather than trying to recover absolute calibration from a lossy
per-image JPG stretch (not recoverable -- the normalization parameters aren't
published), this reuses the SAME relative, percentile-clipped GLCM/edge/shape
feature pipeline used on this project's own scenes, which by construction
compares relative texture/shape structure, not absolute intensity -- the two
sources meet in feature space, not pixel space.

Candidate-region definition per subset:
  - oil (oc/ow): the dataset's own VOC bounding box(es), unioned per patch,
    refined by a LOCAL Otsu threshold within (bbox + margin) to get an actual
    blob shape for compactness features (confirmed on a real sample: local
    Otsu recovers a blob within ~2% of the true annotated bbox area -- a
    whole-image Otsu is far too coarse, same failure mode observed on this
    project's own raw scenes in lookalike_feature_separability_check.py).
  - no_oil (nc/nw): no bounding box exists (a look-alike patch, by
    definition, has no oil object) -- uses the whole-patch darkest-10th-
    percentile blob (largest connected component after morphological
    cleanup), the same category of detector, just without a bbox prior to
    localize it. Patches where no blob clears the minimum area are skipped
    and counted (informative: a nontrivial skip rate would mean the patch's
    look-alike signature isn't a localized dark blob at all).

Outputs: data/external/essd_pangaea/essd_features.csv,
data/external/essd_pangaea/essd_extraction_summary.json
"""
import os
import sys
import json

import numpy as np
import cv2
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO)
from src.analysis.candidate_region_features import (
    largest_component, extract_region_features, FEATURE_COLUMNS)

EXT_DIR = os.path.join(REPO, "data", "external", "essd_pangaea")
IMG_DIR = os.path.join(EXT_DIR, "images")

MIN_BLOB_AREA_NO_OIL = 30  # px, out of a 640x640=409600 patch -- small but real
BBOX_MARGIN = 15


def oil_candidate(gray, bbox):
    """Local Otsu threshold within (bbox + margin) to recover an actual blob
    shape for the annotated oil object (a raw VOC bbox by itself is a
    rectangle -- not a shape, so compactness features on the bbox alone
    would be trivial/uninformative)."""
    xmin, ymin, xmax, ymax = bbox
    H, W = gray.shape
    cx0, cy0 = max(0, int(xmin) - BBOX_MARGIN), max(0, int(ymin) - BBOX_MARGIN)
    cx1, cy1 = min(W, int(xmax) + BBOX_MARGIN), min(H, int(ymax) + BBOX_MARGIN)
    crop = gray[cy0:cy1, cx0:cx1]
    if crop.size == 0 or crop.std() < 1e-3:
        return None
    _, dark = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = (dark > 0).astype(np.uint8)
    result = largest_component(dark, min_area=10)
    if result is None:
        return None
    blob_local, (bx, by, bw, bh), area = result
    full_blob = np.zeros_like(gray, dtype=np.uint8)
    full_blob[cy0:cy1, cx0:cx1][by:by + bh, bx:bx + bw] = blob_local[by:by + bh, bx:bx + bw]
    return full_blob, (cx0 + bx, cy0 + by, bw, bh), area


def no_oil_candidate(gray):
    """Darkest 10th percentile of the whole patch + morphology + largest CC
    -- no bbox prior available, since a look-alike patch has no annotated
    object by definition."""
    p10 = np.percentile(gray, 10)
    dark = (gray <= p10).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    result = largest_component(dark, min_area=MIN_BLOB_AREA_NO_OIL)
    if result is None:
        return None
    blob_mask, bbox, area = result
    return blob_mask, bbox, area


def main():
    patches = pd.read_csv(os.path.join(EXT_DIR, "patch_index.csv"))
    print(f"[*] {len(patches)} patches indexed ({(patches['label']=='oil').sum()} oil, "
          f"{(patches['label']=='no_oil').sum()} no_oil)")

    rows = []
    skipped = {"oil": 0, "no_oil": 0}
    missing_file = 0

    for i, r in patches.iterrows():
        img_path = os.path.join(IMG_DIR, r["jpg_file"])
        if not os.path.exists(img_path):
            missing_file += 1
            continue
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            missing_file += 1
            continue

        if r["label"] == "oil":
            result = oil_candidate(gray, (r["bbox_xmin"], r["bbox_ymin"], r["bbox_xmax"], r["bbox_ymax"]))
        else:
            result = no_oil_candidate(gray)

        if result is None:
            skipped[r["label"]] += 1
            continue
        blob_mask, bbox, area = result
        feats = extract_region_features(gray, blob_mask, bbox, margin=BBOX_MARGIN)
        if feats is None:
            skipped[r["label"]] += 1
            continue

        feats["blob_area_px"] = area
        feats["patch_id"] = r["jpg_file"]
        feats["label"] = r["label"]
        feats["subset"] = r["subset"]
        feats["source"] = "essd_pangaea"
        rows.append(feats)

        if (i + 1) % 500 == 0:
            print(f"  [{i + 1}/{len(patches)}] processed ({len(rows)} kept, "
                  f"{skipped['oil']} oil skipped, {skipped['no_oil']} no_oil skipped)")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(EXT_DIR, "essd_features.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[+] Wrote {len(df)} feature rows to {out_csv}")

    n_oil_total = int((patches["label"] == "oil").sum())
    n_no_oil_total = int((patches["label"] == "no_oil").sum())
    n_oil_kept = int((df["label"] == "oil").sum()) if len(df) else 0
    n_no_oil_kept = int((df["label"] == "no_oil").sum()) if len(df) else 0

    summary = {
        "n_patches_total": len(patches),
        "n_missing_files": missing_file,
        "n_oil_total": n_oil_total,
        "n_oil_kept": n_oil_kept,
        "n_oil_skipped_no_blob": skipped["oil"],
        "n_no_oil_total": n_no_oil_total,
        "n_no_oil_kept": n_no_oil_kept,
        "n_no_oil_skipped_no_blob": skipped["no_oil"],
        "oil_keep_rate": n_oil_kept / n_oil_total if n_oil_total else None,
        "no_oil_keep_rate": n_no_oil_kept / n_no_oil_total if n_no_oil_total else None,
    }
    with open(os.path.join(EXT_DIR, "essd_extraction_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
