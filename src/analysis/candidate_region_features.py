"""
Shared hand-crafted texture/shape feature extraction for candidate oil/lookalike
regions -- GLCM texture, edge density, shape compactness. Used by both:
  - scripts/lookalike_feature_separability_check.py (this project's own
    raw-dB Sentinel-1 scenes, candidate region from the real v2 U-Net)
  - scripts/essd_feature_extraction.py (ESSD/PANGAEA JPG patches, candidate
    region from the dataset's own bbox annotations / a local darkest-blob
    proxy)

Single source of truth so the two scripts' feature definitions can't quietly
drift apart the way dataset.py/train_kaggle.py's patch sampling once did
(see docs/status.md Resolution #5) -- both callers feed this module an
already-loaded 2D grayscale array (dB or 8-bit, this module doesn't care)
plus a binary blob mask, so the feature *values* remain comparable across the
two very different source formats even though the exact number scale isn't
directly comparable pixel-for-pixel (a real, disclosed domain-alignment
choice: comparability lives in the feature space, not raw pixel values --
see docs/status.md's ESSD/PANGAEA Phase 1 section).
"""
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

FEATURE_COLUMNS = [
    "circularity", "solidity", "aspect_ratio", "blob_fill_fraction",
    "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
    "glcm_energy", "glcm_correlation", "glcm_ASM", "edge_density",
]


def compactness_features(blob_mask_bbox, w, h):
    """blob_mask_bbox: binary mask cropped exactly to the blob's own bounding
    box (w, h)."""
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
    return {
        "circularity": circularity,
        "solidity": solidity,
        "aspect_ratio": aspect_ratio,
        "blob_fill_fraction": area / (w * h) if w * h > 0 else np.nan,
    }


def glcm_features(gray_crop, levels=32):
    """gray_crop: 2D float or uint array (any scale -- dB or 8-bit), the
    region around the candidate blob (not masked to the exact blob shape;
    a hard blob-edge mask would distort the co-occurrence statistics)."""
    v = gray_crop.astype(np.float64).copy()
    finite = np.isfinite(v)
    v[~finite] = np.nanmedian(v[finite]) if finite.any() else 0.0
    lo, hi = np.percentile(v, [1, 99])
    if hi <= lo:
        hi = lo + 1e-3
    v_clipped = np.clip(v, lo, hi)
    q = ((v_clipped - lo) / (hi - lo) * (levels - 1)).astype(np.uint8)

    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                         levels=levels, symmetric=True, normed=True)
    out = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        out[f"glcm_{prop}"] = float(np.mean(graycoprops(glcm, prop)))
    return out


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


def largest_component(binary_mask, min_area):
    """binary_mask: uint8 0/1 array. Returns (blob_mask, (x,y,w,h), area) for
    the largest connected component, or None if none clears min_area."""
    n_labels, labels, stats_cc, centroids = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return None
    areas = stats_cc[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    if areas[best - 1] < min_area:
        return None
    blob_mask = (labels == best).astype(np.uint8)
    x, y, w, h, area = stats_cc[best]
    return blob_mask, (x, y, w, h), int(area)


def extract_region_features(gray_full, blob_mask, bbox, margin=15):
    """gray_full: full 2D scene/patch array. blob_mask: same-shape binary
    mask with the candidate blob. bbox: (x, y, w, h) of the blob. Returns the
    standard feature dict (FEATURE_COLUMNS keys) or None."""
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
