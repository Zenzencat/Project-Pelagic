"""
Project Pelagic — Post-Hoc Lookalike Discrimination Filter (Phase 4)
SWU Prasarnmit AI Engineering Final Project

Final, saved, ADDITIVE stage sitting downstream of the existing v2 U-Net
(docs/status.md's Track D scoping): takes the U-Net's own candidate
detection region and decides whether it's real oil or a lookalike, using a
lightweight Gradient Boosting classifier (checkpoints/lookalike_classifier_final.joblib,
12 features: the 11 shared GLCM/edge/shape features in
candidate_region_features.py plus mean_unet_confidence) plus a confidence
gate on top.

CONFIDENCE GATE (Phase 3.9): the classifier is not applied at all when the
U-Net's own mean prediction confidence within the candidate region exceeds
GATE_THRESHOLD -- at that point the U-Net's original "oil" call is kept
as-is. Phase 3.9 found real lookalikes are SOMETIMES also very confidently
(and wrongly) flagged by the U-Net -- e.g. holdout scene lookalike_00001 has
higher U-Net confidence (0.980) than most of the oil scenes this gate exists
to protect -- so this is a real, accepted trade (verified on the real
30-scene holdout, not just simulated): oil bucket 3->6 correct, lookalike
bucket 8->7 correct. GATE_THRESHOLD=0.975 is the exact value swept and
reported in docs/status.md's Phase 3.9 section, not a rounded approximation
-- the valid range that reproduces that exact result is [0.974, 0.977)
(tight enough that 0.975 is not an arbitrary midpoint).

NOT enabled anywhere by default. This module is only invoked when a caller
explicitly opts in (see src/api/main.py's PredictRequest.apply_lookalike_filter,
default False) -- the existing verified pipeline (4 cached demo scenes,
/api/live/fetch) is completely untouched unless a caller asks for this.
"""
import os
import numpy as np
import joblib

from src.analysis.candidate_region_features import (
    largest_component, extract_region_features, FEATURE_COLUMNS)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CLASSIFIER_PATH = os.path.join(REPO_ROOT, "checkpoints", "lookalike_classifier_final.joblib")

MIN_BLOB_AREA = 400  # px, matches every round of this investigation (Phase 1-3.9)
GATE_THRESHOLD = 0.975  # exact value from docs/status.md's Phase 3.9 sweep

_CACHED_BUNDLE = None
_CACHED_PATH = None


def load_classifier(path=None):
    """Loads (and caches) the classifier bundle: {model, scaler, feature_columns,
    model_name}. feature_columns is the authoritative feature order/list --
    always FEATURE_COLUMNS + ['mean_unet_confidence'] for the Phase 3.8+
    bundles this module is built for."""
    global _CACHED_BUNDLE, _CACHED_PATH
    path = path or DEFAULT_CLASSIFIER_PATH
    if _CACHED_BUNDLE is None or _CACHED_PATH != path:
        _CACHED_BUNDLE = joblib.load(path)
        _CACHED_PATH = path
    return _CACHED_BUNDLE


def apply_confidence_gate(mean_unet_confidence, classifier_verdict, threshold=GATE_THRESHOLD):
    """The Phase 3.9 rule, as real callable code: if the U-Net's own mean
    confidence in the candidate region exceeds `threshold`, keep the U-Net's
    original 'oil' call regardless of what the classifier said. Otherwise
    use the classifier's verdict as-is."""
    if mean_unet_confidence > threshold:
        return "oil"
    return classifier_verdict


def classify_and_filter(image_raw, probs_full, pred_mask, classifier_path=None,
                         min_area=MIN_BLOB_AREA, gate_threshold=GATE_THRESHOLD):
    """
    Full pipeline: finds the U-Net's largest candidate connected component in
    `pred_mask` (binary 0/1, e.g. from src/inference.py::run_tiled_inference),
    extracts its features (including mean U-Net confidence from `probs_full`,
    the same array's continuous sigmoid output), runs the trained classifier,
    applies the confidence gate, and returns the FILTERED binary mask plus a
    diagnostic dict.

    Returns (filtered_mask, info) where info is:
      {"applied": bool, "verdict": "oil"|"lookalike"|None,
       "gated": bool, "proba_oil": float|None,
       "mean_unet_confidence": float|None, "reason": str}
    "applied" is False (mask returned unchanged) when there's no candidate
    region at all, or the region's features couldn't be extracted -- there's
    nothing for the classifier to judge in either case.
    """
    result = largest_component(pred_mask, min_area=min_area)
    if result is None:
        return pred_mask, {"applied": False, "verdict": None, "gated": False,
                            "proba_oil": None, "mean_unet_confidence": None,
                            "reason": "no_candidate_region"}

    blob_mask, bbox, area = result
    gray = image_raw[..., 0] if image_raw.ndim == 3 else image_raw
    feats = extract_region_features(gray, blob_mask, bbox, margin=15)
    if feats is None or any(k not in feats or feats[k] is None for k in FEATURE_COLUMNS):
        return pred_mask, {"applied": False, "verdict": None, "gated": False,
                            "proba_oil": None, "mean_unet_confidence": None,
                            "reason": "feature_extraction_failed"}

    mean_conf = float(probs_full[blob_mask > 0].mean())
    feats["mean_unet_confidence"] = mean_conf

    bundle = load_classifier(classifier_path)
    clf, scaler, feature_columns = bundle["model"], bundle["scaler"], bundle["feature_columns"]
    X = np.array([[feats[k] for k in feature_columns]])
    Xs = scaler.transform(X)
    pred_label = clf.predict(Xs)[0]  # 1 = oil, 0 = lookalike
    proba_oil = float(clf.predict_proba(Xs)[0, 1])
    classifier_verdict = "oil" if pred_label == 1 else "lookalike"

    final_verdict = apply_confidence_gate(mean_conf, classifier_verdict, threshold=gate_threshold)
    gated = final_verdict != classifier_verdict

    if final_verdict == "lookalike":
        filtered_mask = np.zeros_like(pred_mask)
    else:
        filtered_mask = pred_mask

    return filtered_mask, {"applied": True, "verdict": final_verdict, "gated": gated,
                            "proba_oil": proba_oil, "mean_unet_confidence": mean_conf,
                            "reason": "classifier_gate_applied" if not gated else "confidence_gate_override"}
