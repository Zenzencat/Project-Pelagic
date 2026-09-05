"""
Project Pelagic — Preprocessing Pipelines
SWU Prasarnmit AI Engineering Final Project

This module contains preprocessing operations for SAR imagery,
including calibration, speckle filtering, decibel scaling, min-max 
normalization, and patch extraction grid slicing.
"""

import random
import numpy as np
import cv2

def calibrate_to_sigma(image_dn, calibration_constant=1.0):
    """
    Calibrates raw digital numbers (DN) to backscatter coefficient Sigma Nought.
    If the image is already calibrated (e.g. synthetic data), returns the input.
    """
    # For a real Sentinel-1 GRD product, calibration requires dividing by 
    # a lookup table (LUT) provided in the metadata. For our PoC, we implement
    # a simple linear scale factor.
    return (image_dn.astype(np.float32) ** 2) * calibration_constant

def speckle_filter(image, window_size=5):
    """
    Reduces radar speckle noise using a sliding-window boxcar (mean) filter.
    Handles single-channel or multi-channel arrays.
    """
    if window_size <= 1:
        return image
        
    # Check if image has multiple channels
    if len(image.shape) == 3:
        filtered = np.zeros_like(image)
        for c in range(image.shape[2]):
            filtered[:, :, c] = cv2.blur(image[:, :, c], (window_size, window_size))
        return filtered
    else:
        return cv2.blur(image, (window_size, window_size))

def to_decibels(image_linear, eps=1e-5):
    """
    Converts linear backscatter coefficient (Sigma Nought) to logarithmic decibel (dB) scale.
    Prevents negative infinity by clipping input values to a minimum epsilon.
    """
    clipped = np.clip(image_linear, eps, None)
    return 10.0 * np.log10(clipped)

def normalize_image(image_db, min_db=-25.0, max_db=0.0):
    """
    Normalizes decibel scaling into a standard [0.0, 1.0] range.
    Clips values outside the min_db and max_db range.
    """
    clipped = np.clip(image_db, min_db, max_db)
    return (clipped - min_db) / (max_db - min_db)

def extract_patches(image, mask, patch_size=256, stride=128):
    """
    Extracts overlapping patches from a large image and corresponding mask.
    
    Args:
        image (np.ndarray): Preprocessed image array of shape (H, W, Channels).
        mask (np.ndarray): Binary mask array of shape (H, W).
        patch_size (int): Dimensions of the square patch.
        stride (int): Slicing stride.
        
    Returns:
        tuple: (list of image patches, list of mask patches)
    """
    H, W = image.shape[:2]
    image_patches = []
    mask_patches = []
    
    # Pad image and mask if they are smaller than patch_size
    pad_h = max(0, patch_size - H)
    pad_w = max(0, patch_size - W)
    
    if pad_h > 0 or pad_w > 0:
        if len(image.shape) == 3:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        H, W = image.shape[:2]
        
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            img_patch = image[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            
            # Simple validity check: ensure patch is not entirely zero (no-signal border)
            if np.mean(img_patch) > 1e-4:
                image_patches.append(img_patch)
                mask_patches.append(mask_patch)
                
    return image_patches, mask_patches

def extract_patches_balanced(image, mask, patch_size=256, stride=128, category="oil"):
    """
    Extracts patches like extract_patches(), but keeps 100% of patches containing
    slick pixels and downsamples background-only patches at a category-dependent
    rate, matching the memory-safe sampling used for the actual Kaggle training run:
      - oil:       1.0% of background patches kept (prob = 0.01)
      - lookalike: 0.5% of background patches kept (prob = 0.005) -> oversampled ~2.5x vs no_oil
      - no_oil:    0.2% of background patches kept (prob = 0.002)

    kaggle_kernel/train_kaggle.py keeps its own copy of this (Kaggle kernels must
    be self-contained single files) — see the sync warning there. If you change
    the rates here, mirror the change in that file too.
    """
    H, W = image.shape[:2]
    image_patches = []
    mask_patches = []

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            img_patch = image[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]

            if np.mean(img_patch) > 1e-4:
                has_slick = np.any(mask_patch > 0)
                if has_slick:
                    image_patches.append(img_patch.copy())
                    mask_patches.append(mask_patch.copy())
                else:
                    if category == "oil":
                        prob = 0.01
                    elif category == "lookalike":
                        prob = 0.005
                    else:
                        prob = 0.002
                    if random.random() < prob:
                        image_patches.append(img_patch.copy())
                        mask_patches.append(mask_patch.copy())

    return image_patches, mask_patches

def preprocess_for_prediction(image_raw, already_calibrated=False):
    """
    Real-vs-synthetic-aware preprocessing for a single full scene, shared by
    src/api/main.py's /api/predict (cached holdout/synthetic scenes) and
    /api/live/fetch (live CDSE scenes), so the two entry points can't drift
    apart the way whole-image-vs-tiled inference once did (see
    src/inference.py::run_tiled_inference's docstring). Branches on
    np.any(image_raw < 0) exactly as /api/predict originally did inline, to
    distinguish already-dB-scale real Sentinel-1 holdout imagery from
    linear-scale data.

    already_calibrated=True skips calibrate_to_sigma()'s DN-squaring step in
    the linear branch -- for live CDSE scenes, which arrive as real
    SIGMA0_ELLIPSOID-calibrated linear sigma0 from Sentinel Hub (see
    src/data/cdse_fetch.py), not uncalibrated raw digital numbers.
    Synthetic scenes (the only other linear-scale caller) keep
    already_calibrated=False, preserving the exact prior behavior.

    Returns the normalized (H, W, 2) array ready for run_tiled_inference().
    """
    if len(image_raw.shape) == 3 and image_raw.shape[0] == 2:
        image_raw = image_raw.transpose(1, 2, 0)

    if np.any(image_raw < 0):
        # Already in dB scale (real Sentinel-1 holdout imagery).
        linear = 10.0 ** (image_raw.astype(np.float32) / 10.0)
        filt = speckle_filter(linear, window_size=5)
        db = 10.0 * np.log10(np.clip(filt, 1e-5, None))
        return normalize_image(db)
    else:
        sig = image_raw.astype(np.float32) if already_calibrated else calibrate_to_sigma(image_raw)
        filt = speckle_filter(sig, window_size=5)
        db = to_decibels(filt)
        return normalize_image(db)


def run_full_preprocessing(image_raw, mask_raw=None, patch_size=256, stride=128, category=None):
    """
    Runs shared real-vs-synthetic-aware preprocessing, then extracts patches.
    ``preprocess_for_prediction()`` detects already-dB Sentinel-1 imagery and
    preserves the established linear-scale calibration path for synthetic data.

    If `category` is given ("oil" / "lookalike" / "no_oil"), background patches are
    downsampled via extract_patches_balanced() instead of extract_patches() keeping
    every patch, matching the class-balanced sampling used for the real training run.
    """
    norm = preprocess_for_prediction(image_raw)

    # 5. Patch Extraction
    if mask_raw is not None:
        if category is not None:
            return extract_patches_balanced(norm, mask_raw, patch_size, stride, category)
        return extract_patches(norm, mask_raw, patch_size, stride)
    else:
        # For prediction, we might only need the image patches
        dummy_mask = np.zeros(norm.shape[:2], dtype=np.uint8)
        img_patches, _ = extract_patches(norm, dummy_mask, patch_size, stride)
        return img_patches, norm
