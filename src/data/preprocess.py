"""
Project Pelagic — Preprocessing Pipelines
SWU Prasarnmit AI Engineering Final Project

This module contains preprocessing operations for SAR imagery,
including calibration, speckle filtering, decibel scaling, min-max 
normalization, and patch extraction grid slicing.
"""

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

def run_full_preprocessing(image_raw, mask_raw=None, patch_size=256, stride=128):
    """
    Runs the full pipeline sequence: calibration -> filtering -> dB scale -> normalize -> extract patches.
    Handles already-decibel (dB) Sentinel-1 imagery by converting to linear for speckle filtering,
    re-scaling back to dB, and normalizing.
    """
    if len(image_raw.shape) == 3 and image_raw.shape[0] == 2:
        image_raw = image_raw.transpose(1, 2, 0)
        
    # 1. Convert already-dB input back to linear space for speckle filtering
    linear = 10.0 ** (image_raw.astype(np.float32) / 10.0)
    
    # 2. Apply speckle filter in linear space
    filtered = speckle_filter(linear, window_size=5)
    
    # 3. Convert back to decibels
    db = 10.0 * np.log10(np.clip(filtered, 1e-5, None))
    
    # 4. Normalize to [0.0, 1.0] range
    min_db, max_db = -25.0, 0.0
    norm = (np.clip(db, min_db, max_db) - min_db) / (max_db - min_db)
    
    # 5. Patch Extraction
    if mask_raw is not None:
        return extract_patches(norm, mask_raw, patch_size, stride)
    else:
        # For prediction, we might only need the image patches
        dummy_mask = np.zeros(norm.shape[:2], dtype=np.uint8)
        img_patches, _ = extract_patches(norm, dummy_mask, patch_size, stride)
        return img_patches, norm
