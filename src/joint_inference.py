"""
Project Pelagic -- Tiled inference for JointUNet (segmentation + aux head)
SWU Prasarnmit AI Engineering Final Project (Phase 3)

Separate from src/inference.py's run_tiled_inference() -- that function is
shared by the live API and every prior evaluation script for the plain UNet
(single logits tensor return) and is left completely untouched. This module
is JointUNet-only (returns (seg_logits, cls_logits) per tile).

Segmentation stitching is identical to run_tiled_inference(): non-overlapping
patch_size tiles, reflect-pad to a multiple of patch_size, crop back.

Scene-level classification aggregates every tile's softmax probability by
simple mean (not majority vote on hard tile predictions), then argmaxes once
at the end -- avoids ties/instability from small tile counts and matches how
the segmentation side already aggregates continuous probabilities before
thresholding.
"""

import numpy as np
import torch
import torch.nn.functional as F


def run_tiled_inference_joint(model, norm, device, patch_size=256, threshold=0.5):
    """
    Returns (probs_full, preds_full, scene_cls_probs, scene_cls_pred):
      - probs_full, preds_full: same as run_tiled_inference() (segmentation)
      - scene_cls_probs: np.array shape (num_classes,), mean tile softmax
      - scene_cls_pred: int argmax of scene_cls_probs (0=lookalike, 1=oil)
    """
    H0, W0, C = norm.shape
    pad_h = (-H0) % patch_size
    pad_w = (-W0) % patch_size
    if pad_h or pad_w:
        norm = np.pad(norm, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    H, W, _ = norm.shape

    patches = []
    coords = []
    for y in range(0, H, patch_size):
        for x in range(0, W, patch_size):
            patch = norm[y:y + patch_size, x:x + patch_size, :]
            patches.append(patch.transpose(2, 0, 1))
            coords.append((y, x))

    patches_tensor = torch.from_numpy(np.array(patches)).float().to(device)

    with torch.no_grad():
        seg_logits, cls_logits = model(patches_tensor)
        probs_tiles = torch.sigmoid(seg_logits).squeeze(1).cpu().numpy()
        cls_probs_tiles = F.softmax(cls_logits, dim=1).cpu().numpy()  # (num_tiles, num_classes)

    probs_full = np.zeros((H, W), dtype=np.float32)
    for p_idx, (y, x) in enumerate(coords):
        probs_full[y:y + patch_size, x:x + patch_size] = probs_tiles[p_idx]

    probs_full = probs_full[:H0, :W0]
    preds_full = (probs_full > threshold).astype(np.uint8)

    scene_cls_probs = cls_probs_tiles.mean(axis=0)
    scene_cls_pred = int(np.argmax(scene_cls_probs))

    return probs_full, preds_full, scene_cls_probs, scene_cls_pred
