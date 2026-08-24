"""
Project Pelagic — Shared Inference Logic
SWU Prasarnmit AI Engineering Final Project

Single source of truth for running the U-Net over a full scene. Both the
live API (src/api/main.py) and the offline evaluator (src/evaluate_holdout.py)
must use this so predictions never diverge based on which entrypoint called them.
"""

import numpy as np
import torch


def run_tiled_inference(model, norm, device, patch_size=256, threshold=0.5):
    """
    Runs inference by tiling into non-overlapping patch_size x patch_size windows
    and stitching the results back to full resolution.

    This matches the patch-based regime the model was trained on (see
    kaggle_kernel/train_kaggle.py PATCH_SIZE=256). Feeding a full scene through
    in one forward pass instead is architecturally valid (shape-wise) but puts
    every interior pixel in a receptive-field context the model never saw during
    training, since training patches were isolated tensors with zero-padding at
    their edges rather than real neighboring pixels. Tiling here reproduces that
    same context at inference time.

    H and W should normally be exact multiples of patch_size (true for all
    current holdout/synthetic scenes: 2048x2048 and 512x512). For inputs that
    aren't (live CDSE scenes are sized to a multiple by
    src/data/cdse_fetch.py already, but this is a defensive fallback, not the
    primary mechanism), this reflect-pads up to the next multiple before
    tiling and crops back down to the original H, W before returning --
    a no-op for already-aligned inputs, so holdout/synthetic/evaluate_holdout.py
    behavior is unchanged.
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
        logits = model(patches_tensor)
        probs_tiles = torch.sigmoid(logits).squeeze(1).cpu().numpy()

    probs_full = np.zeros((H, W), dtype=np.float32)
    for p_idx, (y, x) in enumerate(coords):
        probs_full[y:y + patch_size, x:x + patch_size] = probs_tiles[p_idx]

    probs_full = probs_full[:H0, :W0]
    preds_full = (probs_full > threshold).astype(np.uint8)
    return probs_full, preds_full
