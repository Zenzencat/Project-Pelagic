"""
Project Pelagic — Pipeline Verification Script
SWU Prasarnmit AI Engineering Final Project

This script executes the entire data pipeline end-to-end to verify correctness:
1. Generates a synthetic dataset of dual-channel SAR scenes.
2. Runs calibration, filtering, dB scaling, and normalization.
3. Slices scenes into 256x256 grids.
4. Splits by Scene ID and loads patches into PyTorch DataLoaders.
5. Performs shape/range assertions.
6. Generates a diagnostic plot showing preprocessed outputs.
"""

import os
import sys
import matplotlib.pyplot as plt
import torch

# Add root folder to path to enable package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.generate_synthetic import generate_mock_dataset
from src.data.dataset import get_dataloaders

def main():
    print("[*] Starting Project Pelagic Data Pipeline Verification...")
    
    # 1. Setup paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(base_dir, "data", "synthetic")
    docs_dir = os.path.join(base_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    # 2. Generate a mock dataset for testing if it doesn't exist
    images_dir = os.path.join(data_dir, "images")
    if not os.path.exists(images_dir) or len(os.listdir(images_dir)) == 0:
        generate_mock_dataset(data_dir, num_scenes=5, size=512)
    else:
        print("[+] Mock dataset already exists, skipping generation.")
        
    # 3. Create PyTorch DataLoaders (Scene ID split)
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=data_dir,
            patch_size=256,
            stride=128,
            batch_size=4,
            train_split=0.6,
            val_split=0.2,
            seed=42
        )
    except Exception as e:
        print(f"[!] Error building DataLoaders: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 4. Pull a batch and run shape/range assertions
    print("[*] Fetching first batch from Train Loader to verify shapes...")
    try:
        images_batch, masks_batch = next(iter(train_loader))
    except StopIteration:
        print("[!] Train Loader returned empty iterator!", file=sys.stderr)
        sys.exit(1)
        
    print(f"    - Image Batch shape: {images_batch.shape} (Expected: [B, 2, 256, 256])")
    print(f"    - Mask Batch shape:  {masks_batch.shape} (Expected: [B, 1, 256, 256])")
    
    # Assertions
    assert len(images_batch.shape) == 4, "Images batch must have 4 dimensions [B, C, H, W]"
    assert images_batch.shape[1] == 2, "Images must have 2 channels (VV, VH)"
    assert images_batch.shape[2] == 256 and images_batch.shape[3] == 256, "Patch size must be 256x256"
    assert masks_batch.shape[1] == 1, "Mask must have 1 channel"
    assert masks_batch.shape[2] == 256 and masks_batch.shape[3] == 256, "Mask size must be 256x256"
    
    # Assert value ranges [0, 1]
    img_min, img_max = images_batch.min().item(), images_batch.max().item()
    mask_unique = torch.unique(masks_batch).tolist()
    print(f"    - Image value range: [{img_min:.4f}, {img_max:.4f}] (Expected within [0.0, 1.0])")
    print(f"    - Mask unique values: {mask_unique} (Expected: subset of [0.0, 1.0])")
    
    assert 0.0 <= img_min <= 1.0 and 0.0 <= img_max <= 1.0, "Preprocessed images must be scaled to [0.0, 1.0]"
    for val in mask_unique:
        assert val in [0.0, 1.0], "Mask values must be binary (0.0 or 1.0)"
        
    print("[+] Pipeline checks passed successfully! Tensor structures are correct.")
    
    # 5. Save a visual diagnostic plot for manual inspection
    print("[*] Creating visual diagnostic plot...")
    # Grab the first patch in the batch
    img_patch = images_batch[0].numpy() # shape (2, 256, 256)
    mask_patch = masks_batch[0].numpy().squeeze(0) # shape (256, 256)
    
    vv_channel = img_patch[0]
    vh_channel = img_patch[1]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot normalized VV channel
    im_vv = axes[0].imshow(vv_channel, cmap='gray', vmin=0.0, vmax=1.0)
    axes[0].set_title("VV Polarization (Normalized)")
    fig.colorbar(im_vv, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot normalized VH channel
    im_vh = axes[1].imshow(vh_channel, cmap='gray', vmin=0.0, vmax=1.0)
    axes[1].set_title("VH Polarization (Normalized)")
    fig.colorbar(im_vh, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot binary ground-truth mask
    im_mask = axes[2].imshow(mask_patch, cmap='plasma', vmin=0.0, vmax=1.0)
    axes[2].set_title("Ground-Truth Mask (1=Slick, 0=Sea)")
    fig.colorbar(im_mask, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Clean up layout
    for ax in axes:
        ax.axis('off')
        
    plot_path = os.path.join(docs_dir, "data_inspection.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"[+] Visual inspection plot saved to: {plot_path}")
    print("[*] Data pipeline verification finished successfully!")

if __name__ == "__main__":
    main()
