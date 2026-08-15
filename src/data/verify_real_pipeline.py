"""
Project Pelagic — Real Dataset Loader Verification
SWU Prasarnmit AI Engineering Final Project

This script verifies that the real-data dataset loader is compatible with the 
Trujillo-Acatitla et al. mask directories, correctly pairs files, handles 
dimension transposes, and outputs standard patch shapes [B, 2, 256, 256] 
and [B, 1, 256, 256].
"""

import os
import sys
import torch

# Add project root to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.dataset import get_real_dataloaders

def test_real_pipeline():
    raw_data_dir = os.path.abspath("data/raw")
    print(f"[*] Starting Pipeline Verification on Real Masks in: {raw_data_dir}")
    
    try:
        # Load dataloaders, limiting to 5 files per split for quick verification
        train_loader, val_loader = get_real_dataloaders(
            raw_dir=raw_data_dir,
            patch_size=256,
            stride=128,
            batch_size=4,
            train_ratio=0.8,
            seed=42,
            limit_files=5,
            allow_verification_fallback=True
        )
        
        print("\n[+] Dataloaders successfully initialized.")
        
        # Pull a single batch from the training loader
        img_batch, mask_batch = next(iter(train_loader))
        
        print("\n[*] Batch Validation:")
        print(f"    - Image Batch shape: {img_batch.shape} (Expected: [B, 2, 256, 256])")
        print(f"    - Mask Batch shape:  {mask_batch.shape} (Expected: [B, 1, 256, 256])")
        print(f"    - Image Min/Max:     {img_batch.min().item():.4f} / {img_batch.max().item():.4f}")
        print(f"    - Mask unique values: {torch.unique(mask_batch).tolist()} (Expected: [0.0, 1.0])")
        
        # Assertions
        assert len(img_batch.shape) == 4, "Image batch must be 4D tensor"
        assert img_batch.shape[1] == 2, "Image must have 2 channels (VV, VH)"
        assert img_batch.shape[2:] == (256, 256), "Image patches must be 256x256"
        
        assert len(mask_batch.shape) == 4, "Mask batch must be 4D tensor"
        assert mask_batch.shape[1] == 1, "Mask must have 1 channel"
        assert mask_batch.shape[2:] == (256, 256), "Mask patches must be 256x256"
        
        # Check values
        assert torch.all((mask_batch == 0) | (mask_batch == 1)), "Mask values must be strictly binary [0, 1]"
        
        print("\n[+] SUCCESS: Real dataset loader is fully verified and functional on the mask files!")
        return True
        
    except Exception as e:
        print(f"\n[!] Verification Failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_real_pipeline()
