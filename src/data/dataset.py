"""
Project Pelagic — PyTorch Dataset Loader
SWU Prasarnmit AI Engineering Final Project

This module defines the PyTorch Dataset class and split loader functions.
It splits data by Scene ID to prevent spatial leakage, runs preprocessing, 
and applies synchronized data augmentations on the fly.
"""

import os
import glob
import random
import numpy as np
import tifffile
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.preprocess import run_full_preprocessing

class SARDataset(Dataset):
    """
    PyTorch Dataset for dual-channel SAR patches (VV, VH) and binary masks.
    """
    def __init__(self, image_patches, mask_patches, augment=False):
        """
        Args:
            image_patches (list): List of preprocessed numpy arrays of shape (H, W, 2).
            mask_patches (list): List of binary mask numpy arrays of shape (H, W).
            augment (bool): Whether to apply random on-the-fly augmentations.
        """
        self.images = image_patches
        self.masks = mask_patches
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()
        mask = self.masks[idx].copy()
        
        # Apply synchronized data augmentation
        if self.augment:
            # 1. Random horizontal flip
            if random.random() > 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)
                
            # 2. Random vertical flip
            if random.random() > 0.5:
                img = np.flipud(img)
                mask = np.flipud(mask)
                
            # 3. Random 90-degree rotations
            rot_k = random.choice([0, 1, 2, 3])
            if rot_k > 0:
                img = np.rot90(img, k=rot_k)
                mask = np.rot90(mask, k=rot_k)

        # Ensure memory is contiguous to prevent PyTorch stride errors after flips/rotations
        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)

        # Transpose image to PyTorch shape: (Channels, Height, Width) -> (2, H, W)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        
        # Convert mask to float tensor of shape (1, H, W) and scale to 0.0 or 1.0
        mask_binary = (mask > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).float()
        
        return img_tensor, mask_tensor

def get_dataloaders(data_dir, patch_size=256, stride=128, batch_size=8, train_split=0.6, val_split=0.2, seed=42):
    """
    Finds scenes, splits them by Scene ID (to avoid spatial leak), 
    runs preprocessing and patch extraction, and returns PyTorch DataLoaders.
    
    Args:
        data_dir (str): Root directory containing "images" and "masks" folders.
        patch_size (int): Slicing resolution.
        stride (int): Slicing stride.
        batch_size (int): DataLoader batch size.
        train_split (float): Ratio of scenes for training.
        val_split (float): Ratio of scenes for validation.
        seed (int): Random seed for split reproducibility.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    images_pattern = os.path.join(data_dir, "images", "*.tif")
    image_paths = sorted(glob.glob(images_pattern))
    
    if not image_paths:
        raise FileNotFoundError(f"No GeoTIFF (.tif) images found in {os.path.join(data_dir, 'images')}")
        
    # Get corresponding mask paths and group by scene_id
    scenes = []
    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        scene_id = os.path.splitext(base_name)[0]
        
        # Look for matching mask file (e.g. scene_id_mask.png)
        mask_path = os.path.join(data_dir, "masks", f"{scene_id}_mask.png")
        if os.path.exists(mask_path):
            scenes.append((scene_id, img_path, mask_path))
        else:
            print(f"[!] Warning: Missing mask for scene {scene_id}, skipping.")
            
    if not scenes:
        raise FileNotFoundError("Could not find matching image/mask pairs.")
        
    # Shuffle scenes by Scene ID with a fixed seed to prevent spatial leak
    random.seed(seed)
    random.shuffle(scenes)
    
    num_scenes = len(scenes)
    num_train = int(num_scenes * train_split)
    num_val = int(num_scenes * val_split)
    
    train_scenes = scenes[:num_train]
    val_scenes = scenes[num_train:num_train + num_val]
    test_scenes = scenes[num_train + num_val:]
    
    print(f"[*] Splitting dataset by Scene ID:")
    print(f"    - Total scenes: {num_scenes}")
    print(f"    - Train scenes: {len(train_scenes)}")
    print(f"    - Val scenes:   {len(val_scenes)}")
    print(f"    - Test scenes:  {len(test_scenes)}")
    
    # Process patches for each split
    def collect_patches(scene_list, name):
        img_patches_all = []
        mask_patches_all = []
        
        print(f"[*] Preprocessing and extracting patches for {name} split...")
        for scene_id, img_path, mask_path in scene_list:
            # Read GeoTIFF image (VV, VH)
            image_raw = tifffile.imread(img_path) # Expected shape (H, W, 2)
            # Read binary mask (grayscale PNG)
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Expected shape (H, W)
            
            # Run calibration, filtering, decibel conversion, normalization, and grid patch extraction
            img_p, mask_p = run_full_preprocessing(image_raw, mask_raw, patch_size, stride)
            img_patches_all.extend(img_p)
            mask_patches_all.extend(mask_p)
            
        print(f"    [+] Created {len(img_patches_all)} patches for {name} split.")
        return img_patches_all, mask_patches_all

    # Extract patch grids
    train_imgs, train_masks = collect_patches(train_scenes, "Train")
    val_imgs, val_masks = collect_patches(val_scenes, "Val")
    test_imgs, test_masks = collect_patches(test_scenes, "Test")
    
    # Create PyTorch Datasets
    train_dataset = SARDataset(train_imgs, train_masks, augment=True)
    val_dataset = SARDataset(val_imgs, val_masks, augment=False)
    test_dataset = SARDataset(test_imgs, test_masks, augment=False)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_real_dataloaders(raw_dir, patch_size=256, stride=128, batch_size=8, train_ratio=0.8, seed=42, limit_files=None, allow_verification_fallback=False):
    """
    Finds and loads the real 3-part dataset files (Trujillo-Acatitla et al.) 
    for Parts I and II. Combines them for train/val split.
    
    If the full images are not downloaded yet, it runs in a 'verification mode' 
    generating dummy 2-channel arrays of shape (2048, 2048, 2) matching the real masks 
    to verify pipeline execution.
    """
    import glob
    
    # "category_key" must match the tokens extract_patches_balanced() in
    # src/data/preprocess.py switches on ("oil" / "lookalike" / anything else -> no_oil rate).
    categories = [
        {"name": "Oil Spill", "category_key": "oil", "mask_subdir": "Mask_oil", "img_subdirs": ["01_Train_Val_Oil_Spill_images", "Train_Val_Oil_Spill_images"]},
        {"name": "No Oil", "category_key": "no_oil", "mask_subdir": "Mask_no_oil", "img_subdirs": ["01_Train_Val_No_Oil_Images", "Train_Val_No_Oil_images", "01_Train_Val_No_Oil_images"]},
        {"name": "Lookalike", "category_key": "lookalike", "mask_subdir": "Mask_lookalike", "img_subdirs": ["01_Train_Val_Lookalike_images", "Train_Val_Lookalike_images"]}
    ]
    
    all_pairs = []
    missing = []
    
    # Scan through categories
    for cat in categories:
        mask_path_pattern = os.path.join(raw_dir, cat["mask_subdir"], "*.tif")
        mask_files = sorted(glob.glob(mask_path_pattern))
        
        if not mask_files:
            print(f"[!] Warning: No mask files found for category {cat['name']} in {os.path.join(raw_dir, cat['mask_subdir'])}")
            continue
            
        print(f"[*] Found {len(mask_files)} masks for category {cat['name']}.")
        
        # Look for corresponding images folder
        img_dir = None
        for subdir in cat["img_subdirs"]:
            test_dir = os.path.join(raw_dir, subdir)
            if os.path.isdir(test_dir):
                img_dir = test_dir
                break
                
        for mask_path in mask_files:
            filename = os.path.basename(mask_path)
            img_path = None
            if img_dir is not None:
                test_img_path = os.path.join(img_dir, filename)
                if os.path.exists(test_img_path):
                    img_path = test_img_path
            
            if img_path is None:
                missing.append(filename)
            
            all_pairs.append({
                "category": cat["name"],
                "category_key": cat["category_key"],
                "mask_path": mask_path,
                "img_path": img_path # None if image not downloaded yet
            })
            
    if not all_pairs:
        raise FileNotFoundError(f"No mask files found in any category subfolder under {raw_dir}")

    # Check for missing image files
    if missing:
        if not allow_verification_fallback:
            raise RuntimeError(
                f"[!] Missing Image Files: {len(missing)} out of {len(all_pairs)} total pairs do not have "
                "matching real SAR GeoTIFF images. Since real SAR GeoTIFFs are not downloaded locally, "
                "running at this time will fallback to synthetic data. Refusing to start a real training "
                "run against synthetic fallback data. To explicitly run in verification mode with synthetic "
                "data, pass allow_verification_fallback=True."
            )
        else:
            print("\n" + "="*80)
            print(f"[!] WARNING: {len(missing)} of the total {len(all_pairs)} pairs are using synthetic fallback images.")
            print("    This is verification mode, NOT real training data.")
            print("="*80 + "\n")
        
    print(f"[*] Total real dataset pairs mapped: {len(all_pairs)}")
    
    # Shuffle and split using seed
    random.seed(seed)
    random.shuffle(all_pairs)
    
    num_total = len(all_pairs)
    num_train = int(num_total * train_ratio)
    
    train_pairs = all_pairs[:num_train]
    val_pairs = all_pairs[num_train:]
    
    if limit_files is not None:
        print(f"[*] Restricting files to limit_files={limit_files} for fast verification.")
        train_pairs = train_pairs[:limit_files]
        val_pairs = val_pairs[:limit_files]
    
    print(f"    - Train split: {len(train_pairs)} files")
    print(f"    - Val split:   {len(val_pairs)} files")
    
    def process_pairs(pairs, name):
        img_patches_all = []
        mask_patches_all = []
        
        print(f"[*] Extracting patches for {name} split ({len(pairs)} scenes)...")
        for i, pair in enumerate(pairs):
            mask_raw = cv2.imread(pair["mask_path"], cv2.IMREAD_GRAYSCALE)
            
            if pair["img_path"] is not None:
                image_raw = tifffile.imread(pair["img_path"])
                # Handle channel dimension transpose if channels are first (2, H, W)
                if len(image_raw.shape) == 3 and image_raw.shape[0] == 2:
                    image_raw = image_raw.transpose(1, 2, 0)
            else:
                # Verification mode: Generate dummy 2-channel array
                H, W = mask_raw.shape
                # Create a synthetic image background matching the real mask size
                image_raw = np.random.normal(0.1, 0.05, (H, W, 2)).astype(np.float32)
                # Inject a fake slick in the image where the mask is positive (lower backscatter)
                image_raw[mask_raw > 0] = np.random.normal(0.02, 0.01, (np.sum(mask_raw > 0), 2))
                image_raw = np.clip(image_raw, 0.0, 1.0)
                
            # Preprocess and extract 256x256 patches. Category-balanced downsampling
            # (matching the real Kaggle training run) only applies to genuine imagery —
            # verification-mode dummy scenes keep every patch so a small limit_files
            # run can't accidentally end up with an empty split.
            category = pair["category_key"] if pair["img_path"] is not None else None
            img_p, mask_p = run_full_preprocessing(image_raw, mask_raw, patch_size, stride, category=category)
            img_patches_all.extend(img_p)
            mask_patches_all.extend(mask_p)
            
            # Print periodic progress
            if (i + 1) % 100 == 0 or (i + 1) == len(pairs):
                print(f"    - Processed {i+1}/{len(pairs)} files...")
                
        print(f"    [+] Created {len(img_patches_all)} patches for {name} split.")
        return img_patches_all, mask_patches_all

    train_imgs, train_masks = process_pairs(train_pairs, "Train")
    val_imgs, val_masks = process_pairs(val_pairs, "Val")
    
    train_dataset = SARDataset(train_imgs, train_masks, augment=True)
    val_dataset = SARDataset(val_imgs, val_masks, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
