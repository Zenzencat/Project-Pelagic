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
