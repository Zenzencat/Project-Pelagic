"""
Project Pelagic — Synthetic SAR Scene Generator
SWU Prasarnmit AI Engineering Final Project

This script generates synthetic dual-channel (VV, VH) SAR scenes
and corresponding binary labels to verify the pipeline.
Note: Synthetic data is strictly for code verification, NOT evaluation.
"""

import os
import cv2
import numpy as np
import tifffile

def generate_synthetic_scene(scene_id, output_img_path, output_mask_path, size=512):
    """
    Generates a synthetic dual-channel (VV, VH) SAR image as a GeoTIFF 
    and a corresponding binary mask PNG file.
    
    Args:
        scene_id (str): Scene identifier.
        output_img_path (str): Filepath to save the synthetic GeoTIFF.
        output_mask_path (str): Filepath to save the binary mask PNG.
        size (int): Size of the image (square).
    """
    # 1. Simulate background ocean backscatter in linear scale (Sigma Nought)
    # VV: higher backscatter (mean around -12 dB -> linear 0.063)
    # VH: lower backscatter (mean around -20 dB -> linear 0.010)
    vv_mean = 0.063
    vh_mean = 0.010
    
    # SAR speckle noise is modeled as multi-look Gamma noise
    # L = 4 looks is a standard satellite configuration
    looks = 4
    speckle_vv = np.random.gamma(shape=looks, scale=1.0/looks, size=(size, size)).astype(np.float32)
    speckle_vh = np.random.gamma(shape=looks, scale=1.0/looks, size=(size, size)).astype(np.float32)
    
    vv = np.full((size, size), vv_mean, dtype=np.float32) * speckle_vv
    vh = np.full((size, size), vh_mean, dtype=np.float32) * speckle_vh
    
    # 2. Generate a binary mask representing oil slick shapes
    mask = np.zeros((size, size), dtype=np.uint8)
    
    # Draw random slick shapes (polygons or lines)
    num_slicks = np.random.randint(1, 4)
    for _ in range(num_slicks):
        slick_shape = np.random.choice(["polygon", "line"])
        if slick_shape == "polygon":
            # Generate random vertices for a polygon
            num_vertices = np.random.randint(4, 8)
            x_coords = np.random.randint(size // 6, 5 * size // 6, size=num_vertices)
            y_coords = np.random.randint(size // 6, 5 * size // 6, size=num_vertices)
            pts = np.stack([x_coords, y_coords], axis=1)
            # Reorder points to prevent self-intersection (convex hull simplification)
            center = pts.mean(axis=0)
            angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
            pts = pts[np.argsort(angles)]
            
            cv2.fillPoly(mask, [pts.reshape((-1, 1, 2))], 255)
        else:
            # Draw a thick linear discharge path (bilge dump trace)
            pt1 = (np.random.randint(size // 6, 5 * size // 6), np.random.randint(size // 6, 5 * size // 6))
            pt2 = (np.random.randint(size // 6, 5 * size // 6), np.random.randint(size // 6, 5 * size // 6))
            thickness = np.random.randint(6, 20)
            cv2.line(mask, pt1, pt2, 255, thickness)
            
    # 3. Apply wave damping inside the slick areas
    # Damping reduces backscatter by 6dB to 10dB (linear division factor of 4.0 to 10.0)
    damping_factor_vv = np.random.uniform(4.0, 9.0)
    damping_factor_vh = np.random.uniform(3.0, 6.0)
    
    vv[mask == 255] /= damping_factor_vv
    vh[mask == 255] /= damping_factor_vh
    
    # Stack VV and VH channels into a 2-channel float32 image
    stacked = np.stack([vv, vh], axis=-1)
    
    # Write to local file storage
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    
    tifffile.imwrite(output_img_path, stacked)
    cv2.imwrite(output_mask_path, mask)
    
    print(f"[+] Generated synthetic scene: {scene_id}")
    print(f"    - Image path: {output_img_path}")
    print(f"    - Mask path:  {output_mask_path}")

def generate_mock_dataset(data_dir, num_scenes=5, size=512):
    """
    Generates a folder of synthetic scenes to test the dataloader pipeline.
    """
    print(f"[*] Generating mock dataset with {num_scenes} scenes...")
    for i in range(1, num_scenes + 1):
        scene_id = f"S1_MOCK_SCENE_{i:03d}"
        img_path = os.path.join(data_dir, "images", f"{scene_id}.tif")
        mask_path = os.path.join(data_dir, "masks", f"{scene_id}_mask.png")
        generate_synthetic_scene(scene_id, img_path, mask_path, size=size)

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthetic"))
    generate_mock_dataset(base_dir, num_scenes=5)
