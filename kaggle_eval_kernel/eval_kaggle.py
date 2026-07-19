import os
import sys
import glob
import time
import subprocess
import numpy as np

# -------------------------------------------------------------
# STEP 1: PRE-FLIGHT PACKAGE INSTALLATION
# -------------------------------------------------------------
print("[*] Installing imagecodecs package...")
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "imagecodecs"], check=True)
    print("[+] imagecodecs installed successfully.")
except Exception as e:
    print(f"[!] Warning: Failed to install imagecodecs via pip: {e}")

import tifffile
import cv2
import torch
import torch.nn as nn

# -------------------------------------------------------------
# STEP 2: MODEL ARCHITECTURE DEFINITION
# -------------------------------------------------------------
# WARNING: Synced copy of src/models/unet.py. Do not edit independently.
# If you modify this block, also update src/models/unet.py.

class DoubleConv(nn.Module):
    """(Convolution -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    Symmetric 4-level U-Net model with base 64 channels.
    Matches the architecture trained on the real Sentinel-1 dataset.
    """
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        
        logits = self.outc(x)
        return logits

# -------------------------------------------------------------
# STEP 3: PREPROCESSING HELPERS
# -------------------------------------------------------------
def speckle_filter(img, window_size=5):
    """Apply multiplicative noise reduction (mean filter) per channel."""
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        out[..., c] = cv2.blur(img[..., c], (window_size, window_size))
    return out

def preprocess_image(img_raw):
    """Decibel-aware Sentinel-1 SAR image preprocessing."""
    if len(img_raw.shape) == 3 and img_raw.shape[0] == 2:
        img_raw = img_raw.transpose(1, 2, 0)
        
    # 1. Convert back to linear space for speckle filtering
    linear = 10.0 ** (img_raw.astype(np.float32) / 10.0)
    
    # 2. Apply mean filter in linear space
    filtered = speckle_filter(linear, window_size=5)
    
    # 3. Convert back to dB
    db = 10.0 * np.log10(np.clip(filtered, 1e-5, None))
    
    # 4. Normalize to [-25.0, 0.0] range
    min_db, max_db = -25.0, 0.0
    norm = (np.clip(db, min_db, max_db) - min_db) / (max_db - min_db)
    
    return norm

# -------------------------------------------------------------
# STEP 4: METRICS CALCULATION
# -------------------------------------------------------------
def compute_metrics(pred, target):
    """Calculate pixel-level IoU, Dice, Precision, and Recall for binary segmentation."""
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    
    tp = np.sum((pred == 1) & (target == 1))
    fp = np.sum((pred == 1) & (target == 0))
    fn = np.sum((pred == 0) & (target == 1))
    
    if np.sum(target) == 0:
        # Negative scene (No Oil or Lookalike with empty mask)
        if np.sum(pred) == 0:
            return 1.0, 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0, 0.0
    else:
        # Positive scene (Oil)
        iou = float(tp) / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
        precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
        return iou, dice, precision, recall

# -------------------------------------------------------------
# STEP 5: EVALUATION LOOP
# -------------------------------------------------------------
def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Running evaluation on device: {device}")
    if device.type == "cuda":
        print(f"    GPU Name: {torch.cuda.get_device_name(0)} (CUDA Cap: {torch.cuda.get_device_capability(0)})")

    input_root = "/kaggle/input"

    # 1. Load model checkpoint recursively
    model_path = None
    print("[*] Searching for 'model_real_best.pt' recursively under /kaggle/input...")
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file == "model_real_best.pt":
                model_path = os.path.join(root, file)
                print(f"[+] Found model checkpoint at: {model_path}")
                break
        if model_path:
            break
            
    if not model_path:
        print("[!] ERROR: 'model_real_best.pt' not found anywhere under /kaggle/input!")
        print("[*] Listing all files available in /kaggle/input/ for troubleshooting:")
        for root, dirs, files in os.walk(input_root):
            for file in files:
                print(os.path.join(root, file))
        sys.exit(1)
        
    print(f"[*] Loading model weights...")
    model = UNet(in_channels=2, out_channels=1)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("[+] Model loaded successfully.")

    # 2. Locate Part III test directory recursively by searching for any '00000.tif'
    test_tiff = "00000.tif"
    sample_path = None
    print("[*] Searching for sample test image '00000.tif' recursively under /kaggle/input...")
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file == test_tiff and "Images" in root:
                sample_path = os.path.join(root, file)
                print(f"[+] Found sample test image at: {sample_path}")
                break
        if sample_path:
            break
            
    if sample_path:
        parts = sample_path.split("/")
        images_idx = parts.index("Images")
        TEST_DIR = "/".join(parts[:images_idx])
        print(f"[+] Derived TEST_DIR: {TEST_DIR}")
    else:
        print("[!] ERROR: Sample test image '00000.tif' not found recursively under /kaggle/input!")
        print("[*] Listing all files available in /kaggle/input/:")
        for root, dirs, files in os.walk(input_root):
            for file in files:
                print(os.path.join(root, file))
        sys.exit(1)

    categories = ["Oil", "No oil", "Lookalike"]
    results = {}
    overall_metrics = {"iou": [], "dice": [], "precision": [], "recall": []}
    
    print("\n=== STARTING INFERENCE ON HELDOUT PART III TEST SET ===")
    
    for cat in categories:
        img_dir = os.path.join(TEST_DIR, "Images", cat)
        mask_dir = os.path.join(TEST_DIR, "Mask", cat)
        
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        print(f"\n[*] Evaluating Category: '{cat}' ({len(img_paths)} scenes)...")
        
        cat_metrics = {"iou": [], "dice": [], "precision": [], "recall": []}
        
        for idx, img_path in enumerate(img_paths):
            filename = os.path.basename(img_path)
            mask_filename = filename.replace(".tif", "_segmentation.tif")
            mask_path = os.path.join(mask_dir, mask_filename)
            
            # Load raw data
            img_raw = tifffile.imread(img_path)
            mask_raw = tifffile.imread(mask_path)
            
            # Preprocess
            norm = preprocess_image(img_raw)
            
            # Slice into 256x256 non-overlapping patches (stride 256)
            H, W, C = norm.shape
            patch_size = 256
            patches = []
            coords = []
            
            for y in range(0, H, patch_size):
                for x in range(0, W, patch_size):
                    patch = norm[y:y+patch_size, x:x+patch_size, :]
                    patches.append(patch.transpose(2, 0, 1))
                    coords.append((y, x))
            
            patches = np.array(patches) # shape (64, 2, 256, 256)
            patches_tensor = torch.from_numpy(patches).float().to(device)
            
            # Inference
            with torch.no_grad():
                logits = model(patches_tensor)
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                preds = (probs > 0.5).astype(np.uint8)
                
            # Stitch patches back together
            pred_full = np.zeros((H, W), dtype=np.uint8)
            for p_idx, (y, x) in enumerate(coords):
                pred_full[y:y+patch_size, x:x+patch_size] = preds[p_idx]
                
            # Compute metrics
            iou, dice, precision, recall = compute_metrics(pred_full, mask_raw)
            
            cat_metrics["iou"].append(iou)
            cat_metrics["dice"].append(dice)
            cat_metrics["precision"].append(precision)
            cat_metrics["recall"].append(recall)
            
            overall_metrics["iou"].append(iou)
            overall_metrics["dice"].append(dice)
            overall_metrics["precision"].append(precision)
            overall_metrics["recall"].append(recall)
            
            if (idx + 1) % 50 == 0 or (idx + 1) == len(img_paths):
                print(f"    Processed {idx+1}/{len(img_paths)} scenes...")

        # Summarize category performance
        results[cat] = {
            "iou": np.mean(cat_metrics["iou"]),
            "dice": np.mean(cat_metrics["dice"]),
            "precision": np.mean(cat_metrics["precision"]),
            "recall": np.mean(cat_metrics["recall"])
        }
        print(f"  [+] category '{cat}' Metrics:")
        print(f"      IoU:       {results[cat]['iou']:.4f}")
        print(f"      Dice:      {results[cat]['dice']:.4f}")
        print(f"      Precision: {results[cat]['precision']:.4f}")
        print(f"      Recall:    {results[cat]['recall']:.4f}")

    # 3. Overall performance report
    overall = {
        "iou": np.mean(overall_metrics["iou"]),
        "dice": np.mean(overall_metrics["dice"]),
        "precision": np.mean(overall_metrics["precision"]),
        "recall": np.mean(overall_metrics["recall"])
    }
    
    print("\n==============================================")
    print("=== OVERALL PART III TEST SET RESULTS ===")
    print("==============================================")
    print(f"  Overall IoU:       {overall['iou']:.4f}")
    print(f"  Overall Dice:      {overall['dice']:.4f}  (F1-Score)")
    print(f"  Overall Precision: {overall['precision']:.4f}")
    print(f"  Overall Recall:    {overall['recall']:.4f}")
    print("==============================================")
    
    # 4. Save metrics as CSV
    out_csv = "/kaggle/working/evaluation_metrics.csv"
    with open(out_csv, "w") as f:
        f.write("category,iou,dice,precision,recall\n")
        for cat in categories:
            f.write(f"{cat},{results[cat]['iou']:.6f},{results[cat]['dice']:.6f},{results[cat]['precision']:.6f},{results[cat]['recall']:.6f}\n")
        f.write(f"Overall,{overall['iou']:.6f},{overall['dice']:.6f},{overall['precision']:.6f},{overall['recall']:.6f}\n")
    print(f"\n[+] Results saved successfully to {out_csv}")

if __name__ == "__main__":
    t_start = time.time()
    run_evaluation()
    print(f"\n[*] Total execution time: {time.time() - t_start:.2f} seconds.")
