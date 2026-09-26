"""
Project Pelagic -- v3 loss-function ablation (Focal Loss + IoU Loss)

This is a controlled comparison against v2 (kaggle_kernel/train_kaggle.py):
the ONLY change is the loss function (BCEDiceLoss -> FocalIoULoss, see STEP 4).
Architecture, data pipeline, split seed/ratio, batch size, epochs, optimizer,
and LR schedule are byte-identical to v2's kernel -- diff this file against
kaggle_kernel/train_kaggle.py to confirm nothing else moved.

Split-fingerprint check (new in this kernel, STEP 3): get_real_dataloaders()
hashes its own computed train/val scene-ID assignment and compares it against
the v2 split's SHA-256 fingerprints, reconstructed offline from the same
deterministic algorithm (sorted mask globs + seed=42 shuffle) against the
locally-held mask files, and cross-checked against v2's own real training log
(kaggle_output/project-pelagic-training.log: 1200 oil masks, 0 no-oil/lookalike
duplicates removed, 2056 train / 514 val). If the live-computed fingerprint on
Kaggle doesn't match, this kernel hard-fails before spending any GPU time on
an uncontrolled comparison.
"""
import os
import sys
import glob
import time
import random
import hashlib
import shutil
import subprocess

# Installs imagecodecs for LZW TIFF decompression on Kaggle if missing
try:
    import imagecodecs
except ImportError:
    print("[*] Installing imagecodecs package...")
    subprocess.run([sys.executable, "-m", "pip", "install", "imagecodecs"], check=True)
    import imagecodecs
    print("[+] imagecodecs installed successfully.")

import numpy as np
import cv2
import tifffile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURATION ---
# NOTE: PART1_DIR/PART2_DIR below are kept only as a fallback label for
# print statements -- actual path resolution goes through
# find_dir_with_tifs() (see below), because Kaggle's /kaggle/input mount
# layout has been observed to drift (confirmed via a live diagnostic run
# on 2026-09-15: the flat "/kaggle/input/<slug>/..." convention v2 trained
# under no longer exists; the real mount is now
# "/kaggle/input/datasets/<owner>/<slug>/..."). This is the same drift
# already documented and fixed the same way in
# kaggle_kernel_lookalike/train_lookalike_classifier.py's find_dir_with_tifs()
# -- reused verbatim here rather than re-solving the same problem differently.
PART1_DIR = "/kaggle/input/oil-spill-dartis-part1"
PART2_DIR = "/kaggle/input/oil-spill-dartis-part2"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
BATCH_SIZE = 16
PATCH_SIZE = 256
STRIDE = 256
EPOCHS = 15
LEARNING_RATE = 1e-3
SEED = 42

# Reference split fingerprints, reconstructed offline (see docstring above).
# get_real_dataloaders() below asserts its live-computed hashes match these
# before any patch extraction / training happens.
EXPECTED_TRAIN_SPLIT_SHA256 = "f57790486055a6034b7b1e62e872ce4880d6051cfb39f6b5fea0bd11fb8e17c7"
EXPECTED_VAL_SPLIT_SHA256 = "13a0a0e68b4f64ce1e527678d81a8c15a8909b6eb71a5cb957484914163674a6"
EXPECTED_TRAIN_COUNT = 2056
EXPECTED_VAL_COUNT = 514

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------------------------------------------
# STEP 0: RESILIENT DATASET PATH RESOLUTION (new in this v3 kernel)
# -------------------------------------------------------------
def find_dir_with_tifs(root, dirname):
    """Kaggle's actual /kaggle/input mount path layout doesn't reliably match
    the flat "<root>/oil-spill-dartis-part1/..." convention PART1_DIR/PART2_DIR
    assume. Ported verbatim from
    kaggle_kernel_lookalike/train_lookalike_classifier.py's find_dir_with_tifs(),
    which hit and fixed this exact drift already. Search for a directory whose
    name matches AND that directly contains .tif files (both Mask_oil/Mask_oil
    and Oil/Oil nest the name twice, so matching name alone isn't enough --
    must also confirm real file content one level in)."""
    for r, dirs, files in os.walk(root):
        if os.path.basename(r) == dirname and any(f.endswith(".tif") for f in files):
            return r
    return None

def resolve_dataset_paths():
    """Resolves all 6 real mask/image directories once via find_dir_with_tifs()
    and hard-fails immediately, listing what's missing, if any can't be found --
    rather than letting a later step fail on an empty-path symptom."""
    names = {
        "OIL_MASK_DIR": "Mask_oil", "OIL_IMG_DIR": "Oil",
        "NO_OIL_MASK_DIR": "no_oil_mask", "NO_OIL_IMG_DIR": "no_oil",
        "LOOKALIKE_MASK_DIR": "lookalike_mask", "LOOKALIKE_IMG_DIR": "lookalike",
    }
    resolved = {}
    missing = []
    for var_name, dirname in names.items():
        path = find_dir_with_tifs("/kaggle/input", dirname)
        resolved[var_name] = path
        if path is None:
            missing.append(dirname)
        else:
            print(f"  [+] {var_name} -> {path}")
    if missing:
        print(f"[!] HARD FAIL: could not locate directories for: {missing}")
        for r, dirs, files in os.walk("/kaggle/input"):
            if files:
                print(r, "->", files[:3], f"({len(files)} files)")
        raise ValueError(f"[!] HARD FAIL: missing dataset directories: {missing}")
    return resolved

print("\n=== STEP 0: RESOLVING DATASET PATHS (drift-resilient) ===")
_PATHS = resolve_dataset_paths()
OIL_MASK_DIR = _PATHS["OIL_MASK_DIR"]
OIL_IMG_DIR = _PATHS["OIL_IMG_DIR"]
NO_OIL_MASK_DIR = _PATHS["NO_OIL_MASK_DIR"]
NO_OIL_IMG_DIR = _PATHS["NO_OIL_IMG_DIR"]
LOOKALIKE_MASK_DIR = _PATHS["LOOKALIKE_MASK_DIR"]
LOOKALIKE_IMG_DIR = _PATHS["LOOKALIKE_IMG_DIR"]

# -------------------------------------------------------------
# STEP 1: RESOLVE PART II DISCREPANCY CHECK
# -------------------------------------------------------------
def check_part2_discrepancy():
    print("\n=== STEP 1: PART II DISCREPANCY CHECK ===")
    no_oil_mask_dir = NO_OIL_MASK_DIR
    lookalike_mask_dir = LOOKALIKE_MASK_DIR

    if not os.path.exists(no_oil_mask_dir) or not os.path.exists(lookalike_mask_dir):
        print("[!] Error: Part II mask folders not found at expected input paths.")
        return

    no_oil_files = sorted(os.listdir(no_oil_mask_dir))
    lookalike_files = sorted(os.listdir(lookalike_mask_dir))

    print(f"  - No Oil Mask file count: {len(no_oil_files)}")
    print(f"  - Lookalike Mask file count: {len(lookalike_files)}")

    # Hash comparison
    matches = 0
    total_compared = min(len(no_oil_files), len(lookalike_files))
    for i in range(total_compared):
        f1 = os.path.join(no_oil_mask_dir, no_oil_files[i])
        f2 = os.path.join(lookalike_mask_dir, lookalike_files[i])
        h1 = hashlib.md5(open(f1, 'rb').read()).hexdigest()
        h2 = hashlib.md5(open(f2, 'rb').read()).hexdigest()
        if h1 == h2:
            matches += 1
            if i < 5:
                print(f"    File {no_oil_files[i]}: No-Oil Hash = {h1} | Lookalike Hash = {h2} | Identical = True")
        else:
            if i < 5:
                print(f"    File {no_oil_files[i]}: No-Oil Hash = {h1} | Lookalike Hash = {h2} | Identical = False")

    print(f"  [+] Discrepancy comparison summary: {matches}/{total_compared} mask files are identical between No-Oil and Lookalike categories.")

# -------------------------------------------------------------
# STEP 2: HARD-FAIL IMAGE VERIFICATION
# -------------------------------------------------------------
def verify_real_images():
    print("\n=== STEP 2: REAL DATA VERIFICATION ===")
    paths_to_check = [
        OIL_IMG_DIR, OIL_MASK_DIR,
        NO_OIL_IMG_DIR, NO_OIL_MASK_DIR,
        LOOKALIKE_IMG_DIR, LOOKALIKE_MASK_DIR,
    ]

    for path in paths_to_check:
        if not os.path.exists(path) or len(os.listdir(path)) == 0:
            raise ValueError(f"[!] HARD FAIL: Dataset path '{path}' is missing or empty!")

    # Load check on a sample image
    test_img_path = os.path.join(OIL_IMG_DIR, "00000.tif")
    test_img = tifffile.imread(test_img_path)
    print(f"  [+] Loaded test image {os.path.basename(test_img_path)}")
    print(f"      Dimensions: {test_img.shape} | Data Type: {test_img.dtype}")

    # Assert size is 2048x2048 and has 2 channels
    if test_img.shape != (2048, 2048, 2) and test_img.shape != (2, 2048, 2048):
        raise ValueError(f"[!] HARD FAIL: Image shape is invalid: {test_img.shape}")

    print("[✔] Hard verification passed! Real satellite images are fully loaded and correct.")

# -------------------------------------------------------------
# STEP 3: PREPROCESSING & OPTIMIZED PATCH LOADING (MEMORY SAFE)
# -------------------------------------------------------------
# WARNING: Synced copy of src/data/preprocess.py (calibrate_to_sigma, speckle_filter,
# to_decibels, normalize_image, preprocess_for_prediction, extract_patches_balanced,
# run_full_preprocessing) and src/data/dataset.py (get_real_dataloaders). Kaggle
# kernels must be a single self-contained file, so these can't be imported directly
# the way the rest of this codebase does -- if you change the category oversampling
# rates, the hard-fail behavior, or the dB-scale handling in either
# src/data/preprocess.py or src/data/dataset.py, mirror the change here too, and
# vice versa. This v3 fork additionally adds a split-fingerprint hard-fail check
# (see get_real_dataloaders() below) not present in v2's kernel.
#
# Forked from kaggle_kernel/train_kaggle.py as of its 2026-09-05 sync point.
def calibrate_to_sigma(image_dn, calibration_constant=1.0):
    return (image_dn.astype(np.float32) ** 2) * calibration_constant

def speckle_filter(image, window_size=5):
    if len(image.shape) == 3:
        filtered = np.zeros_like(image)
        for c in range(image.shape[2]):
            filtered[:, :, c] = cv2.blur(image[:, :, c], (window_size, window_size))
        return filtered
    return cv2.blur(image, (window_size, window_size))

def to_decibels(image_linear, eps=1e-5):
    clipped = np.clip(image_linear, eps, None)
    return 10.0 * np.log10(clipped)

def normalize_image(image_db, min_db=-25.0, max_db=0.0):
    clipped = np.clip(image_db, min_db, max_db)
    return (clipped - min_db) / (max_db - min_db)
def extract_patches_balanced(image, mask, patch_size=256, stride=128, category="oil"):
    """
    Extracts patches from a 2048x2048 image.
    Optimizes memory footprint to prevent Kaggle OOM (16GB RAM limit).
    Keeps 100% of patches with positive slick pixels.
    Downsamples background patches based on category:
      - oil: 1.0% (prob = 0.01)
      - lookalike: 0.5% (prob = 0.005) -> Oversampled 2.5x relative to clean sea!
      - no_oil: 0.2% (prob = 0.002)
    Forces copies to prevent NumPy views from holding the large base image in memory.
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
    # Branches on np.any(image_raw < 0) to tell already-dB-scale real Sentinel-1
    # imagery from linear-scale data. This kernel's inputs are always the former.
    if len(image_raw.shape) == 3 and image_raw.shape[0] == 2:
        image_raw = image_raw.transpose(1, 2, 0)

    if np.any(image_raw < 0):
        # Already in dB scale (the real Sentinel-1 training/holdout imagery).
        # Back to linear for speckle filtering, then to dB, then normalize.
        linear = 10.0 ** (image_raw.astype(np.float32) / 10.0)
        filt = speckle_filter(linear, window_size=5)
        db = 10.0 * np.log10(np.clip(filt, 1e-5, None))
        return normalize_image(db)
    else:
        sig = image_raw.astype(np.float32) if already_calibrated else calibrate_to_sigma(image_raw)
        filt = speckle_filter(sig, window_size=5)
        db = to_decibels(filt)
        return normalize_image(db)

def run_full_preprocessing(image_raw, mask_raw, patch_size=256, stride=128, category="oil"):
    norm = preprocess_for_prediction(image_raw)
    return extract_patches_balanced(norm, mask_raw, patch_size, stride, category)
class SARDataset(Dataset):
    def __init__(self, image_patches, mask_patches, augment=False):
        self.images = image_patches
        self.masks = mask_patches
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()
        mask = self.masks[idx].copy()

        if self.augment:
            if random.random() > 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)
            if random.random() > 0.5:
                img = np.flipud(img)
                mask = np.flipud(mask)
            rot_k = random.choice([0, 1, 2, 3])
            if rot_k > 0:
                img = np.rot90(img, k=rot_k)
                mask = np.rot90(mask, k=rot_k)

        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)

        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask_binary = (mask > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).float()

        return img_tensor, mask_tensor

def get_real_dataloaders(patch_size=256, stride=128, batch_size=8, train_ratio=0.8, seed=42):
    # 1. Load Part I (Oil Spill)
    oil_mask_dir = OIL_MASK_DIR
    oil_img_dir = OIL_IMG_DIR
    oil_masks = sorted(glob.glob(os.path.join(oil_mask_dir, "*.tif")))

    all_pairs = []
    print(f"[*] Pairing {len(oil_masks)} masks for category Oil Spill")
    for mask_path in oil_masks:
        filename = os.path.basename(mask_path)
        img_path = os.path.join(oil_img_dir, filename)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"[!] Hard-fail: Matching image not found: {img_path}")
        all_pairs.append({
            "mask_path": mask_path,
            "img_path": img_path,
            "category": "oil"
        })

    # 2. Compare and Deduplicate Part II at the IMAGE level (not mask level)
    lookalike_mask_dir = LOOKALIKE_MASK_DIR
    lookalike_img_dir = LOOKALIKE_IMG_DIR
    lookalike_masks = sorted(glob.glob(os.path.join(lookalike_mask_dir, "*.tif")))

    no_oil_mask_dir = NO_OIL_MASK_DIR
    no_oil_img_dir = NO_OIL_IMG_DIR
    no_oil_masks = sorted(glob.glob(os.path.join(no_oil_mask_dir, "*.tif")))

    # Map common filenames and check image-level duplicates
    lookalike_filenames = {os.path.basename(p): p for p in lookalike_masks}
    common_filenames = sorted(list(set(lookalike_filenames.keys()) & set([os.path.basename(p) for p in no_oil_masks])))

    print(f"\n[*] Performing image-level duplication check for {len(common_filenames)} common filenames...")
    img_duplicates = 0
    t0 = time.time()
    for fname in common_filenames:
        p1 = os.path.join(lookalike_img_dir, fname)
        p2 = os.path.join(no_oil_img_dir, fname)
        # Check size first (instant)
        sz1 = os.path.getsize(p1)
        sz2 = os.path.getsize(p2)
        if sz1 == sz2:
            h1 = hashlib.md5(open(p1, 'rb').read()).hexdigest()
            h2 = hashlib.md5(open(p2, 'rb').read()).hexdigest()
            if h1 == h2:
                img_duplicates += 1
    print(f"  - Duplication check completed in {time.time() - t0:.2f} seconds.")
    print(f"  - Found {img_duplicates} identical image files out of {len(common_filenames)} compared.")

    # Store lookalike image hashes only if duplicates exist
    lookalike_image_hashes = {}
    if img_duplicates > 0:
        for fname in common_filenames:
            p_img = os.path.join(lookalike_img_dir, fname)
            sz = os.path.getsize(p_img)
            h = hashlib.md5(open(p_img, 'rb').read()).hexdigest()
            lookalike_image_hashes[(sz, h)] = p_img

    deduped_no_oil_masks = []
    removed_count = 0
    for mask_path in no_oil_masks:
        if img_duplicates > 0:
            filename = os.path.basename(mask_path)
            p_img = os.path.join(no_oil_img_dir, filename)
            sz = os.path.getsize(p_img)
            h = hashlib.md5(open(p_img, 'rb').read()).hexdigest()
            if (sz, h) in lookalike_image_hashes:
                removed_count += 1
                continue
        deduped_no_oil_masks.append(mask_path)

    print(f"\n[Deduplication Summary]")
    print(f"  - Removed {removed_count} duplicate No-Oil scenes based on image identity.")
    print(f"  - Kept {len(deduped_no_oil_masks)} distinct No-Oil scenes.")
    print(f"  - Kept {len(lookalike_masks)} distinct Lookalike scenes.")
    print(f"  - Total distinct negative scenes: {len(lookalike_masks) + len(deduped_no_oil_masks)}")

    # 3. Add Lookalike pairs
    for mask_path in lookalike_masks:
        filename = os.path.basename(mask_path)
        img_path = os.path.join(lookalike_img_dir, filename)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"[!] Hard-fail: Matching image not found: {img_path}")
        all_pairs.append({
            "mask_path": mask_path,
            "img_path": img_path,
            "category": "lookalike"
        })

    # 4. Add Deduplicated No-Oil pairs
    for mask_path in deduped_no_oil_masks:
        filename = os.path.basename(mask_path)
        img_path = os.path.join(no_oil_img_dir, filename)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"[!] Hard-fail: Matching image not found: {img_path}")
        all_pairs.append({
            "mask_path": mask_path,
            "img_path": img_path,
            "category": "no_oil"
        })

    random.seed(seed)
    random.shuffle(all_pairs)

    num_total = len(all_pairs)
    num_train = int(num_total * train_ratio)

    train_pairs = all_pairs[:num_train]
    val_pairs = all_pairs[num_train:]

    print(f"\n[*] Split distribution:")
    print(f"    - Train split: {len(train_pairs)} scenes")
    print(f"    - Val split:   {len(val_pairs)} scenes")

    # -------------------------------------------------------------
    # SPLIT-FINGERPRINT HARD-FAIL CHECK (new in this v3 kernel, not in v2)
    # -------------------------------------------------------------
    # Reproduces the exact fingerprint algorithm used to reconstruct v2's split
    # offline: sort "<category>/<filename>" strings per side, join with "\n",
    # SHA-256 the result. If this doesn't match the reference hashes computed
    # from the same deterministic seed/shuffle against the same mask filename
    # sets, something about the input data or algorithm has drifted since v2
    # trained -- hard-fail rather than silently run an uncontrolled comparison.
    def _fingerprint(pairs):
        ids = sorted(f"{p['category']}/{os.path.basename(p['mask_path'])}" for p in pairs)
        return hashlib.sha256("\n".join(ids).encode()).hexdigest()

    train_fp = _fingerprint(train_pairs)
    val_fp = _fingerprint(val_pairs)

    print(f"\n[*] Split-fingerprint check (vs. v2's reconstructed split):")
    print(f"    - Train: {len(train_pairs)} scenes (expected {EXPECTED_TRAIN_COUNT}) | SHA256 {train_fp}")
    print(f"    - Val:   {len(val_pairs)} scenes (expected {EXPECTED_VAL_COUNT}) | SHA256 {val_fp}")

    fingerprint_report = {
        "expected_train_count": EXPECTED_TRAIN_COUNT,
        "actual_train_count": len(train_pairs),
        "expected_val_count": EXPECTED_VAL_COUNT,
        "actual_val_count": len(val_pairs),
        "expected_train_sha256": EXPECTED_TRAIN_SPLIT_SHA256,
        "actual_train_sha256": train_fp,
        "expected_val_sha256": EXPECTED_VAL_SPLIT_SHA256,
        "actual_val_sha256": val_fp,
        "train_match": train_fp == EXPECTED_TRAIN_SPLIT_SHA256,
        "val_match": val_fp == EXPECTED_VAL_SPLIT_SHA256,
    }
    os.makedirs("/kaggle/working", exist_ok=True)
    with open("/kaggle/working/split_fingerprint_v3.json", "w") as f:
        import json
        json.dump(fingerprint_report, f, indent=2)

    if not (fingerprint_report["train_match"] and fingerprint_report["val_match"]):
        print("[!!!] SPLIT FINGERPRINT MISMATCH -- this run's train/val assignment does")
        print("      NOT match v2's split. Aborting before any GPU time is spent on an")
        print("      uncontrolled comparison. See /kaggle/working/split_fingerprint_v3.json.")
        raise RuntimeError("Split fingerprint mismatch: this run is not comparable to v2.")
    print("[✔] Split fingerprint MATCHES v2's split exactly -- comparison is controlled.")

    def process_pairs(pairs, name):
        import gc
        img_patches = []
        mask_patches = []
        print(f"[*] Preprocessing and extracting balanced patches for {name} split...")
        for i, pair in enumerate(pairs):
            mask_raw = cv2.imread(pair["mask_path"], cv2.IMREAD_GRAYSCALE)
            image_raw = tifffile.imread(pair["img_path"])

            img_p, mask_p = run_full_preprocessing(image_raw, mask_raw, patch_size, stride, pair["category"])
            img_patches.extend(img_p)
            mask_patches.extend(mask_p)

            # Explicitly delete large arrays and force garbage collection to reclaim memory
            del mask_raw, image_raw, img_p, mask_p
            if (i+1) % 100 == 0:
                gc.collect()

            if (i+1) % 200 == 0 or (i+1) == len(pairs):
                print(f"    - Processed {i+1}/{len(pairs)} files...")

        gc.collect()
        print(f"    [+] Created {len(img_patches)} balanced patches for {name} split.")
        return img_patches, mask_patches

    train_imgs, train_masks = process_pairs(train_pairs, "Train")
    val_imgs, val_masks = process_pairs(val_pairs, "Val")

    train_dataset = SARDataset(train_imgs, train_masks, augment=True)
    val_dataset = SARDataset(val_imgs, val_masks, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# -------------------------------------------------------------
# STEP 4: MODEL DEFINITION (U-NET) & LOSS & METRICS
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

class FocalIoULoss(nn.Module):
    """
    Focal Loss + soft IoU Loss, matching the reference paper's (Trujillo-
    Acatitla et al.) reported loss setup. Replaces v2's BCEDiceLoss -- the
    ONLY change in this v3 ablation kernel.

    Focal Loss (Lin et al. 2017 defaults, alpha=0.25/gamma=2.0) down-weights
    easy (background) pixels and focuses training on hard/misclassified ones.
    Soft IoU loss is 1 - (intersection / union) computed on sigmoid
    probabilities, directly optimizing the same metric used for evaluation.
    Combined 1:1, matching v2's BCEDiceLoss(alpha=1.0, beta=1.0) weighting
    convention so the only real change is which two terms are being summed.
    """
    def __init__(self, alpha=0.25, gamma=2.0, focal_weight=1.0, iou_weight=1.0):
        super(FocalIoULoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.iou_weight = iou_weight

    def forward(self, inputs, targets):
        # Focal loss (from logits, numerically stable via BCE-with-logits per element)
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = (alpha_t * (1 - p_t).pow(self.gamma) * bce).mean()

        # Soft IoU loss
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum() - intersection
        iou = (intersection + 1e-5) / (union + 1e-5)
        iou_loss = 1.0 - iou

        return self.focal_weight * focal_loss + self.iou_weight * iou_loss

def calculate_metrics(outputs, targets, threshold=0.5):
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum().item()
    union = preds_flat.sum().item() + targets_flat.sum().item() - intersection

    iou = (intersection + 1e-5) / (union + 1e-5)
    dice = (2. * intersection + 1e-5) / (preds_flat.sum().item() + targets_flat.sum().item() + 1e-5)

    return iou, dice

# -------------------------------------------------------------
# STEP 5: TRAINING ORCHESTRATION WITH GPU & MIXED PRECISION
# -------------------------------------------------------------
def train_and_evaluate():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        if cap[0] >= 7:
            device = torch.device("cuda")
            print(f"\n[*] Compatible GPU found: {torch.cuda.get_device_name(0)} (CUDA Capability {cap[0]}.{cap[1]})")
        else:
            print(f"\n[!] Warning: GPU {torch.cuda.get_device_name(0)} (CUDA Capability {cap[0]}.{cap[1]}) is incompatible with pre-installed PyTorch. Falling back to CPU to prevent crash.")
    else:
        print("\n[*] No GPU available, using CPU.")
    # -------------------------------------------------------------
    # PRE-FLIGHT REAL IMAGE NORMALIZATION DIAGNOSTIC
    # -------------------------------------------------------------
    print("\n=== PRE-FLIGHT REAL IMAGE NORMALIZATION DIAGNOSTIC ===")
    oil_img_dir = OIL_IMG_DIR
    test_paths = []
    if os.path.exists(oil_img_dir):
        oil_imgs = sorted(glob.glob(os.path.join(oil_img_dir, "*.tif")))
        test_paths = oil_imgs[:3]

    if not test_paths:
        print("[!] Warning: No real images found at expected paths for diagnostic check.")
    else:
        diagnostic_failed = False
        for path in test_paths:
            fname = os.path.basename(path)
            raw_img = tifffile.imread(path)

            print(f"  - File {fname} Raw Stats:")
            print(f"    Min: {raw_img.min():.4f} | Max: {raw_img.max():.4f} | Mean: {raw_img.mean():.4f}")

            if len(raw_img.shape) == 3 and raw_img.shape[0] == 2:
                img_proc = raw_img.transpose(1, 2, 0)
            else:
                img_proc = raw_img

            # Run decibel-aware normalization
            linear = 10.0 ** (img_proc.astype(np.float32) / 10.0)
            filtered = speckle_filter(linear, window_size=5)
            db = 10.0 * np.log10(np.clip(filtered, 1e-5, None))
            min_db, max_db = -25.0, 0.0
            norm = (np.clip(db, min_db, max_db) - min_db) / (max_db - min_db)

            print(f"    Normalized Stats (Post-Fix):")
            print(f"    Min: {norm.min():.4f} | Max: {norm.max():.4f} | Mean: {norm.mean():.4f} | Std: {norm.std():.4f}")

            if norm.std() < 1e-3:
                print(f"[!] ERROR: Normalized image has collapsed (std = {norm.std():.4e})")
                diagnostic_failed = True
            elif np.allclose(norm, 1.0) or np.allclose(norm, 0.0):
                print("[!] ERROR: Normalized image is flat 0.0 or 1.0.")
                diagnostic_failed = True

        if diagnostic_failed:
            print("[!] CRITICAL: Pre-flight diagnostic check FAILED. Aborting training to prevent resource waste.")
            sys.exit(1)
        else:
            print("[✔] Pre-flight diagnostic check PASSED. Normalized images show healthy variance.")

    # 1. Initialize loaders
    train_loader, val_loader = get_real_dataloaders(
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        train_ratio=0.8,
        seed=SEED
    )

    # 2. Build model and load state
    model = UNet(in_channels=2, out_channels=1).to(device)
    criterion = FocalIoULoss(alpha=0.25, gamma=2.0, focal_weight=1.0, iou_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print("\n=== STARTING TRAINING FRESH FROM SCRATCH (v3: Focal+IoU loss) ===")

    log_file_path = "/kaggle/working/training_v3_log.csv"
    with open(log_file_path, "w") as lf:
        lf.write("epoch,train_loss,train_iou,val_loss,val_iou,val_dice,active_prediction_pixels\n")

    best_val_iou = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_ious = []
        start_time = time.time()

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            # Autocast mixed precision
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            iou, _ = calculate_metrics(outputs, masks)
            train_ious.append(iou)

        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_iou = np.mean(train_ious)

        # Validation Loop (with Global Metric Pooling)
        model.eval()
        val_loss = 0.0
        total_intersection = 0.0
        total_union = 0.0
        total_preds_sum = 0.0
        total_targets_sum = 0.0
        total_active_pixels = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                preds_flat = preds.view(-1)
                targets_flat = masks.view(-1)

                intersection = (preds_flat * targets_flat).sum().item()
                union = preds_flat.sum().item() + targets_flat.sum().item() - intersection

                total_intersection += intersection
                total_union += union
                total_preds_sum += preds_flat.sum().item()
                total_targets_sum += targets_flat.sum().item()
                total_active_pixels += preds.sum().item()

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_iou = (total_intersection + 1e-5) / (total_union + 1e-5)
        epoch_val_dice = (2.0 * total_intersection + 1e-5) / (total_preds_sum + total_targets_sum + 1e-5)
        elapsed = time.time() - start_time

        # Advance learning rate scheduler
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train IoU: {epoch_train_iou:.4f}")
        print(f"  Val Loss:   {epoch_val_loss:.4f} | Val IoU:   {epoch_val_iou:.4f} | Val Dice: {epoch_val_dice:.4f}")
        print(f"  Active Pixels predicted: {total_active_pixels:.0f}")

        # Log to file
        with open(log_file_path, "a") as lf:
            lf.write(f"{epoch},{epoch_train_loss:.6f},{epoch_train_iou:.6f},{epoch_val_loss:.6f},{epoch_val_iou:.6f},{epoch_val_dice:.6f},{total_active_pixels}\n")

        # Check collapse anomaly (if prediction collapses to all-zeros)
        if epoch >= 3 and total_active_pixels == 0:
            print("[!!!] WARNING: Model predictions collapsed to all-zero background masks. Stopping early.")
            break

        # Save best model
        if epoch_val_iou > best_val_iou:
            best_val_iou = epoch_val_iou
            torch.save(model.state_dict(), "/kaggle/working/model_real_v3_best.pt")
            print("  [+] Saved new best model checkpoint: model_real_v3_best.pt")

        # Save periodic checkpoints (for resume safety)
        if epoch % 5 == 0:
            chk_path = f"/kaggle/working/model_real_v3_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': epoch_val_iou
            }, chk_path)
            print(f"  [+] Periodic backup saved: {os.path.basename(chk_path)}")

    # Save final model
    torch.save(model.state_dict(), "/kaggle/working/model_real_v3_final.pt")
    print("\n[+] Training complete. Saved final model: model_real_v3_final.pt")

if __name__ == "__main__":
    check_part2_discrepancy()
    verify_real_images()
    train_and_evaluate()
