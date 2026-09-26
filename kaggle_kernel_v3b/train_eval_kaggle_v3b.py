"""
Project Pelagic -- Phase 3b: class-weighted aux head + full in-kernel evaluation

Forked from kaggle_kernel_v3_aux/train_kaggle_v3_aux.py (which itself forks
kaggle_kernel_v3/train_kaggle_v3.py). Segmentation path (FocalIoULoss, data
pipeline, split seed/ratio, batch size, epochs, optimizer, LR schedule,
split-fingerprint hard-fail gate, find_dir_with_tifs() path resolution) is
unchanged -- diff against train_kaggle_v3_aux.py to confirm.

ONE targeted change to Part A (the fix): the auxiliary classification
cross-entropy is now weighted inversely to the REAL, MEASURED oil-vs-
lookalike patch counts in this run's own train split (not a hardcoded/
assumed ratio) -- computed once, right after process_pairs() builds the
training patches, before the model is even constructed. aux_weight stays
fixed at 0.3 (the stable variant from Phase 3; no reason to repeat the
unstable 1.0 run). Segmentation loss, oversampling rates, and everything
else in the pipeline are untouched.

Part B: this kernel ALSO runs the full 6-way holdout evaluation itself,
right after training, using two additional dataset_sources (mounted
read-only, drift-resilient path resolution applied to both):
  - trongzen/project-pelagic-model-best: v1/v2/v3/v3-aux-0.3/v3-aux-1.0
    checkpoints (already existed for v1/v2; Phase 3b's setup step added
    v3/v3-aux-0.3/v3-aux-1.0 to it).
  - trongzen/project-pelagic-holdout-30: the exact same 30-scene holdout
    (10 oil/10 no_oil/10 lookalike) used in every prior phase's comparison
    table, packaged fresh this phase specifically so evaluation never needs
    to leave Kaggle and never risks silently evaluating a different subset
    of Part III than history has used.
Nothing downloads back to the local machine except this kernel's own
printed log.
"""
import os
import sys
import glob
import time
import random
import hashlib
import shutil
import subprocess

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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURATION ---
AUX_WEIGHT = 0.3
MODEL_TAG = "v3b"   # -> model_real_v3b_best.pt, training_v3b_log.csv, etc.

PART1_DIR = "/kaggle/input/oil-spill-dartis-part1"
PART2_DIR = "/kaggle/input/oil-spill-dartis-part2"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
BATCH_SIZE = 16
PATCH_SIZE = 256
STRIDE = 256
EPOCHS = 15
LEARNING_RATE = 1e-3
SEED = 42

EXPECTED_TRAIN_SPLIT_SHA256 = "f57790486055a6034b7b1e62e872ce4880d6051cfb39f6b5fea0bd11fb8e17c7"
EXPECTED_VAL_SPLIT_SHA256 = "13a0a0e68b4f64ce1e527678d81a8c15a8909b6eb71a5cb957484914163674a6"
EXPECTED_TRAIN_COUNT = 2056
EXPECTED_VAL_COUNT = 514

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------------------------------------------
# STEP 0: RESILIENT DATASET PATH RESOLUTION
# -------------------------------------------------------------
def find_dir_with_tifs(root, dirname):
    """Ported verbatim from kaggle_kernel_lookalike/train_lookalike_classifier.py."""
    for r, dirs, files in os.walk(root):
        if os.path.basename(r) == dirname and any(f.endswith(".tif") for f in files):
            return r
    return None

def find_file(root, filename):
    """Search for a specific file by name anywhere under root. Used for the
    checkpoint dataset (mount path has drifted before -- same reasoning as
    find_dir_with_tifs, applied to individual files instead of directories)."""
    for r, dirs, files in os.walk(root):
        if filename in files:
            return os.path.join(r, filename)
    return None

def find_holdout_dirs(root):
    """The holdout dataset (trongzen/project-pelagic-holdout-30) has an
    images/ and masks/ dir, each with exactly 30 .tif files. Search by that
    signature rather than a hardcoded mount path."""
    images_dir, masks_dir = None, None
    for r, dirs, files in os.walk(root):
        tifs = [f for f in files if f.endswith(".tif")]
        base = os.path.basename(r)
        if base == "images" and len(tifs) == 30:
            images_dir = r
        elif base == "masks" and len(tifs) == 30:
            masks_dir = r
    return images_dir, masks_dir

def resolve_dataset_paths():
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

print(f"\n=== STEP 0: RESOLVING DATASET PATHS (drift-resilient) [AUX_WEIGHT={AUX_WEIGHT}, TAG={MODEL_TAG}] ===")
_PATHS = resolve_dataset_paths()
OIL_MASK_DIR = _PATHS["OIL_MASK_DIR"]
OIL_IMG_DIR = _PATHS["OIL_IMG_DIR"]
NO_OIL_MASK_DIR = _PATHS["NO_OIL_MASK_DIR"]
NO_OIL_IMG_DIR = _PATHS["NO_OIL_IMG_DIR"]
LOOKALIKE_MASK_DIR = _PATHS["LOOKALIKE_MASK_DIR"]
LOOKALIKE_IMG_DIR = _PATHS["LOOKALIKE_IMG_DIR"]

# -------------------------------------------------------------
# STEP 1: PART II DISCREPANCY CHECK
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
    matches = 0
    total_compared = min(len(no_oil_files), len(lookalike_files))
    for i in range(total_compared):
        f1 = os.path.join(no_oil_mask_dir, no_oil_files[i])
        f2 = os.path.join(lookalike_mask_dir, lookalike_files[i])
        h1 = hashlib.md5(open(f1, 'rb').read()).hexdigest()
        h2 = hashlib.md5(open(f2, 'rb').read()).hexdigest()
        if h1 == h2:
            matches += 1
    print(f"  [+] Discrepancy comparison summary: {matches}/{total_compared} mask files are identical between No-Oil and Lookalike categories.")

# -------------------------------------------------------------
# STEP 2: HARD-FAIL IMAGE VERIFICATION
# -------------------------------------------------------------
def verify_real_images():
    print("\n=== STEP 2: REAL DATA VERIFICATION ===")
    paths_to_check = [OIL_IMG_DIR, OIL_MASK_DIR, NO_OIL_IMG_DIR, NO_OIL_MASK_DIR, LOOKALIKE_IMG_DIR, LOOKALIKE_MASK_DIR]
    for path in paths_to_check:
        if not os.path.exists(path) or len(os.listdir(path)) == 0:
            raise ValueError(f"[!] HARD FAIL: Dataset path '{path}' is missing or empty!")
    test_img_path = os.path.join(OIL_IMG_DIR, "00000.tif")
    test_img = tifffile.imread(test_img_path)
    print(f"  [+] Loaded test image {os.path.basename(test_img_path)}")
    print(f"      Dimensions: {test_img.shape} | Data Type: {test_img.dtype}")
    if test_img.shape != (2048, 2048, 2) and test_img.shape != (2, 2048, 2048):
        raise ValueError(f"[!] HARD FAIL: Image shape is invalid: {test_img.shape}")
    print("[OK] Hard verification passed! Real satellite images are fully loaded and correct.")

# -------------------------------------------------------------
# STEP 3: PREPROCESSING & PATCH LOADING
# -------------------------------------------------------------
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
    if len(image_raw.shape) == 3 and image_raw.shape[0] == 2:
        image_raw = image_raw.transpose(1, 2, 0)
    if np.any(image_raw < 0):
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

# oil=1, lookalike=0, no_oil=-1 (ignore_index). Same convention as Phase 3.
CATEGORY_TO_CLS_LABEL = {"oil": 1, "lookalike": 0, "no_oil": -1}

class SARDataset(Dataset):
    def __init__(self, image_patches, mask_patches, cls_labels, augment=False):
        self.images = image_patches
        self.masks = mask_patches
        self.cls_labels = cls_labels
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()
        mask = self.masks[idx].copy()
        cls_label = self.cls_labels[idx]
        if self.augment:
            if random.random() > 0.5:
                img = np.fliplr(img); mask = np.fliplr(mask)
            if random.random() > 0.5:
                img = np.flipud(img); mask = np.flipud(mask)
            rot_k = random.choice([0, 1, 2, 3])
            if rot_k > 0:
                img = np.rot90(img, k=rot_k); mask = np.rot90(mask, k=rot_k)
        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask_binary = (mask > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).float()
        cls_tensor = torch.tensor(cls_label, dtype=torch.long)
        return img_tensor, mask_tensor, cls_tensor

def get_real_dataloaders(patch_size=256, stride=128, batch_size=8, train_ratio=0.8, seed=42):
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
        all_pairs.append({"mask_path": mask_path, "img_path": img_path, "category": "oil"})

    lookalike_mask_dir = LOOKALIKE_MASK_DIR
    lookalike_img_dir = LOOKALIKE_IMG_DIR
    lookalike_masks = sorted(glob.glob(os.path.join(lookalike_mask_dir, "*.tif")))
    no_oil_mask_dir = NO_OIL_MASK_DIR
    no_oil_img_dir = NO_OIL_IMG_DIR
    no_oil_masks = sorted(glob.glob(os.path.join(no_oil_mask_dir, "*.tif")))

    lookalike_filenames = {os.path.basename(p): p for p in lookalike_masks}
    common_filenames = sorted(list(set(lookalike_filenames.keys()) & set([os.path.basename(p) for p in no_oil_masks])))

    print(f"\n[*] Performing image-level duplication check for {len(common_filenames)} common filenames...")
    img_duplicates = 0
    t0 = time.time()
    for fname in common_filenames:
        p1 = os.path.join(lookalike_img_dir, fname)
        p2 = os.path.join(no_oil_img_dir, fname)
        sz1 = os.path.getsize(p1)
        sz2 = os.path.getsize(p2)
        if sz1 == sz2:
            h1 = hashlib.md5(open(p1, 'rb').read()).hexdigest()
            h2 = hashlib.md5(open(p2, 'rb').read()).hexdigest()
            if h1 == h2:
                img_duplicates += 1
    print(f"  - Duplication check completed in {time.time() - t0:.2f} seconds.")
    print(f"  - Found {img_duplicates} identical image files out of {len(common_filenames)} compared.")

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

    for mask_path in lookalike_masks:
        filename = os.path.basename(mask_path)
        img_path = os.path.join(lookalike_img_dir, filename)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"[!] Hard-fail: Matching image not found: {img_path}")
        all_pairs.append({"mask_path": mask_path, "img_path": img_path, "category": "lookalike"})

    for mask_path in deduped_no_oil_masks:
        filename = os.path.basename(mask_path)
        img_path = os.path.join(no_oil_img_dir, filename)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"[!] Hard-fail: Matching image not found: {img_path}")
        all_pairs.append({"mask_path": mask_path, "img_path": img_path, "category": "no_oil"})

    random.seed(seed)
    random.shuffle(all_pairs)

    num_total = len(all_pairs)
    num_train = int(num_total * train_ratio)
    train_pairs = all_pairs[:num_train]
    val_pairs = all_pairs[num_train:]

    print(f"\n[*] Split distribution:")
    print(f"    - Train split: {len(train_pairs)} scenes")
    print(f"    - Val split:   {len(val_pairs)} scenes")

    def _fingerprint(pairs):
        ids = sorted(f"{p['category']}/{os.path.basename(p['mask_path'])}" for p in pairs)
        return hashlib.sha256("\n".join(ids).encode()).hexdigest()

    train_fp = _fingerprint(train_pairs)
    val_fp = _fingerprint(val_pairs)

    print(f"\n[*] Split-fingerprint check (vs. v2/v3's reconstructed split):")
    print(f"    - Train: {len(train_pairs)} scenes (expected {EXPECTED_TRAIN_COUNT}) | SHA256 {train_fp}")
    print(f"    - Val:   {len(val_pairs)} scenes (expected {EXPECTED_VAL_COUNT}) | SHA256 {val_fp}")

    fingerprint_report = {
        "aux_weight": AUX_WEIGHT, "model_tag": MODEL_TAG,
        "expected_train_count": EXPECTED_TRAIN_COUNT, "actual_train_count": len(train_pairs),
        "expected_val_count": EXPECTED_VAL_COUNT, "actual_val_count": len(val_pairs),
        "expected_train_sha256": EXPECTED_TRAIN_SPLIT_SHA256, "actual_train_sha256": train_fp,
        "expected_val_sha256": EXPECTED_VAL_SPLIT_SHA256, "actual_val_sha256": val_fp,
        "train_match": train_fp == EXPECTED_TRAIN_SPLIT_SHA256,
        "val_match": val_fp == EXPECTED_VAL_SPLIT_SHA256,
    }
    os.makedirs("/kaggle/working", exist_ok=True)
    with open(f"/kaggle/working/split_fingerprint_{MODEL_TAG}.json", "w") as f:
        import json
        json.dump(fingerprint_report, f, indent=2)

    if not (fingerprint_report["train_match"] and fingerprint_report["val_match"]):
        print("[!!!] SPLIT FINGERPRINT MISMATCH -- aborting before any GPU time is spent.")
        raise RuntimeError("Split fingerprint mismatch: this run is not comparable to v2/v3.")
    print("[OK] Split fingerprint MATCHES v2/v3's split exactly -- comparison is controlled.")

    def process_pairs(pairs, name):
        import gc
        img_patches = []
        mask_patches = []
        cls_patches = []
        print(f"[*] Preprocessing and extracting balanced patches for {name} split...")
        for i, pair in enumerate(pairs):
            mask_raw = cv2.imread(pair["mask_path"], cv2.IMREAD_GRAYSCALE)
            image_raw = tifffile.imread(pair["img_path"])
            img_p, mask_p = run_full_preprocessing(image_raw, mask_raw, patch_size, stride, pair["category"])
            img_patches.extend(img_p)
            mask_patches.extend(mask_p)
            cls_label = CATEGORY_TO_CLS_LABEL[pair["category"]]
            cls_patches.extend([cls_label] * len(img_p))
            del mask_raw, image_raw, img_p, mask_p
            if (i+1) % 100 == 0:
                gc.collect()
            if (i+1) % 200 == 0 or (i+1) == len(pairs):
                print(f"    - Processed {i+1}/{len(pairs)} files...")
        gc.collect()
        print(f"    [+] Created {len(img_patches)} balanced patches for {name} split.")
        return img_patches, mask_patches, cls_patches

    train_imgs, train_masks, train_cls = process_pairs(train_pairs, "Train")
    val_imgs, val_masks, val_cls = process_pairs(val_pairs, "Val")

    train_dataset = SARDataset(train_imgs, train_masks, train_cls, augment=True)
    val_dataset = SARDataset(val_imgs, val_masks, val_cls, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_cls, val_cls

# -------------------------------------------------------------
# STEP 4: MODEL DEFINITIONS & LOSSES & METRICS
# -------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """Plain segmentation-only U-Net -- used to load/evaluate v1/v2/v3
    checkpoints (single logits tensor return, no aux head)."""
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
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        x = self.up1(x4); x = torch.cat([x, x3], dim=1); x = self.conv_up1(x)
        x = self.up2(x); x = torch.cat([x, x2], dim=1); x = self.conv_up2(x)
        x = self.up3(x); x = torch.cat([x, x1], dim=1); x = self.conv_up3(x)
        return self.outc(x)

class JointUNet(nn.Module):
    """UNet + aux classification head off the bottleneck. Used for v3-aux-0.3,
    v3-aux-1.0, and this run's new v3b checkpoint."""
    def __init__(self, in_channels=2, out_channels=1, num_classes=2):
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
        self.aux_pool = nn.AdaptiveAvgPool2d(1)
        self.aux_fc = nn.Sequential(nn.Linear(512, 64), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, num_classes))
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        cls_feat = self.aux_pool(x4).flatten(1)
        cls_logits = self.aux_fc(cls_feat)
        x = self.up1(x4); x = torch.cat([x, x3], dim=1); x = self.conv_up1(x)
        x = self.up2(x); x = torch.cat([x, x2], dim=1); x = self.conv_up2(x)
        x = self.up3(x); x = torch.cat([x, x1], dim=1); x = self.conv_up3(x)
        seg_logits = self.outc(x)
        return seg_logits, cls_logits

class FocalIoULoss(nn.Module):
    """Unchanged from v3 / Phase 3's aux kernel."""
    def __init__(self, alpha=0.25, gamma=2.0, focal_weight=1.0, iou_weight=1.0):
        super(FocalIoULoss, self).__init__()
        self.alpha = alpha; self.gamma = gamma
        self.focal_weight = focal_weight; self.iou_weight = iou_weight
    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = (alpha_t * (1 - p_t).pow(self.gamma) * bce).mean()
        probs_flat = probs.view(-1); targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum() - intersection
        iou = (intersection + 1e-5) / (union + 1e-5)
        iou_loss = 1.0 - iou
        return self.focal_weight * focal_loss + self.iou_weight * iou_loss

def calculate_metrics(outputs, targets, threshold=0.5):
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()
    preds_flat = preds.view(-1); targets_flat = targets.view(-1)
    intersection = (preds_flat * targets_flat).sum().item()
    union = preds_flat.sum().item() + targets_flat.sum().item() - intersection
    iou = (intersection + 1e-5) / (union + 1e-5)
    dice = (2. * intersection + 1e-5) / (preds_flat.sum().item() + targets_flat.sum().item() + 1e-5)
    return iou, dice

def classification_loss_and_acc(cls_logits, cls_labels, class_weight=None):
    """Cross-entropy (optionally class-weighted) + accuracy over only the
    non-ignored (oil/lookalike) subset of the batch. Also returns per-class
    correct/total counts so per-class accuracy can be tracked, not just
    overall (Phase 3's overall-accuracy number hid the majority-class
    collapse -- per-class breakdown would have caught it immediately)."""
    valid = cls_labels != -1
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        zero = cls_logits.sum() * 0.0
        return zero, 0, 0, {0: [0, 0], 1: [0, 0]}
    loss = F.cross_entropy(cls_logits[valid], cls_labels[valid], weight=class_weight)
    preds = cls_logits[valid].argmax(dim=1)
    labels_valid = cls_labels[valid]
    n_correct = int((preds == labels_valid).sum().item())
    per_class = {}
    for c in (0, 1):
        c_mask = labels_valid == c
        c_total = int(c_mask.sum().item())
        c_correct = int(((preds == labels_valid) & c_mask).sum().item())
        per_class[c] = [c_correct, c_total]
    return loss, n_valid, n_correct, per_class

# -------------------------------------------------------------
# STEP 5: TRAINING ORCHESTRATION
# -------------------------------------------------------------
def train_and_evaluate():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        if cap[0] >= 7:
            device = torch.device("cuda")
            print(f"\n[*] Compatible GPU found: {torch.cuda.get_device_name(0)} (CUDA Capability {cap[0]}.{cap[1]})")
        else:
            print(f"\n[!] Warning: GPU incompatible with pre-installed PyTorch. Falling back to CPU.")
    else:
        print("\n[*] No GPU available, using CPU.")

    print("\n=== PRE-FLIGHT REAL IMAGE NORMALIZATION DIAGNOSTIC ===")
    oil_img_dir = OIL_IMG_DIR
    test_paths = sorted(glob.glob(os.path.join(oil_img_dir, "*.tif")))[:3] if os.path.exists(oil_img_dir) else []
    if not test_paths:
        print("[!] Warning: No real images found at expected paths for diagnostic check.")
    else:
        diagnostic_failed = False
        for path in test_paths:
            raw_img = tifffile.imread(path)
            img_proc = raw_img.transpose(1, 2, 0) if (len(raw_img.shape) == 3 and raw_img.shape[0] == 2) else raw_img
            linear = 10.0 ** (img_proc.astype(np.float32) / 10.0)
            filtered = speckle_filter(linear, window_size=5)
            db = 10.0 * np.log10(np.clip(filtered, 1e-5, None))
            norm = (np.clip(db, -25.0, 0.0) - (-25.0)) / (0.0 - (-25.0))
            if norm.std() < 1e-3 or np.allclose(norm, 1.0) or np.allclose(norm, 0.0):
                diagnostic_failed = True
        if diagnostic_failed:
            print("[!] CRITICAL: Pre-flight diagnostic check FAILED. Aborting.")
            sys.exit(1)
        else:
            print("[OK] Pre-flight diagnostic check PASSED.")

    train_loader, val_loader, train_cls, val_cls = get_real_dataloaders(
        patch_size=PATCH_SIZE, stride=STRIDE, batch_size=BATCH_SIZE, train_ratio=0.8, seed=SEED)

    # -------------------------------------------------------------
    # PART A FIX: measure the REAL train-split patch class counts and
    # compute inverse-frequency class weights from them -- not assumed.
    # -------------------------------------------------------------
    n_lookalike_train = train_cls.count(0)
    n_oil_train = train_cls.count(1)
    n_no_oil_train = train_cls.count(-1)
    n_valid_train = n_lookalike_train + n_oil_train
    weight_lookalike = n_valid_train / (2.0 * n_lookalike_train) if n_lookalike_train > 0 else 1.0
    weight_oil = n_valid_train / (2.0 * n_oil_train) if n_oil_train > 0 else 1.0
    class_weight = torch.tensor([weight_lookalike, weight_oil], dtype=torch.float32).to(device)

    print(f"\n=== PART A FIX: MEASURED TRAIN-SPLIT PATCH CLASS COUNTS ===")
    print(f"  Oil-labeled patches:       {n_oil_train}")
    print(f"  Lookalike-labeled patches: {n_lookalike_train}")
    print(f"  No-oil patches (ignored):  {n_no_oil_train}")
    print(f"  Observed imbalance ratio (oil:lookalike): {n_oil_train / max(n_lookalike_train,1):.2f} : 1")
    print(f"  Computed class weights -> lookalike={weight_lookalike:.4f}, oil={weight_oil:.4f}")

    model = JointUNet(in_channels=2, out_channels=1, num_classes=2).to(device)
    seg_criterion = FocalIoULoss(alpha=0.25, gamma=2.0, focal_weight=1.0, iou_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print(f"\n=== STARTING TRAINING FRESH FROM SCRATCH ({MODEL_TAG}: Focal+IoU seg loss + aux_weight={AUX_WEIGHT} + class-weighted cls loss) ===")

    log_file_path = f"/kaggle/working/training_{MODEL_TAG}_log.csv"
    with open(log_file_path, "w") as lf:
        lf.write("epoch,train_seg_loss,train_cls_loss,train_total_loss,train_iou,train_cls_acc,train_oil_acc,train_lookalike_acc,"
                  "val_seg_loss,val_cls_loss,val_total_loss,val_iou,val_dice,val_cls_acc,val_oil_acc,val_lookalike_acc,active_prediction_pixels\n")

    best_val_iou = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_seg_loss = 0.0
        train_cls_loss_sum = 0.0
        train_ious = []
        train_cls_correct = 0
        train_cls_valid = 0
        train_per_class = {0: [0, 0], 1: [0, 0]}
        start_time = time.time()

        for batch_idx, (images, masks, cls_labels) in enumerate(train_loader):
            images, masks, cls_labels = images.to(device), masks.to(device), cls_labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                seg_logits, cls_logits = model(images)
                seg_loss = seg_criterion(seg_logits, masks)
                cls_loss, n_valid, n_correct, per_class = classification_loss_and_acc(cls_logits, cls_labels, class_weight)
                total_loss = seg_loss + AUX_WEIGHT * cls_loss
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_seg_loss += seg_loss.item()
            train_cls_loss_sum += cls_loss.item()
            train_cls_correct += n_correct
            train_cls_valid += n_valid
            for c in (0, 1):
                train_per_class[c][0] += per_class[c][0]
                train_per_class[c][1] += per_class[c][1]
            iou, _ = calculate_metrics(seg_logits, masks)
            train_ious.append(iou)

        epoch_train_seg_loss = train_seg_loss / len(train_loader)
        epoch_train_cls_loss = train_cls_loss_sum / len(train_loader)
        epoch_train_total_loss = epoch_train_seg_loss + AUX_WEIGHT * epoch_train_cls_loss
        epoch_train_iou = np.mean(train_ious)
        epoch_train_cls_acc = (train_cls_correct / train_cls_valid) if train_cls_valid > 0 else float('nan')
        epoch_train_oil_acc = (train_per_class[1][0] / train_per_class[1][1]) if train_per_class[1][1] > 0 else float('nan')
        epoch_train_lookalike_acc = (train_per_class[0][0] / train_per_class[0][1]) if train_per_class[0][1] > 0 else float('nan')

        model.eval()
        val_seg_loss = 0.0
        val_cls_loss_sum = 0.0
        total_intersection = 0.0
        total_union = 0.0
        total_preds_sum = 0.0
        total_targets_sum = 0.0
        total_active_pixels = 0
        val_cls_correct = 0
        val_cls_valid = 0
        val_per_class = {0: [0, 0], 1: [0, 0]}

        with torch.no_grad():
            for images, masks, cls_labels in val_loader:
                images, masks, cls_labels = images.to(device), masks.to(device), cls_labels.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    seg_logits, cls_logits = model(images)
                    seg_loss = seg_criterion(seg_logits, masks)
                    cls_loss, n_valid, n_correct, per_class = classification_loss_and_acc(cls_logits, cls_labels, class_weight)
                val_seg_loss += seg_loss.item()
                val_cls_loss_sum += cls_loss.item()
                val_cls_correct += n_correct
                val_cls_valid += n_valid
                for c in (0, 1):
                    val_per_class[c][0] += per_class[c][0]
                    val_per_class[c][1] += per_class[c][1]
                probs = torch.sigmoid(seg_logits)
                preds = (probs > 0.5).float()
                preds_flat = preds.view(-1); targets_flat = masks.view(-1)
                intersection = (preds_flat * targets_flat).sum().item()
                union = preds_flat.sum().item() + targets_flat.sum().item() - intersection
                total_intersection += intersection
                total_union += union
                total_preds_sum += preds_flat.sum().item()
                total_targets_sum += targets_flat.sum().item()
                total_active_pixels += preds.sum().item()

        epoch_val_seg_loss = val_seg_loss / len(val_loader)
        epoch_val_cls_loss = val_cls_loss_sum / len(val_loader)
        epoch_val_total_loss = epoch_val_seg_loss + AUX_WEIGHT * epoch_val_cls_loss
        epoch_val_iou = (total_intersection + 1e-5) / (total_union + 1e-5)
        epoch_val_dice = (2.0 * total_intersection + 1e-5) / (total_preds_sum + total_targets_sum + 1e-5)
        epoch_val_cls_acc = (val_cls_correct / val_cls_valid) if val_cls_valid > 0 else float('nan')
        epoch_val_oil_acc = (val_per_class[1][0] / val_per_class[1][1]) if val_per_class[1][1] > 0 else float('nan')
        epoch_val_lookalike_acc = (val_per_class[0][0] / val_per_class[0][1]) if val_per_class[0][1] > 0 else float('nan')
        elapsed = time.time() - start_time
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"  Train SegLoss: {epoch_train_seg_loss:.4f} | Train ClsLoss: {epoch_train_cls_loss:.4f} | Train IoU: {epoch_train_iou:.4f} | Train ClsAcc: {epoch_train_cls_acc:.4f} (oil={epoch_train_oil_acc:.4f} lookalike={epoch_train_lookalike_acc:.4f})")
        print(f"  Val SegLoss:   {epoch_val_seg_loss:.4f} | Val ClsLoss:   {epoch_val_cls_loss:.4f} | Val IoU:   {epoch_val_iou:.4f} | Val Dice: {epoch_val_dice:.4f} | Val ClsAcc: {epoch_val_cls_acc:.4f} (oil={epoch_val_oil_acc:.4f} lookalike={epoch_val_lookalike_acc:.4f})")
        print(f"  Active Pixels predicted: {total_active_pixels:.0f}")

        with open(log_file_path, "a") as lf:
            lf.write(f"{epoch},{epoch_train_seg_loss:.6f},{epoch_train_cls_loss:.6f},{epoch_train_total_loss:.6f},"
                      f"{epoch_train_iou:.6f},{epoch_train_cls_acc:.6f},{epoch_train_oil_acc:.6f},{epoch_train_lookalike_acc:.6f},"
                      f"{epoch_val_seg_loss:.6f},{epoch_val_cls_loss:.6f},{epoch_val_total_loss:.6f},"
                      f"{epoch_val_iou:.6f},{epoch_val_dice:.6f},{epoch_val_cls_acc:.6f},{epoch_val_oil_acc:.6f},{epoch_val_lookalike_acc:.6f},{total_active_pixels}\n")

        if epoch >= 3 and total_active_pixels == 0:
            print("[!!!] WARNING: Model predictions collapsed to all-zero background masks. Stopping early.")
            break

        if epoch_val_iou > best_val_iou:
            best_val_iou = epoch_val_iou
            torch.save(model.state_dict(), f"/kaggle/working/model_real_{MODEL_TAG}_best.pt")
            print(f"  [+] Saved new best model checkpoint: model_real_{MODEL_TAG}_best.pt")

        if epoch % 5 == 0:
            chk_path = f"/kaggle/working/model_real_{MODEL_TAG}_epoch_{epoch}.pt"
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'val_iou': epoch_val_iou}, chk_path)
            print(f"  [+] Periodic backup saved: {os.path.basename(chk_path)}")

    torch.save(model.state_dict(), f"/kaggle/working/model_real_{MODEL_TAG}_final.pt")
    print(f"\n[+] Training complete. Saved final model: model_real_{MODEL_TAG}_final.pt")
    return model, device

# -------------------------------------------------------------
# STEP 6: FULL IN-KERNEL 6-WAY HOLDOUT EVALUATION (Part B)
# -------------------------------------------------------------
def run_tiled_inference(model, norm, device, patch_size=256, threshold=0.5):
    """Plain UNet tiled inference -- for v1/v2/v3 checkpoints."""
    H0, W0, C = norm.shape
    pad_h = (-H0) % patch_size; pad_w = (-W0) % patch_size
    if pad_h or pad_w:
        norm = np.pad(norm, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    H, W, _ = norm.shape
    patches, coords = [], []
    for y in range(0, H, patch_size):
        for x in range(0, W, patch_size):
            patches.append(norm[y:y+patch_size, x:x+patch_size, :].transpose(2, 0, 1))
            coords.append((y, x))
    patches_tensor = torch.from_numpy(np.array(patches)).float().to(device)
    with torch.no_grad():
        logits = model(patches_tensor)
        probs_tiles = torch.sigmoid(logits).squeeze(1).cpu().numpy()
    probs_full = np.zeros((H, W), dtype=np.float32)
    for p_idx, (y, x) in enumerate(coords):
        probs_full[y:y+patch_size, x:x+patch_size] = probs_tiles[p_idx]
    probs_full = probs_full[:H0, :W0]
    preds_full = (probs_full > threshold).astype(np.uint8)
    return probs_full, preds_full

def run_tiled_inference_joint(model, norm, device, patch_size=256, threshold=0.5):
    """JointUNet tiled inference -- for v3-aux-0.3, v3-aux-1.0, and this
    run's new checkpoint. Scene-level classification = mean tile softmax."""
    H0, W0, C = norm.shape
    pad_h = (-H0) % patch_size; pad_w = (-W0) % patch_size
    if pad_h or pad_w:
        norm = np.pad(norm, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    H, W, _ = norm.shape
    patches, coords = [], []
    for y in range(0, H, patch_size):
        for x in range(0, W, patch_size):
            patches.append(norm[y:y+patch_size, x:x+patch_size, :].transpose(2, 0, 1))
            coords.append((y, x))
    patches_tensor = torch.from_numpy(np.array(patches)).float().to(device)
    with torch.no_grad():
        seg_logits, cls_logits = model(patches_tensor)
        probs_tiles = torch.sigmoid(seg_logits).squeeze(1).cpu().numpy()
        cls_probs_tiles = F.softmax(cls_logits, dim=1).cpu().numpy()
    probs_full = np.zeros((H, W), dtype=np.float32)
    for p_idx, (y, x) in enumerate(coords):
        probs_full[y:y+patch_size, x:x+patch_size] = probs_tiles[p_idx]
    probs_full = probs_full[:H0, :W0]
    preds_full = (probs_full > threshold).astype(np.uint8)
    scene_cls_probs = cls_probs_tiles.mean(axis=0)
    scene_cls_pred = int(np.argmax(scene_cls_probs))
    return probs_full, preds_full, scene_cls_probs, scene_cls_pred

def eval_preprocess(image_raw):
    """Matches src/compare_checkpoints_v3aux.py's preprocess() exactly --
    dB-scale real Sentinel-1 input assumed (same as every prior eval)."""
    linear = 10.0 ** (image_raw.astype(np.float32) / 10.0)
    filt = speckle_filter(linear, window_size=5)
    db = 10.0 * np.log10(np.clip(filt, 1e-5, None))
    return normalize_image(db)

def eval_scene_category(scene_id):
    for cat in ("oil", "no_oil", "lookalike"):
        if scene_id.startswith(cat):
            return cat
    return None

def eval_classify_outcome(category, tp, fp, fn, iou):
    if category != "oil":
        return "correct" if fp == 0 else "false_positive"
    if tp == 0:
        return "false_negative"
    return "correct" if iou >= 0.5 else "partial"

CLS_GROUND_TRUTH = {"oil": 1, "lookalike": 0}

def run_full_evaluation(new_model, new_device):
    print("\n\n=== STEP 6: FULL IN-KERNEL 6-WAY HOLDOUT EVALUATION ===")
    device = new_device

    print("[*] Locating checkpoint dataset...")
    ckpt_names = {
        "v1": "model_real_best.pt", "v2": "model_real_v2_best.pt", "v3": "model_real_v3_best.pt",
        "v3-aux-0.3": "model_real_v3_aux03_best.pt", "v3-aux-1.0": "model_real_v3_aux10_best.pt",
    }
    ckpt_paths = {}
    for v, fname in ckpt_names.items():
        path = find_file("/kaggle/input", fname)
        if path is None:
            raise ValueError(f"[!] HARD FAIL: could not locate checkpoint {fname} under /kaggle/input")
        ckpt_paths[v] = path
        print(f"  [+] {v} -> {path}")

    print("[*] Locating holdout dataset...")
    images_dir, masks_dir = find_holdout_dirs("/kaggle/input")
    if images_dir is None or masks_dir is None:
        raise ValueError(f"[!] HARD FAIL: could not locate holdout images/masks dirs (images={images_dir}, masks={masks_dir})")
    print(f"  [+] images_dir -> {images_dir}")
    print(f"  [+] masks_dir  -> {masks_dir}")

    plain_models = {}
    for v in ("v1", "v2", "v3"):
        m = UNet(in_channels=2, out_channels=1)
        sd = torch.load(ckpt_paths[v], map_location=device)
        m.load_state_dict(sd)
        m.to(device); m.eval()
        plain_models[v] = m

    joint_models = {}
    for v in ("v3-aux-0.3", "v3-aux-1.0"):
        m = JointUNet(in_channels=2, out_channels=1, num_classes=2)
        sd = torch.load(ckpt_paths[v], map_location=device)
        m.load_state_dict(sd)
        m.to(device); m.eval()
        joint_models[v] = m
    new_model.eval()
    joint_models[MODEL_TAG] = new_model

    img_paths = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
    print(f"[*] Found {len(img_paths)} holdout scenes")

    per_scene_rows = []
    t_start = time.time()

    all_versions = list(plain_models.keys()) + list(joint_models.keys())

    for idx, img_path in enumerate(img_paths):
        scene_id = os.path.splitext(os.path.basename(img_path))[0]
        cat = eval_scene_category(scene_id)
        if cat is None:
            print(f"  [!] skipping {scene_id}: unknown category")
            continue
        mask_path = os.path.join(masks_dir, os.path.basename(img_path))
        image_raw = tifffile.imread(img_path)
        mask_raw = tifffile.imread(mask_path)
        target = (mask_raw > 0).astype(np.uint8)
        norm = eval_preprocess(image_raw)

        row = {"scene_id": scene_id, "category": cat, "gt_positive_px": int(target.sum()),
               "cls_ground_truth": CLS_GROUND_TRUTH.get(cat, "")}

        for v, model in plain_models.items():
            t0 = time.time()
            _, pred = run_tiled_inference(model, norm, device)
            elapsed = time.time() - t0
            tp = int(np.sum((pred == 1) & (target == 1)))
            fp = int(np.sum((pred == 1) & (target == 0)))
            fn = int(np.sum((pred == 0) & (target == 1)))
            if target.sum() == 0:
                iou = 1.0 if fp == 0 else 0.0
                dice = 1.0 if fp == 0 else 0.0
            else:
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                dice = 2.0 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if target.sum() == 0 and fp == 0 else 0.0)
            recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if target.sum() == 0 else 0.0)
            outcome = eval_classify_outcome(cat, tp, fp, fn, iou)
            row[f"{v}_iou"] = round(iou, 4); row[f"{v}_dice"] = round(dice, 4)
            row[f"{v}_precision"] = round(precision, 4); row[f"{v}_recall"] = round(recall, 4)
            row[f"{v}_fp_px"] = fp; row[f"{v}_outcome"] = outcome
            print(f"  [{idx+1}/{len(img_paths)}] {scene_id:16s} [{v}] iou={iou:.3f} outcome={outcome:15s} ({elapsed:.1f}s)")

        for v, model in joint_models.items():
            t0 = time.time()
            _, pred, cls_probs, cls_pred = run_tiled_inference_joint(model, norm, device)
            elapsed = time.time() - t0
            tp = int(np.sum((pred == 1) & (target == 1)))
            fp = int(np.sum((pred == 1) & (target == 0)))
            fn = int(np.sum((pred == 0) & (target == 1)))
            if target.sum() == 0:
                iou = 1.0 if fp == 0 else 0.0
                dice = 1.0 if fp == 0 else 0.0
            else:
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                dice = 2.0 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if target.sum() == 0 and fp == 0 else 0.0)
            recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if target.sum() == 0 else 0.0)
            outcome = eval_classify_outcome(cat, tp, fp, fn, iou)
            safe_v = v.replace(".", "_").replace("-", "_")
            row[f"{safe_v}_iou"] = round(iou, 4); row[f"{safe_v}_dice"] = round(dice, 4)
            row[f"{safe_v}_precision"] = round(precision, 4); row[f"{safe_v}_recall"] = round(recall, 4)
            row[f"{safe_v}_fp_px"] = fp; row[f"{safe_v}_outcome"] = outcome
            row[f"{safe_v}_cls_pred"] = cls_pred
            if cat in CLS_GROUND_TRUTH:
                row[f"{safe_v}_cls_correct"] = int(cls_pred == CLS_GROUND_TRUTH[cat])
            else:
                row[f"{safe_v}_cls_correct"] = ""
            print(f"  [{idx+1}/{len(img_paths)}] {scene_id:16s} [{v}] iou={iou:.3f} outcome={outcome:15s} cls_pred={cls_pred} ({elapsed:.1f}s)")

        per_scene_rows.append(row)

    print(f"\n[+] Evaluation done in {time.time()-t_start:.1f}s total")

    import json
    with open("/kaggle/working/holdout_per_scene_results_v3b.json", "w") as f:
        json.dump(per_scene_rows, f, indent=2)

    print("\n=== SEGMENTATION SUMMARY (all 6 versions) ===")
    for cat in ("oil", "no_oil", "lookalike"):
        cat_rows = [r for r in per_scene_rows if r["category"] == cat]
        print(f"\n-- {cat} (n={len(cat_rows)}) --")
        for v in all_versions:
            safe_v = v.replace(".", "_").replace("-", "_")
            iou_avg = np.mean([r[f"{safe_v}_iou"] for r in cat_rows])
            dice_avg = np.mean([r[f"{safe_v}_dice"] for r in cat_rows])
            prec_avg = np.mean([r[f"{safe_v}_precision"] for r in cat_rows])
            rec_avg = np.mean([r[f"{safe_v}_recall"] for r in cat_rows])
            outcomes = [r[f"{safe_v}_outcome"] for r in cat_rows]
            counts = {o: outcomes.count(o) for o in set(outcomes)}
            print(f"  [{v:12s}] IoU={iou_avg:.4f} Dice={dice_avg:.4f} Prec={prec_avg:.4f} Rec={rec_avg:.4f} buckets={counts}")

    print("\n=== AUX HEAD CLASSIFICATION ACCURACY (oil+lookalike holdout, n=20) ===")
    cls_rows = [r for r in per_scene_rows if r["category"] in CLS_GROUND_TRUTH]
    for v in ("v3-aux-0.3", "v3-aux-1.0", MODEL_TAG):
        safe_v = v.replace(".", "_").replace("-", "_")
        n = len(cls_rows)
        n_correct = sum(int(r[f"{safe_v}_cls_correct"]) for r in cls_rows)
        overall_acc = n_correct / n if n > 0 else float("nan")
        oil_rows = [r for r in cls_rows if r["category"] == "oil"]
        lookalike_rows = [r for r in cls_rows if r["category"] == "lookalike"]
        oil_correct = sum(int(r[f"{safe_v}_cls_correct"]) for r in oil_rows)
        lookalike_correct = sum(int(r[f"{safe_v}_cls_correct"]) for r in lookalike_rows)
        oil_acc = oil_correct / len(oil_rows) if oil_rows else float("nan")
        lookalike_acc = lookalike_correct / len(lookalike_rows) if lookalike_rows else float("nan")
        print(f"  [{v:12s}] overall={n_correct}/{n}={overall_acc:.4f} | oil={oil_correct}/{len(oil_rows)}={oil_acc:.4f} | lookalike={lookalike_correct}/{len(lookalike_rows)}={lookalike_acc:.4f}")

    print("\n[+] Full evaluation complete.")

if __name__ == "__main__":
    check_part2_discrepancy()
    verify_real_images()
    trained_model, trained_device = train_and_evaluate()
    run_full_evaluation(trained_model, trained_device)
