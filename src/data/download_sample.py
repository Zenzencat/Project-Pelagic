"""
Project Pelagic — Sentinel-1 SAR Oil Spill Dataset Downloader
SWU Prasarnmit AI Engineering Final Project

This script handles programmatic downloads of the 3-part Sentinel-1 SAR 
Oil Spill Image Dataset by Trujillo-Acatitla et al. from Zenodo.

Record mappings:
- Part I (Oil Spills): record ID 8346860 (~40.7 GB images, 6.2 MB masks)
- Part II (No Oil & Lookalikes): record ID 8253899 (~45.9 GB images, 843 KB masks)
- Part III (Held-Out Test Set): record ID 13761290 (~9.86 GB images & masks)
"""

import os
import sys
import argparse
import requests

# Record IDs for the 3-part dataset
ZENODO_RECORDS = {
    1: {
        "id": 8346860,
        "description": "Part I (Training/Validation — Oil Spill images): 1,200 Sentinel-1 SAR images + masks",
        "masks": ["01_Train_Val_Oil_Spill_mask.7z"],
        "images": ["01_Train_Val_Oil_Spill_images.7z"]
    },
    2: {
        "id": 8253899,
        "description": "Part II (Training/Validation — No Oil + Lookalike images): 685 images + masks",
        "masks": ["01_Train_Val_No_Oil_mask.7z", "01_Train_Val_Lookalike_mask.7z"],
        "images": ["01_Train_Val_No_Oil_Images.7z", "01_Train_Val_Lookalike_images.7z"]
    },
    3: {
        "id": 13761290,
        "description": "Part III (Test set — 150 Oil, 150 No Oil, 150 Lookalike): 450 images + masks",
        "masks": [], # Part III holds both images and ground truth in a single package
        "images": ["02_Test_images_and_ground_truth.7z"]
    }
}

def download_file(url, file_path):
    """Downloads a file with streaming and console progress logs."""
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            # Log every 10% or print progress
                            if downloaded % (8192 * 100) == 0 or downloaded == total_size:
                                print(f"    - Download progress: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB)", end='\r')
            print("\n    [+] Download complete.")
            return True
    except Exception as e:
        print(f"\n    [!] Error downloading file: {e}", file=sys.stderr)
        return False

def download_part(part_num, output_dir, include_images=False):
    """
    Downloads specified dataset part from Zenodo.
    
    Args:
        part_num (int): 1, 2, or 3.
        output_dir (str): Location to save files.
        include_images (bool): If True, downloads massive image archives.
    """
    if part_num not in ZENODO_RECORDS:
        print(f"[!] Error: Invalid part number {part_num}. Must be 1, 2, or 3.")
        return False
        
    record = ZENODO_RECORDS[part_num]
    record_id = record["id"]
    
    print(f"\n[*] Initializing Download for: {record['description']}")
    print(f"    - Zenodo Record ID: {record_id}")
    
    api_url = f"https://zenodo.org/api/records/{record_id}"
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        metadata = response.json()
    except Exception as e:
        print(f"[!] Failed to fetch metadata from Zenodo: {e}", file=sys.stderr)
        return False
        
    files = metadata.get("files", [])
    if not files:
        print("[!] Error: No files listed in Zenodo metadata.")
        return False
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter files based on user flags
    targets = []
    if part_num == 3:
        # Part III holds everything in one file
        targets = record["images"]
        if not include_images:
            print("[*] Notice: Part III contains images and masks in a single 9.8 GB archive.")
            print("    Please run with '--include-images' flag to download Part III.")
            return True
    else:
        # For Part I & II, we can download masks separately from images
        targets.extend(record["masks"])
        if include_images:
            targets.extend(record["images"])
        else:
            print("[*] Downloading MASKS ONLY (saves ~90 GB disk space & bandwidth).")
            print("    To download the full image archives, rerun the script with '--include-images'.")
            
    success = True
    for file_info in files:
        filename = file_info.get("key")
        if filename not in targets:
            continue
            
        download_url = file_info.get("links", {}).get("self")
        file_size_gb = file_info.get("size", 0) / (1024 * 1024 * 1024)
        
        print(f"\n[*] File: {filename} ({file_size_gb:.3f} GB)")
        file_path = os.path.join(output_dir, filename)
        
        if os.path.exists(file_path):
            print(f"    [+] File already exists at: {file_path}. Skipping.")
            continue
            
        if not download_file(download_url, file_path):
            success = False
            
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Trujillo-Acatitla et al. Sentinel-1 Oil Spill Dataset from Zenodo.")
    parser.add_argument("--part", type=int, choices=[1, 2, 3], default=1, help="Dataset part to download (1, 2, or 3).")
    parser.add_argument("--include-images", action="store_true", help="Download full 7z image archives (warning: files are 10-40 GB).")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save downloaded archives.")
    
    args = parser.parse_args()
    
    # Setup default output path: data/raw/
    if args.output_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        output_path = os.path.join(base_dir, "data", "raw")
    else:
        output_path = args.output_dir
        
    download_part(args.part, output_path, include_images=args.include_images)
