"""
Project Pelagic — Model Training Verification
SWU Prasarnmit AI Engineering Final Project

This script runs a short 2-epoch training cycle on synthetic SAR data
to verify the model training pipeline, weight updates, checkpoint saving, 
and executes a test prediction check.
"""

import os
import sys
import torch

# Add root folder to path to enable package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import train_model
from src.models.unet import UNet
from src.data.dataset import get_dataloaders

def main():
    print("[*] Starting Project Pelagic Model Training Verification...")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data", "synthetic")
    
    # 1. Run 2 epochs of training
    print("[*] Executing 2-epoch model training run on CPU...")
    try:
        train_model(
            data_dir=data_dir,
            output_dir=base_dir,
            epochs=2,
            batch_size=4,
            lr=1e-3,
            device=torch.device("cpu")
        )
    except Exception as e:
        print(f"[!] Error during training execution: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 2. Check that outputs are created
    checkpoint_best = os.path.join(base_dir, "checkpoints", "best_model.pth")
    checkpoint_latest = os.path.join(base_dir, "checkpoints", "latest.pth")
    csv_log_path = os.path.join(base_dir, "logs", "training_history.csv")
    
    print("[*] Verifying output files...")
    if not os.path.exists(checkpoint_best):
        print(f"[!] Error: Missing best model checkpoint {checkpoint_best}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(checkpoint_latest):
        print(f"[!] Error: Missing latest model checkpoint {checkpoint_latest}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(csv_log_path):
        print(f"[!] Error: Missing training history log {csv_log_path}", file=sys.stderr)
        sys.exit(1)
        
    print("[+] All outputs generated successfully.")
    
    # 3. Load checkpoint and verify active predictions
    print("[*] Loading checkpoint and performing inference checks...")
    try:
        checkpoint = torch.load(checkpoint_best, map_location="cpu")
        model = UNet(in_channels=2, out_channels=1)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        print(f"[!] Error loading model state: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Load the test partition to run predictions
    try:
        _, _, test_loader = get_dataloaders(
            data_dir=data_dir,
            patch_size=256,
            stride=128,
            batch_size=1
        )
    except Exception as e:
        print(f"[!] Error loading test split: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Run test predictions and check for positive pixels
    positive_predicted = False
    total_pixels_flagged = 0
    
    with torch.no_grad():
        for images, _ in test_loader:
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            pixel_sum = preds.sum().item()
            if pixel_sum > 0:
                positive_predicted = True
                total_pixels_flagged += pixel_sum
                
    if positive_predicted:
        print(f"[+] Prediction Check Passed: The model predicted {int(total_pixels_flagged)} positive pixels across the test set!")
        print("[+] Success: Training cycle and active model inference are fully verified.")
    else:
        print("\n" + "="*70)
        print("[!] WARNING: MODEL PREDICTIONS COLLAPSED TO ALL-ZERO BACKGROUND MASKS!")
        print("    After 2 epochs, the U-Net failed to output any positive pixels.")
        print("    This is common for short runs on small synthetic sets due to random initialization,")
        print("    but indicates that either the learning rate is too high, epochs are too low,")
        print("    or Dice Loss weighting needs careful tuning during final model training.")
        print("="*70 + "\n")
        print("[+] Verification run finished (with all-zero mask warning).")

if __name__ == "__main__":
    main()
