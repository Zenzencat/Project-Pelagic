import os
import sys
import glob
import argparse
import numpy as np
import tifffile
import cv2
import torch
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath("."))
from src.models.unet import UNet
from src.data.preprocess import run_full_preprocessing
from src.inference import run_tiled_inference

def compute_metrics(pred, target):
    """Computes pixel-level IoU, Dice, Precision, and Recall."""
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate U-Net on holdout test set.")
    parser.add_argument("--version", type=str, choices=["v1", "v2"], default="v2",
                        help="Model version to evaluate (v1 or v2).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluating on device: {device}")
    
    base_dir = os.path.abspath(".")
    
    # Select model path
    checkpoint_name = "model_real_best.pt" if args.version == "v1" else "model_real_v2_best.pt"
    model_path = os.path.join(base_dir, "checkpoints", checkpoint_name)
    
    # Load model
    print(f"[*] Loading model checkpoint from {model_path}...")
    model = UNet(in_channels=2, out_channels=1)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("[+] Model loaded successfully.")
    
    # Paths
    holdout_img_dir = os.path.join(base_dir, "data", "holdout", "images")
    holdout_mask_dir = os.path.join(base_dir, "data", "holdout", "masks")
    docs_dir = os.path.join(base_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    img_paths = sorted(glob.glob(os.path.join(holdout_img_dir, "*.tif")))
    print(f"[*] Found {len(img_paths)} scenes in holdout dataset.")
    
    categories = ["oil", "no_oil", "lookalike"]
    cat_metrics = {cat: {"iou": [], "dice": [], "precision": [], "recall": []} for cat in categories}
    overall_metrics = {"iou": [], "dice": [], "precision": [], "recall": []}
    
    viz_scenes = ["oil_00000", "oil_00001", "oil_00002", "lookalike_00000", "lookalike_00001",
                  "no_oil_00000", "no_oil_00001"]
    
    for img_path in img_paths:
        filename = os.path.basename(img_path)
        scene_id = os.path.splitext(filename)[0]
        
        cat_name = None
        for cat in categories:
            if scene_id.startswith(cat):
                cat_name = cat
                break
        if not cat_name:
            continue
            
        mask_path = os.path.join(holdout_mask_dir, filename)
        
        img_raw = tifffile.imread(img_path)
        mask_raw = tifffile.imread(mask_path)
        
        _, norm = run_full_preprocessing(img_raw)

        _, pred_full = run_tiled_inference(model, norm, device)

        iou, dice, precision, recall = compute_metrics(pred_full, mask_raw)
        
        cat_metrics[cat_name]["iou"].append(iou)
        cat_metrics[cat_name]["dice"].append(dice)
        cat_metrics[cat_name]["precision"].append(precision)
        cat_metrics[cat_name]["recall"].append(recall)
        
        overall_metrics["iou"].append(iou)
        overall_metrics["dice"].append(dice)
        overall_metrics["precision"].append(precision)
        overall_metrics["recall"].append(recall)
        
        # Save viz
        if scene_id in viz_scenes:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(norm[..., 0], cmap='gray')
            axes[0].set_title(f"SAR VV Channel - {scene_id}")
            axes[0].axis('off')
            
            axes[1].imshow(mask_raw, cmap='plasma')
            axes[1].set_title("Ground Truth Mask")
            axes[1].axis('off')
            
            axes[2].imshow(pred_full, cmap='plasma')
            axes[2].set_title(f"Prediction ({args.version.upper()})\n(IoU: {iou:.3f}, Dice: {dice:.3f})")
            axes[2].axis('off')
            
            plt.tight_layout()
            out_plot_path = os.path.join(docs_dir, f"holdout_{args.version}_viz_{scene_id}.png")
            # bbox_inches="tight": the third panel's title is two lines (the other
            # two are one line), and tight_layout() sizes the axes to fit within the
            # figure's existing bounds rather than growing the figure to fit a taller
            # title -- without this, the top of that title renders past the canvas
            # edge and savefig() silently clips it.
            plt.savefig(out_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            
    # Print Summary
    print(f"\n==============================================")
    print(f"=== SUMMARY METRICS FOR MODEL VERSION: {args.version.upper()} ===")
    print(f"==============================================")
    for cat in categories:
        iou_avg = np.mean(cat_metrics[cat]["iou"]) if cat_metrics[cat]["iou"] else 0.0
        dice_avg = np.mean(cat_metrics[cat]["dice"]) if cat_metrics[cat]["dice"] else 0.0
        prec_avg = np.mean(cat_metrics[cat]["precision"]) if cat_metrics[cat]["precision"] else 0.0
        rec_avg = np.mean(cat_metrics[cat]["recall"]) if cat_metrics[cat]["recall"] else 0.0
        print(f"Category: {cat:<10} | IoU: {iou_avg:.4f} | Dice: {dice_avg:.4f} (F1) | Precision: {prec_avg:.4f} | Recall: {rec_avg:.4f}")
        
    print(f"\nOVERALL METRICS:")
    print(f"  Overall IoU:       {np.mean(overall_metrics['iou']):.4f}")
    print(f"  Overall Dice (F1): {np.mean(overall_metrics['dice']):.4f}")
    print(f"  Overall Precision: {np.mean(overall_metrics['precision']):.4f}")
    print(f"  Overall Recall:    {np.mean(overall_metrics['recall']):.4f}")
    print(f"==============================================")

if __name__ == "__main__":
    main()
