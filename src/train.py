"""
Project Pelagic — Baseline U-Net Model Training Orchestrator
SWU Prasarnmit AI Engineering Final Project

This script handles the training and validation loops for our baseline
U-Net semantic segmentation model. It records stats to a CSV log file
and saves the best model checkpoint.
"""

import os
import csv
import torch
import torch.optim as optim
from src.data.dataset import get_dataloaders
from src.models.unet import UNet
from src.models.loss import BCEDiceLoss
from src.models.metrics import get_batch_metrics

def train_model(data_dir, output_dir, epochs=10, batch_size=8, lr=1e-3, patch_size=256, stride=128, device=None):
    """
    Orchestrates the model training process.
    """
    # 1. Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training baseline model on device: {device}")
    
    # Create output directories
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    csv_log_path = os.path.join(logs_dir, "training_history.csv")
    
    # 2. Get DatLoaders
    print(f"[*] Loading datasets from {data_dir}...")
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=data_dir,
        patch_size=patch_size,
        stride=stride,
        batch_size=batch_size
    )
    
    # 3. Instantiate model, optimizer, and loss function
    model = UNet(in_channels=2, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    
    # Initialize CSV history logger
    history_fields = ["epoch", "train_loss", "train_iou", "train_dice", "train_precision", "train_recall",
                      "val_loss", "val_iou", "val_dice", "val_precision", "val_recall"]
    
    with open(csv_log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history_fields)
        
    best_val_iou = -1.0
    
    # 4. Loop epochs
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
            # Batch metrics
            batch_m = get_batch_metrics(outputs, masks)
            for k in train_metrics:
                train_metrics[k] += batch_m[k] * images.size(0)
                
        # Normalize training metrics by total dataset size
        total_train_samples = len(train_loader.dataset)
        epoch_train_loss = train_loss / total_train_samples
        for k in train_metrics:
            train_metrics[k] /= total_train_samples
            
        print(f"[Train] Loss: {epoch_train_loss:.4f} | IoU: {train_metrics['iou']:.4f} | Dice: {train_metrics['dice']:.4f}")
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                
                val_loss += loss.item() * images.size(0)
                
                batch_m = get_batch_metrics(outputs, masks)
                for k in val_metrics:
                    val_metrics[k] += batch_m[k] * images.size(0)
                    
        # Normalize validation metrics by dataset size
        total_val_samples = len(val_loader.dataset)
        epoch_val_loss = val_loss / total_val_samples
        for k in val_metrics:
            val_metrics[k] /= total_val_samples
            
        print(f"[Val]   Loss: {epoch_val_loss:.4f} | IoU: {val_metrics['iou']:.4f} | Dice: {val_metrics['dice']:.4f}")
        
        # 5. Log metrics to CSV
        with open(csv_log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{epoch_train_loss:.6f}", f"{train_metrics['iou']:.6f}", f"{train_metrics['dice']:.6f}", f"{train_metrics['precision']:.6f}", f"{train_metrics['recall']:.6f}",
                f"{epoch_val_loss:.6f}", f"{val_metrics['iou']:.6f}", f"{val_metrics['dice']:.6f}", f"{val_metrics['precision']:.6f}", f"{val_metrics['recall']:.6f}"
            ])
            
        # 6. Save checkpoints
        latest_checkpoint = os.path.join(checkpoints_dir, "latest.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
        }, latest_checkpoint)
        
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            best_checkpoint = os.path.join(checkpoints_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_val_iou,
                'val_loss': epoch_val_loss,
            }, best_checkpoint)
            print(f"[+] Saved new best model checkpoint (Val IoU improved to {best_val_iou:.4f})")
            
    print(f"\n[+] Training finished. Checkpoints saved to: {checkpoints_dir}")
    print(f"[+] CSV log history written to: {csv_log_path}")

if __name__ == "__main__":
    # Test call stub
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(base_dir, "data", "synthetic")
    train_model(data_path, base_dir, epochs=2, batch_size=4)
