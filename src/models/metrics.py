"""
Project Pelagic — Evaluation Metrics
SWU Prasarnmit AI Engineering Final Project

This module provides utility functions to compute evaluation metrics over 
model prediction tensors and ground-truth targets.
"""

import torch

def get_batch_metrics(outputs, targets, threshold=0.5, eps=1e-7):
    """
    Computes binary segmentation metrics (IoU, Dice, Precision, Recall)
    over a batch of logits and ground-truth binary masks.
    
    Args:
        outputs (torch.Tensor): Raw model logits of shape [B, 1, H, W].
        targets (torch.Tensor): Binary ground-truth target masks of shape [B, 1, H, W].
        threshold (float): Classification boundary threshold.
        eps (float): Epsilon smoothing factor to prevent division by zero.
        
    Returns:
        dict: Containing scalar values for IoU, Dice, Precision, and Recall.
    """
    with torch.no_grad():
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(outputs)
        preds = (probs > threshold).float()
        
        # Ensure targets are binary floats
        targets_binary = (targets > 0.5).float()
        
        # Flatten tensors
        preds_flat = preds.view(-1)
        targets_flat = targets_binary.view(-1)
        
        # Compute True Positives, False Positives, False Negatives
        tp = (preds_flat * targets_flat).sum()
        fp = (preds_flat * (1.0 - targets_flat)).sum()
        fn = ((1.0 - preds_flat) * targets_flat).sum()
        
        # 1. Intersection over Union (IoU)
        iou = (tp + eps) / (tp + fp + fn + eps)
        
        # 2. Dice Coefficient / F1-Score
        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        
        # 3. Precision
        precision = (tp + eps) / (tp + fp + eps)
        
        # 4. Recall
        recall = (tp + eps) / (tp + fn + eps)
        
        return {
            "iou": iou.item(),
            "dice": dice.item(),
            "precision": precision.item(),
            "recall": recall.item()
        }
