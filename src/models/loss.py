"""
Project Pelagic — Custom Loss Functions
SWU Prasarnmit AI Engineering Final Project

This module defines loss functions designed for training semantic segmentation 
models under severe class imbalance (rare oil slick pixels vs abundant ocean pixels).
"""

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Computes 1 - Dice Coefficient.
    Optimizes the spatial overlap directly, avoiding background pixel bias.
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs before sigmoid activation.
            targets (torch.Tensor): Ground-truth binary masks [0.0, 1.0].
        """
        # Apply sigmoid to obtain probability map
        probs = torch.sigmoid(logits)
        
        # Flatten batch and spatial dimensions
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        denominator = probs.sum() + targets.sum()
        
        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
        return 1. - dice

class BCEDiceLoss(nn.Module):
    r"""
    Joint Loss function combining Binary Cross-Entropy (BCE) and Dice Loss.
    $\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{BCE}} + \beta \mathcal{L}_{\text{Dice}}$
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.dice_loss_fn = DiceLoss()

    def forward(self, logits, targets):
        bce = self.bce_loss_fn(logits, targets)
        dice = self.dice_loss_fn(logits, targets)
        
        return self.bce_weight * bce + self.dice_weight * dice
