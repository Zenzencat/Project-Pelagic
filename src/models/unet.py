"""
Project Pelagic — U-Net Architecture
SWU Prasarnmit AI Engineering Final Project

This module defines a standard, lightweight U-Net segmentation model
in PyTorch, taking 2 input channels (VV, VH) and outputting 1 channel (logits).
"""

# WARNING: Keep synced with kaggle_kernel/train_kaggle.py (DoubleConv and UNet classes).
# If you modify this file, you must update the copies in kaggle_kernel/train_kaggle.py.

import torch
import torch.nn as nn

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

if __name__ == "__main__":
    # Test print shape validation
    model = UNet(in_channels=2, out_channels=1)
    test_tensor = torch.randn(2, 2, 256, 256)
    output = model(test_tensor)
    print(f"[*] Input shape:  {test_tensor.shape}")
    print(f"[*] Output shape: {output.shape} (Expected: [2, 1, 256, 256])")
