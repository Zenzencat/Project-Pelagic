"""
Project Pelagic — U-Net Architecture
SWU Prasarnmit AI Engineering Final Project

This module defines a standard, lightweight U-Net segmentation model
in PyTorch, taking 2 input channels (VV, VH) and outputting 1 channel (logits).
"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Convolution -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """
    Symmetric 4-level U-Net model with skip connections.
    Ideal for binary segmentation of oil slicks on small datasets.
    """
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 16)
        
        # Contracting Path (Encoder)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Expansive Path (Decoder)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)  # skip connection concatenates x4 (128 channels) -> 256 total
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)   # skip connection concatenates x3 (64 channels) -> 128 total
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)     # skip connection concatenates x2 (32 channels) -> 64 total
        
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(32, 16)     # skip connection concatenates x1 (16 channels) -> 32 total
        
        # Final output convolution
        self.outc = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connection concatenation
        u1 = self.up1(x5)
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.conv_up2(u2)
        
        u3 = self.up3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.conv_up3(u3)
        
        u4 = self.up4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.conv_up4(u4)
        
        logits = self.outc(u4)
        return logits

if __name__ == "__main__":
    # Test print shape validation
    model = UNet(in_channels=2, out_channels=1)
    test_tensor = torch.randn(2, 2, 256, 256)
    output = model(test_tensor)
    print(f"[*] Input shape:  {test_tensor.shape}")
    print(f"[*] Output shape: {output.shape} (Expected: [2, 1, 256, 256])")
