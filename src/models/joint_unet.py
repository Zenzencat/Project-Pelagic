"""
Project Pelagic -- Joint U-Net with an auxiliary binary classification head
SWU Prasarnmit AI Engineering Final Project (Phase 3)

Subclasses UNet exactly (same encoder/decoder, same weight layout for the
segmentation path) and adds a small auxiliary head off the bottleneck
feature (x4, 512 channels, deepest encoder activation) to predict oil vs.
lookalike at the scene/patch level. This is intentionally binary, not
3-class: per Phase 3 Step 0, the only scene-level labels this project
already has and trusts are oil vs. lookalike (from
kaggle_kernel_lookalike/train_lookalike_classifier.py's build_scene_pairs());
no-oil scenes were never labeled for that pipeline, and Zen chose not to
fabricate a third class rather than extend the labeling convention.

forward() returns (seg_logits, cls_logits) -- a strict superset of UNet's
forward() (single logits tensor), so this is a separate class rather than a
change to UNet itself: v1/v2/v3 checkpoints and every caller of the plain
UNet (src/api/main.py, src/inference.py's run_tiled_inference, all prior
evaluation scripts) are completely unaffected.

WARNING: Keep synced with the joint training kernels under
kaggle_kernel_v3_aux/ (Kaggle kernels must be single-file, so this class is
duplicated there per this project's established train_kaggle.py convention).
"""

import torch
import torch.nn as nn

from src.models.unet import UNet


class JointUNet(UNet):
    """
    UNet + a lightweight auxiliary classification head (GAP + 2-layer FC)
    off the bottleneck. num_classes=2 (lookalike=0, oil=1); no-oil patches
    are excluded from the classification loss via ignore_index=-1 at the
    caller, not represented in this head's output space at all.
    """
    def __init__(self, in_channels=2, out_channels=1, num_classes=2):
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.aux_pool = nn.AdaptiveAvgPool2d(1)
        self.aux_fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        cls_feat = self.aux_pool(x4).flatten(1)
        cls_logits = self.aux_fc(cls_feat)

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)

        seg_logits = self.outc(x)
        return seg_logits, cls_logits


if __name__ == "__main__":
    model = JointUNet(in_channels=2, out_channels=1, num_classes=2)
    test_tensor = torch.randn(2, 2, 256, 256)
    seg_logits, cls_logits = model(test_tensor)
    print(f"[*] Input shape:       {test_tensor.shape}")
    print(f"[*] Seg logits shape:  {seg_logits.shape} (Expected: [2, 1, 256, 256])")
    print(f"[*] Cls logits shape:  {cls_logits.shape} (Expected: [2, 2])")
