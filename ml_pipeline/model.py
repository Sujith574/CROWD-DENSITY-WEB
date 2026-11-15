"""
CSRNet — Crowd counting network with VGG-16 frontend + dilated CNN backend.
Paper: "CSRNet: Dilated Convolutional Neural Networks for Understanding the
Highly Congested Scenes", Li et al., CVPR 2018.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CSRNet(nn.Module):
    """
    CSRNet architecture:
      Frontend : first 10 conv layers of VGG-16 (up to pool3), 1/8 spatial.
      Backend  : 6 dilated 3×3 conv layers (dilation=2) for density estimation.
      Head     : 1×1 conv → 1-channel continuous density map.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # ── Frontend: VGG-16 up to (but NOT including) pool3 ─────────────
        # children() of vgg16.features:
        #   0-1  : Conv-64, ReLU
        #   2-3  : Conv-64, ReLU
        #   4    : MaxPool  (→ 1/2)
        #   5-6  : Conv-128, ReLU
        #   7-8  : Conv-128, ReLU
        #   9    : MaxPool  (→ 1/4)
        #  10-11 : Conv-256, ReLU
        #  12-13 : Conv-256, ReLU
        #  14-15 : Conv-256, ReLU
        #  16    : MaxPool  (→ 1/8)
        #  17-18 : Conv-512, ReLU
        #  19-20 : Conv-512, ReLU
        #  21-22 : Conv-512, ReLU          ← [:23] captures indices 0-22
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg16(weights=weights)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        # ── Backend: dilated CNN ──────────────────────────────────────────
        self.backend = nn.Sequential(
            # (512 → 512)
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            # (512 → 512)
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            # (512 → 512)
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            # (512 → 256)
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            # (256 → 128)
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            # (128 → 64)
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )

        # ── Output head ──────────────────────────────────────────────────
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        # Initialise backend & head with Gaussian weights
        self._init_weights()

    # --------------------------------------------------------------------- #
    def _init_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)

    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)   # B × 512 × H/8 × W/8
        x = self.backend(x)    # B × 64  × H/8 × W/8
        x = self.output_layer(x)  # B × 1   × H/8 × W/8
        return x


# ── Quick sanity-check ────────────────────────────────────────────────────
if __name__ == "__main__":
    net = CSRNet(pretrained=False)
    dummy = torch.randn(1, 3, 256, 256)
    out = net(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}  (expected: [1, 1, 32, 32])")
    total = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total:,}")

