"""ChannelNet = SRCNN (super-resolution) followed by DnCNN (image restoration / denoising).

Input  : low-resolution channel image = LS pilot estimates interpolated onto the 72 x 14 grid
         (real and imaginary parts processed as separate single-channel images)
Output : estimate of the full ground-truth channel image

SRCNN (Dong et al., TPAMI 2016) - 3 layers, "9-1-5" setting
    conv 9x9, 64 filters + ReLU    patch extraction / representation
    conv 1x1, 32 filters + ReLU    non-linear mapping
    conv 5x5,  1 filter            reconstruction
DnCNN (Zhang et al., TIP 2017) - depth D (=20), 64 feature maps, residual learning
    conv 3x3 + ReLU
    (D-2) x [conv 3x3 + BatchNorm + ReLU]
    conv 3x3                       -> predicts the residual (noise) image R(y)
    output  x_hat = y - R(y)
'same' zero padding keeps the 72 x 14 size everywhere.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self, n1=64, n2=32, f1=9, f2=1, f3=5):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, n1, f1, padding=f1 // 2), nn.ReLU(inplace=True),
            nn.Conv2d(n1, n2, f2, padding=f2 // 2), nn.ReLU(inplace=True),
            nn.Conv2d(n2, 1, f3, padding=f3 // 2),
        )

    def forward(self, x):
        return self.body(x)


class DnCNN(nn.Module):
    def __init__(self, depth=20, features=64):
        super().__init__()
        layers = [nn.Conv2d(1, features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(features, features, 3, padding=1, bias=False),
                       nn.BatchNorm2d(features), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(features, 1, 3, padding=1)]
        self.body = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.body[-1].weight)       # start as the identity mapping (residual = 0)

    def forward(self, y):
        return y - self.body(y)                    # residual learning: x_hat = y - R(y)


class ChannelNet(nn.Module):
    """Cascade used at inference time: H_hat = DnCNN(SRCNN(H_low))."""

    def __init__(self, sr: SRCNN, ir: DnCNN):
        super().__init__()
        self.sr = sr
        self.ir = ir

    def forward(self, x, return_sr: bool = False):
        s = self.sr(x)
        out = self.ir(s)
        return (out, s) if return_sr else out


def build_srcnn(cfg: dict) -> SRCNN:
    return SRCNN(**cfg["model"]["srcnn"])


def build_dncnn(cfg: dict) -> DnCNN:
    return DnCNN(**cfg["model"]["dncnn"])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
