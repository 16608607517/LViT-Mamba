# -*- coding: utf-8 -*-
"""
LVMamba: Language-guided Vision Mamba Network
Replaces VisionTransformer with VisionMamba in LViT architecture.

Backbone options:
- backbone='transformer' → original LViT
- backbone='mamba' → LVMamba (this model)

Text options:
- use_text=True → text-guided segmentation (default)
- use_text=False → pure vision backbone (for ablation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Mamba import VisionMamba, Reconstruct
from .Vit import VisionTransformer, Reconstruct as VitReconstruct
from .pixlevel import PixLevelModule


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU() if activation == 'ReLU' else nn.GELU()

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = [ConvBatchNorm(in_channels, out_channels, activation)]
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        return self.nConvs(self.maxpool(x))


class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pixModule = PixLevelModule(in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.pixModule(skip_x)
        x = torch.cat([skip_x_att, up], dim=1)
        return self.nConvs(x)


class LVMamba(nn.Module):
    """
    Language-guided Vision Mamba Network.

    Args:
        config: CTranS config dict
        n_channels: input image channels (default 3)
        n_classes: output classes (default 1 for binary)
        img_size: input image size (default 224)
        backbone: 'mamba' or 'transformer'
        use_text: whether to use text guidance
    """
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224,
                 backbone='mamba', use_text=True, vis=False,
                 mamba_depth=1, scan_mode='bidirectional', text_gate=True,
                 text_cond=False, cond_target='dtBC'):
        super().__init__()
        self.backbone = backbone
        self.use_text = use_text
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel  # 64

        # Select backbone class
        if backbone == 'mamba':
            VBlock = VisionMamba
            extra_kwargs = dict(depth=mamba_depth, scan_mode=scan_mode, text_gate=text_gate,
                                text_cond=text_cond, cond_target=cond_target)
        else:
            VBlock = VisionTransformer
            extra_kwargs = {}

        # ===== Encoder =====
        self.inc = ConvBatchNorm(n_channels, in_channels)

        # 各尺度的特征图分辨率（支持任意输入尺寸）
        s = img_size
        s1, s2, s3, s4 = s, s // 2, s // 4, s // 8

        # Vision backbone blocks (down path)
        self.downVit = VBlock(config, vis, img_size=s1, channel_num=64, patch_size=16, embed_dim=64, **extra_kwargs)
        self.downVit1 = VBlock(config, vis, img_size=s2, channel_num=128, patch_size=8, embed_dim=128, **extra_kwargs)
        self.downVit2 = VBlock(config, vis, img_size=s3, channel_num=256, patch_size=4, embed_dim=256, **extra_kwargs)
        self.downVit3 = VBlock(config, vis, img_size=s4, channel_num=512, patch_size=2, embed_dim=512, **extra_kwargs)

        # Vision backbone blocks (up path)
        self.upVit = VBlock(config, vis, img_size=s1, channel_num=64, patch_size=16, embed_dim=64, **extra_kwargs)
        self.upVit1 = VBlock(config, vis, img_size=s2, channel_num=128, patch_size=8, embed_dim=128, **extra_kwargs)
        self.upVit2 = VBlock(config, vis, img_size=s3, channel_num=256, patch_size=4, embed_dim=256, **extra_kwargs)
        self.upVit3 = VBlock(config, vis, img_size=s4, channel_num=512, patch_size=2, embed_dim=512, **extra_kwargs)

        # ===== CNN UNet path =====
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)

        self.up4 = UpblockAttention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpblockAttention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpblockAttention(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpblockAttention(in_channels * 2, in_channels, nb_Conv=2)

        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid() if n_classes == 1 else None

        # ===== Reconstruct modules =====
        self.reconstruct1 = VitReconstruct(in_channels=64, out_channels=64, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = VitReconstruct(in_channels=128, out_channels=128, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct3 = VitReconstruct(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(4, 4))
        self.reconstruct4 = VitReconstruct(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(2, 2))

        # ===== Text projection modules =====
        if use_text:
            self.text_module4 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
            self.text_module3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
            self.text_module2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
            self.text_module1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

    def forward(self, x, text=None, return_feat=False):
        x = x.float()  # (B, 3, 224, 224)

        # ===== Text projection =====
        if self.use_text and text is not None:
            text4 = self.text_module4(text.transpose(1, 2)).transpose(1, 2)
            text3 = self.text_module3(text4.transpose(1, 2)).transpose(1, 2)
            text2 = self.text_module2(text3.transpose(1, 2)).transpose(1, 2)
            text1 = self.text_module1(text2.transpose(1, 2)).transpose(1, 2)
        else:
            # No text: use zero tensors (model degrades to pure vision)
            B, _, H, W = x.shape
            device = x.device
            text1 = torch.zeros(B, 10, 64, device=device)
            text2 = torch.zeros(B, 10, 128, device=device)
            text3 = torch.zeros(B, 10, 256, device=device)
            text4 = torch.zeros(B, 10, 512, device=device)

        # ===== Encoder =====
        x1 = self.inc(x)                                          # (B, 64, 224, 224)
        y1 = self.downVit(x1, x1, text1)                          # (B, 196, 128)
        x2 = self.down1(x1)                                       # (B, 128, 112, 112)
        y2 = self.downVit1(x2, y1, text2)                         # (B, 196, 256)
        x3 = self.down2(x2)                                       # (B, 256, 56, 56)
        y3 = self.downVit2(x3, y2, text3)                         # (B, 196, 512)
        x4 = self.down3(x3)                                       # (B, 512, 28, 28)
        y4 = self.downVit3(x4, y3, text4)                         # (B, 196, 512)
        x5 = self.down4(x4)                                       # (B, 512, 14, 14)

        # ===== Decoder (with skip connections) =====
        y4 = self.upVit3(y4, y4, text4, True)                     # (B, 196, 512)
        y3 = self.upVit2(y3, y4, text3, True)                     # (B, 196, 256)
        y2 = self.upVit1(y2, y3, text2, True)                     # (B, 196, 128)
        y1 = self.upVit(y1, y2, text1, True)                      # (B, 196, 64)

        x1 = self.reconstruct1(y1) + x1
        x2 = self.reconstruct2(y2) + x2
        x3 = self.reconstruct3(y3) + x3
        x4 = self.reconstruct4(y4) + x4

        # ===== CNN Decoder =====
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)

        if return_feat:
            # x5: (B, 512, 14, 14) bottleneck 特征，用于 LV Loss 对比对齐
            return logits, x5

        return logits
