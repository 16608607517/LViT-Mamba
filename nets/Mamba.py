# -*- coding: utf-8 -*-
"""
MambaVision: Mamba-based Vision Backbone for Medical Image Segmentation.
Pure PyTorch implementation of Selective SSM (Mamba/S6 core).
Numerically stable version with strict value clamping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.modules.utils import _pair

# 自动检测官方 CUDA 加速版 mamba-ssm（Linux/WSL 可用，Windows 无）
# 环境变量 LVIT_NO_MAMBA_SSM=1 可强制走纯 PyTorch 版（用于加载纯 PyTorch 版训练的 checkpoint）
import os as _os
try:
    from mamba_ssm import Mamba as _CudaMamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _selective_scan_fn
    HAS_MAMBA_SSM = _os.environ.get('LVIT_NO_MAMBA_SSM') != '1'
except ImportError:
    HAS_MAMBA_SSM = False
    _selective_scan_fn = None


# ============================================================================
# 官方 kernel + 文本调制的 Mamba 块（创新点①的加速版）
# ============================================================================

class TextCondMamba(nn.Module):
    """参数结构与官方 mamba_ssm.Mamba 完全一致，但 dt/B/C 可被文本调制。

    用官方 selective_scan_fn（CUDA 加速，SSM 扫描的主导开销）而自行计算 dt/B/C，
    从而绕开官方融合 kernel 无法注入的限制，速度接近官方实现。

    文本调制为零初始化（起点与官方 Mamba 完全等价）。
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, cond_target='dtBC'):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.cond_target = cond_target

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 与官方一致的初始化
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
                       + math.log(0.001)).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # 文本调制头（零初始化 → 起点等价官方 Mamba）
        if 'dt' in cond_target:
            self.t_dt = nn.Linear(d_model, self.d_inner)
            nn.init.zeros_(self.t_dt.weight); nn.init.zeros_(self.t_dt.bias)
        if 'B' in cond_target:
            self.t_B = nn.Linear(d_model, d_state)
            nn.init.zeros_(self.t_B.weight); nn.init.zeros_(self.t_B.bias)
        if 'C' in cond_target:
            self.t_C = nn.Linear(d_model, d_state)
            nn.init.zeros_(self.t_C.weight); nn.init.zeros_(self.t_C.bias)

    def forward(self, u, t_vec=None):
        B0, L, _ = u.shape
        xz = self.in_proj(u)
        x, z = xz.chunk(2, dim=-1)                      # (B, L, d_inner)
        x = x.transpose(1, 2)                           # (B, d_inner, L)
        x = self.act(self.conv1d(x)[..., :L])           # 因果卷积

        x_dbl = self.x_proj(x.transpose(1, 2))          # (B, L, dt_rank + 2*d_state)
        dt, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight, self.dt_proj.bias)   # (B, L, d_inner)

        # ---- 文本调制（加性，注入在 softplus 之前）----
        if t_vec is not None:
            if 'dt' in self.cond_target:
                dt = dt + self.t_dt(t_vec).unsqueeze(1)
            if 'B' in self.cond_target:
                Bm = Bm + self.t_B(t_vec).unsqueeze(1)
            if 'C' in self.cond_target:
                Cm = Cm + self.t_C(t_vec).unsqueeze(1)

        dt = dt.transpose(1, 2).contiguous()            # (B, d_inner, L)
        Bm = Bm.transpose(1, 2).contiguous()            # (B, d_state, L)
        Cm = Cm.transpose(1, 2).contiguous()
        A = -torch.exp(self.A_log.float())              # (d_inner, d_state)

        y = _selective_scan_fn(x.contiguous(), dt, A, Bm, Cm, self.D.float(),
                               z=None, delta_bias=None, delta_softplus=True)
        y = y.transpose(1, 2) * self.act(z)             # 门控 + 输出投影
        return self.out_proj(y)


# ============================================================================
# Numerically Stable Selective SSM
# ============================================================================

class TextSummary(nn.Module):
    """注意力池化: (B, K, d) 的文本 token 序列 → (B, d) 的文本条件向量。

    以可学习 query 对 K 个 token 做加权(比 mean pooling 更能突出关键词),
    参数极少(K 通常为 10)。
    """

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale = dim ** -0.5

    def forward(self, text):
        w = torch.softmax((text * self.query).sum(-1) * self.scale, dim=1)  # (B, K)
        return (text * w.unsqueeze(-1)).sum(1)                              # (B, d)


class SelectiveScan(nn.Module):
    """Stable Selective Scan (Mamba core) with value clamping.

    text_cond=True 时, 文本向量 t 会直接调制选择性扫描参数 Δ/B/C
    (加性注入在激活函数之前, 数值安全; 文本投影零初始化 → 训练起点等价 baseline)。
    """

    def __init__(self, d_model, d_state=16, dt_rank=None, text_cond=False,
                 cond_target='dtBC'):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.text_cond = text_cond
        self.cond_target = cond_target      # 可消融: 'dt' / 'B' / 'C' / 'dtBC' 等组合
        dt_rank = dt_rank or max(d_model // 16, 4)

        self.dt_proj = nn.Sequential(
            nn.Linear(d_model, dt_rank),
            nn.SiLU(),
            nn.Linear(dt_rank, d_model),
        )
        # A_log: initialized negative for stable SSM (exp(positive * negative) <= 1)
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model, d_state) * 0.5 + 0.5))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # 文本 → 扫描参数 的调制头(零初始化: 起点与 baseline 完全一致)
        if text_cond:
            if 'dt' in cond_target:
                self.t_dt = nn.Linear(d_model, d_model)
                nn.init.zeros_(self.t_dt.weight); nn.init.zeros_(self.t_dt.bias)
            if 'B' in cond_target:
                self.t_B = nn.Linear(d_model, d_state)
                nn.init.zeros_(self.t_B.weight); nn.init.zeros_(self.t_B.bias)
            if 'C' in cond_target:
                self.t_C = nn.Linear(d_model, d_state)
                nn.init.zeros_(self.t_C.weight); nn.init.zeros_(self.t_C.bias)

    def forward(self, x, t_vec=None):
        B_s, L, D = x.shape

        u = F.silu(self.in_proj(x))

        # Stable delta（文本调制: 加性注入在 softplus 之前）
        dt_in = self.dt_proj(x)
        if self.text_cond and t_vec is not None and 'dt' in self.cond_target:
            dt_in = dt_in + self.t_dt(t_vec).unsqueeze(1)
        delta = F.softplus(dt_in)
        delta = delta.clamp(max=15.0)

        # Bounded projections（文本调制 B/C）
        b_in = self.B_proj(x)
        c_in = self.C_proj(x)
        if self.text_cond and t_vec is not None:
            if 'B' in self.cond_target:
                b_in = b_in + self.t_B(t_vec).unsqueeze(1)
            if 'C' in self.cond_target:
                c_in = c_in + self.t_C(t_vec).unsqueeze(1)
        B = torch.tanh(b_in)
        C = torch.tanh(c_in)

        # A_bar in (0, 1] since A_log < 0
        A_bar = torch.exp(delta.unsqueeze(-1) * self.A_log)
        A_bar = A_bar.clamp(min=1e-5, max=1.0)

        # B_bar
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)
        B_bar = B_bar.clamp(min=-5.0, max=5.0)

        u_exp = u.unsqueeze(-1).clamp(min=-5.0, max=5.0)

        y = self._scan(u_exp, A_bar, B_bar, C)
        y = self.out_proj(y)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y.clamp(min=-20.0, max=20.0)

    def _scan(self, u, A_bar, B_bar, C):
        """Parallel scan with numerical guards."""
        B_s, L, D, N = A_bar.shape
        C_exp = C.unsqueeze(2).expand(-1, -1, D, -1)

        log_A = torch.log(A_bar.clamp(min=1e-7))
        log_A_cum = torch.cumsum(log_A, dim=1)

        B_u = (B_bar * u).clamp(min=-10.0, max=10.0)

        A_cum = torch.exp(log_A_cum).clamp(min=1e-10, max=1e10)
        ratio = (B_u / A_cum).clamp(min=-1e6, max=1e6)

        h = torch.cumsum(ratio, dim=1) * A_cum
        h = h.clamp(min=-1e6, max=1e6)

        y = (h * C_exp).sum(dim=-1)
        return y.clamp(min=-20.0, max=20.0)


# ============================================================================
# Mamba Block
# ============================================================================

class MambaBlock(nn.Module):
    """
    Mamba block with configurable scan mode.
    - 'bidirectional': forward + backward scan (2 SSMs)
    - 'cross': VMamba-style 4-direction cross-scan (4 SSMs)
    """
    def __init__(self, dim, d_state=16, dt_rank=None, mlp_ratio=4., drop=0., drop_path=0.,
                 scan_mode='bidirectional', text_cond=False, cond_target='dtBC'):
        super().__init__()
        self.scan_mode = scan_mode
        # cond_target='film' → 用 FiLM 仿射调制"块输入特征"(对齐 TG-MUNet 的注入形式),
        # 不做 SSM 内部调制; 用于"注入形式 × 参数量"的对照实验
        self.text_film = (cond_target == 'film')
        self.text_cond = text_cond and not self.text_film
        if self.text_film:
            self.film = nn.Linear(dim, 2 * dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)                 # 零初始化 → 起点等价 baseline
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # 实现选择：
        #   text_cond + 官方可用 → TextCondMamba（官方 selective_scan_fn，速度快）
        #   text_cond + 无官方   → 纯 PyTorch SelectiveScan（带文本调制）
        #   无 text_cond + 官方  → 官方 _CudaMamba
        #   无 text_cond + 无官方 → 纯 PyTorch SelectiveScan
        # LVIT_TEXTCOND_PYTORCH=1 可强制走纯 PyTorch 版（用于加载 2026-09-13 前训练的 checkpoint）
        _force_pt = _os.environ.get('LVIT_TEXTCOND_PYTORCH') == '1'
        if text_cond and HAS_MAMBA_SSM and not _force_pt:
            self.ssm_impl = 'textcond_cuda'
            def _make():
                return TextCondMamba(d_model=dim, d_state=d_state, cond_target=cond_target)
        elif text_cond:
            self.ssm_impl = 'textcond_pytorch'
            def _make():
                return SelectiveScan(dim, d_state, dt_rank, text_cond=True,
                                     cond_target=cond_target)
        elif HAS_MAMBA_SSM:
            self.ssm_impl = 'official'
            def _make():
                return _CudaMamba(d_model=dim, d_state=d_state)
        else:
            self.ssm_impl = 'pytorch'
            def _make():
                return SelectiveScan(dim, d_state, dt_rank)

        if scan_mode == 'cross':
            # 4-direction cross-scan (VMamba style)
            self.ssms = nn.ModuleList([_make() for _ in range(4)])
        else:
            self.ssm_fwd = _make()
            self.ssm_bwd = _make()

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )
        self.fwd_w = nn.Parameter(torch.ones(1))
        self.bwd_w = nn.Parameter(torch.ones(1))

    def _cross_scan(self, x_n, seq_shape, t_vec=None):
        """VMamba 4-direction scan: rows fwd/bwd, cols fwd/bwd."""
        B, N, C = x_n.shape
        H, W = seq_shape
        x_img = x_n.reshape(B, H, W, C)

        # 1. Row forward (row-major flatten)
        s1 = x_img
        # 2. Row backward (flip each row)
        s2 = torch.flip(x_img, dims=[2])
        # 3. Col forward (transpose, then flatten)
        s3 = x_img.transpose(1, 2)
        # 4. Col backward
        s4 = torch.flip(x_img, dims=[1]).transpose(1, 2)

        scans = [s1, s2, s3, s4]
        outs = []
        for i, s in enumerate(scans):
            seq = s.reshape(B, H * W, C)
            out_seq = self.ssms[i](seq, t_vec) if self.text_cond else self.ssms[i](seq)
            out_img = out_seq.reshape(B, H, W, C)
            outs.append(out_img)

        o1, o2, o3, o4 = outs
        # Inverse transforms
        o2 = torch.flip(o2, dims=[2])
        o3 = o3.transpose(1, 2)
        o4 = o4.transpose(1, 2)
        o4 = torch.flip(o4, dims=[1])

        fused = (o1 + o2 + o3 + o4) / 4.0
        return fused.reshape(B, N, C)

    def forward(self, x, seq_shape=None, t_vec=None):
        residual = x
        x_n = self.norm1(x)

        # FiLM 文本调制（作用于块输入特征, 非 SSM 参数）
        if self.text_film and t_vec is not None:
            g, b = self.film(t_vec).chunk(2, dim=-1)
            x_n = x_n * (1 + g.unsqueeze(1)) + b.unsqueeze(1)

        if self.scan_mode == 'cross' and seq_shape is not None:
            out = self._cross_scan(x_n, seq_shape, t_vec)
        elif self.text_cond:
            fwd = self.ssm_fwd(x_n, t_vec)
            bwd = self.ssm_bwd(torch.flip(x_n, dims=[1]), t_vec)
            bwd = torch.flip(bwd, dims=[1])
            out = self.fwd_w * fwd + self.bwd_w * bwd
        else:
            fwd = self.ssm_fwd(x_n)
            bwd = self.ssm_bwd(torch.flip(x_n, dims=[1]))
            bwd = torch.flip(bwd, dims=[1])
            out = self.fwd_w * fwd + self.bwd_w * bwd

        x = residual + out
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# Patch Embedding
# ============================================================================

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x).flatten(2).transpose(-1, -2)
        return self.dropout(x + self.position_embeddings)


# ============================================================================
# Helper: Conv1d + BN + ReLU
# ============================================================================

class ConvTransBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


# ============================================================================
# VisionMamba (replaces VisionTransformer)
# ============================================================================

class VisionMamba(nn.Module):
    """Same interface as VisionTransformer, uses Mamba blocks instead of Attention."""
    def __init__(self, config, vis, img_size, channel_num, patch_size, embed_dim,
                 depth=1, d_state=16, mlp_ratio=4., drop_rate=0.,
                 scan_mode='bidirectional', text_gate=True,
                 text_cond=False, cond_target='dtBC'):
        super().__init__()
        self.config = config
        self.vis = vis
        self.dim = embed_dim
        self.scan_mode = scan_mode
        self.text_cond = text_cond
        self.img_size = img_size
        self.patch_size = patch_size

        self.embeddings = PatchEmbedding(patch_size, img_size, channel_num)

        # ModuleList instead of Sequential so we can pass seq_shape per block
        self.blocks = nn.ModuleList([
            MambaBlock(dim=embed_dim, d_state=d_state, mlp_ratio=mlp_ratio, drop=drop_rate,
                       scan_mode=scan_mode, text_cond=text_cond, cond_target=cond_target)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 文本 → 扫描参数 的条件向量(注意力池化)
        self.text_summary = TextSummary(embed_dim) if text_cond else None

        n_patches = (img_size // patch_size) * (img_size // patch_size)
        self.seq_shape = (img_size // patch_size, img_size // patch_size)

        self.CTBN = ConvTransBN(embed_dim, embed_dim // 2)
        self.CTBN2 = ConvTransBN(embed_dim * 2, embed_dim)
        self.CTBN3 = ConvTransBN(10, n_patches)

        # Learnable text gate: model decides per-patch how much text to inject
        self.text_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        ) if text_gate else None

    def _run_blocks(self, x, t_vec=None):
        for blk in self.blocks:
            x = blk(x, seq_shape=self.seq_shape if self.scan_mode == 'cross' else None,
                    t_vec=t_vec)
        return x

    def forward(self, x, skip_x, text, reconstruct=False):
        # 文本条件向量: 直接送入 Mamba 的选择性扫描(Δ/B/C 调制)
        t_vec = None
        if self.text_cond and self.text_summary is not None and text is not None:
            t_vec = self.text_summary(text)   # (B, embed_dim)

        if not reconstruct:
            x = self.embeddings(x)            # (B, n_patches, embed_dim)
            if self.dim == 64 and text is not None:
                if self.text_gate is not None:
                    # gated text injection
                    gate = self.text_gate(x)               # (B, n_patches, embed_dim)
                    x = x + gate * self.CTBN3(text)
                else:
                    x = x + self.CTBN3(text)
            x = self._run_blocks(x, t_vec)   # Mamba blocks
            x = self.norm(x)
        else:
            x = self._run_blocks(x, t_vec)
            x = self.norm(x)

        if (self.dim == 64 and not reconstruct) or (self.dim == 512 and reconstruct):
            return x
        elif not reconstruct:
            x = x.transpose(1, 2)
            x = self.CTBN(x)
            x = x.transpose(1, 2)
            return torch.cat([x, skip_x], dim=2)
        elif reconstruct:
            skip_x = skip_x.transpose(1, 2)
            skip_x = self.CTBN2(skip_x)
            skip_x = skip_x.transpose(1, 2)
            return x + skip_x


# ============================================================================
# Reconstruct (same as Vit)
# ============================================================================

class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None
        B, n_patch, hidden = x.size()
        h = w = int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x)
        return self.activation(self.norm(self.conv(x)))
