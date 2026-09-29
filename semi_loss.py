# -*- coding: utf-8 -*-
"""
论文原版 LV Loss (Language-Vision Loss) —— 基于对比标签 (Contrastive Labels)。

论文方法（IEEE TMI 2024, LViT）:
- 对比标签: 一组 (结构化文本 → 对应 mask) 对，提供病灶位置先验
- TextSim (Eqn.16): 无标注样本文本特征 与 对比标签文本特征 的余弦相似度，选最相似的
- ImgSim  (Eqn.17): 模型预测特征 与 选中对比标签 mask 特征的余弦相似度
- L_LV    (Eqn.18): 1 - ImgSim

LV Loss 仅用于无标注数据 (α=0.1)，避免伪标签质量恶化。
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd


class ContrastiveLVLoss(nn.Module):
    """论文原版 LV Loss：利用对比标签的文本相似度选择 + 掩码余弦相似度。

    Args:
        lv_xlsx: 对比标签文本文件路径 (LV_loss.xlsx)
        lv_img_dir: 对比标签 mask 图片目录 (fig1.png ... fig14.png)
        image_size: mask resize 尺寸（与模型输入一致）
        max_text_len: 文本嵌入截断长度
    """

    def __init__(self, lv_xlsx=None, lv_img_dir=None,
                 image_size=224, max_text_len=10):
        super().__init__()
        # 路径自动探测：Windows 用 d:/，Linux/WSL 用 /mnt/d/
        if lv_xlsx is None:
            candidates = [os.path.expanduser('~/LViT/LV_loss/LV_loss.xlsx'),
                          'd:/LViT_official/LV_loss/LV_loss.xlsx',
                          '/mnt/d/LViT_official/LV_loss/LV_loss.xlsx']
            for c in candidates:
                if os.path.exists(c):
                    lv_xlsx = c
                    break
        if lv_img_dir is None:
            lv_img_dir = os.path.dirname(lv_xlsx) + '/'
        self.image_size = image_size
        self.max_text_len = max_text_len

        # 1. 读取对比标签文本
        df = pd.read_excel(lv_xlsx)
        self.contrast_texts = df['Description'].tolist()
        self.contrast_names = df['Image'].tolist()
        n_cl = len(self.contrast_texts)

        # 2. 加载对比标签 mask（350x350，多值分区 → 二值化）
        contrast_masks = []
        for name in self.contrast_names:
            mask = cv2.imread(os.path.join(lv_img_dir, name), 0)
            mask = cv2.resize(mask, (image_size, image_size))
            mask = (mask > 0).astype(np.float32)  # 二值化：>0 即病灶区域
            contrast_masks.append(mask)
        # (n_cl, H*W) 归一化特征，供 ImgSim 余弦
        masks_flat = np.stack(contrast_masks).reshape(n_cl, -1)          # (n_cl, H*W)
        self.register_buffer('contrast_masks', torch.from_numpy(masks_flat))
        self.register_buffer('contrast_masks_norm',
                             F.normalize(torch.from_numpy(masks_flat), dim=-1))

        # 3. 对比标签文本 → BERT 嵌入（优先预计算缓存，服务器无外网时离线可用）
        emb_cache = os.path.join(lv_img_dir, 'contrast_text_emb.pt')
        if os.path.exists(emb_cache):
            self.contrast_text_emb = torch.load(emb_cache, map_location='cpu')
        else:
            self.contrast_text_emb = self._embed_texts(self.contrast_texts).cpu()
            torch.save(self.contrast_text_emb, emb_cache)
        self.contrast_text_emb = F.normalize(self.contrast_text_emb, dim=-1)
        self.contrast_text_emb = self.contrast_text_emb.to(self.contrast_masks.device)

    def _embed_texts(self, texts):
        """用 BERT 编码文本，返回 mean-pool 特征 (n, 768)。"""
        from transformers import BertTokenizer, BertModel

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()

        embeds = []
        with torch.no_grad():
            for t in texts:
                sent = t.split('\n')[0]
                inputs = tokenizer(sent, return_tensors='pt', padding=True,
                                   truncation=True, max_length=512)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                out = model(**inputs).last_hidden_state  # (1, seq, 768)
                emb = out.mean(dim=1).squeeze(0).cpu()   # (768,)
                embeds.append(emb)
        return torch.stack(embeds).to(self.contrast_masks.device)

    def forward(self, pred, text_embed, contrast_idx=None):
        """
        论文 LV Loss。

        Args:
            pred: (B, 1, H, W) 无标注样本的模型预测分割图
            text_embed: (B, L, 768) 无标注样本的 BERT 文本嵌入
            contrast_idx: (B,) 预计算的对比标签索引（混合匹配），None 则用余弦 TextSim
        Returns:
            L_LV scalar
        """
        device = pred.device
        B = pred.shape[0]
        pred_flat = pred.reshape(B, -1)                       # (B, H*W)
        pred_flat = F.normalize(pred_flat, dim=-1)

        cmn = self.contrast_masks_norm.to(device)             # (n_cl, H*W)

        if contrast_idx is not None:
            # 混合匹配：直接使用预计算索引
            best_idx = contrast_idx.to(device)
        else:
            # 论文 TextSim: 余弦选择
            t = F.normalize(text_embed.mean(dim=1), dim=-1)   # (B, 768)
            ct = self.contrast_text_emb.to(device)            # (n_cl, 768)
            sim = t @ ct.t()
            best_idx = sim.argmax(dim=1)

        mask_norm = cmn[best_idx]                             # (B, H*W)

        # ImgSim: 预测特征 与 对比 mask 特征的余弦相似度 (Eqn.17)
        img_sim = (pred_flat * mask_norm).sum(dim=1)          # (B,)

        # L_LV = 1 - ImgSim (Eqn.18)
        lv_loss = (1.0 - img_sim).mean()
        return lv_loss


if __name__ == '__main__':
    lv = ContrastiveLVLoss()
    pred = torch.rand(4, 1, 224, 224)
    text_embed = torch.rand(4, 10, 768)
    loss = lv(pred, text_embed)
    print(f'LV Loss test: {loss.item():.4f}')
    print(f'对比标签数: {len(lv.contrast_texts)}, mask 维度: {lv.contrast_masks.shape}')
