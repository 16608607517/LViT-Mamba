# -*- coding: utf-8 -*-
"""
对比标签混合匹配：为每个样本预计算最匹配的对比标签索引。

策略（混合）：
1. 语义过滤：优先 (laterality, n_areas) 精确匹配的候选
2. 余弦精化：在候选内用 BERT 文本余弦相似度选最相似的
3. 兜底：语义无匹配时，退化为纯余弦

输出：{img_name: contrast_idx} 字典，训练时直接查表。
"""
import re
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

FULL = frozenset({'upper', 'middle', 'lower'})
NUM = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
       'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}


def parse_profile(s):
    """解析文本 → (laterality, n_areas, {side: subs})"""
    s = s.lower()
    laterality = 0 if 'unilateral' in s else (1 if 'bilateral' in s else -1)
    n = 0
    m = re.search(r'(\w+)\s+infected area', s)
    if m and m.group(1) in NUM:
        n = NUM[m.group(1)]
    profile = {}
    for p in re.split(r'[,;]| and ', s):
        if 'lung' not in p:
            continue
        p = p.strip()
        if 'left' in p:
            side = 'left'
        elif 'right' in p:
            side = 'right'
        else:
            continue
        if 'all' in p:
            profile[side] = FULL
        else:
            profile[side] = frozenset(sub for sub in ['upper', 'middle', 'lower'] if sub in p)
    return laterality, n, profile


def semantic_match_score(sample_f, cl_f):
    """语义匹配得分：laterality/n_areas 强匹配 + 位置 overlap"""
    lat_s, n_s, prof_s = sample_f
    lat_c, n_c, prof_c = cl_f
    score = 0.0
    if lat_s == lat_c and lat_s >= 0:
        score += 3.0
    if n_s == n_c and n_s > 0:
        score += 3.0
    # 位置 overlap（带惩罚：未覆盖区域算 miss）
    total_c = sum(len(v) for v in prof_c.values()) or 1
    matched = 0
    for side, subs_c in prof_c.items():
        subs_s = prof_s.get(side, frozenset())
        matched += len(subs_s & subs_c)
    score += 2.0 * (matched / total_c)
    return score


class ContrastMatch:
    """为给定文本字典预计算对比标签索引。"""

    def __init__(self, lv_xlsx='d:/LViT_official/LV_loss/LV_loss.xlsx'):
        df = pd.read_excel(lv_xlsx)
        self.contrast_texts = df['Description'].tolist()
        self.cl_feats = [parse_profile(t) for t in self.contrast_texts]
        # 预计算对比标签 BERT 文本特征 (n_cl, 768)
        self.cl_emb = F.normalize(self._embed(self.contrast_texts), dim=-1)

    def _embed(self, texts):
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased').cuda()
        model.eval()
        embeds = []
        with torch.no_grad():
            for t in texts:
                inp = tokenizer(t.split('\n')[0], return_tensors='pt',
                                padding=True, truncation=True, max_length=512)
                inp = {k: v.cuda() for k, v in inp.items()}
                out = model(**inp).last_hidden_state.mean(dim=1).squeeze(0).cpu()
                embeds.append(out)
        return torch.stack(embeds).cuda()

    def _text_embed(self, text_embed_np):
        """样本 BERT 嵌入 (10,768) → 归一化 mean-pool"""
        t = torch.from_numpy(text_embed_np).float().cuda()
        return F.normalize(t.mean(dim=0, keepdim=True), dim=-1)

    def match_one(self, text_embed_np):
        """混合匹配单个样本，返回对比标签索引。"""
        t = self._text_embed(text_embed_np)  # (1,768)
        text = None  # 需要原始文本做语义解析，由外部传入
        # 语义无法从 embedding 解析，走余弦兜底（若外部未提供文本）
        sim = t @ self.cl_emb.t()
        return int(sim.argmax(dim=1).item())

    def match_with_text(self, raw_text, text_embed_np):
        """混合匹配：语义过滤 + 余弦精化。"""
        sf = parse_profile(raw_text)
        t = self._text_embed(text_embed_np)

        # 候选：语义得分高者优先
        scores = [semantic_match_score(sf, cf) for cf in self.cl_feats]
        max_s = max(scores)
        if max_s >= 5.0:
            # 有语义匹配，候选 = 得分最高的一组
            cands = [i for i, s in enumerate(scores) if s >= max_s - 1e-6]
        else:
            cands = list(range(len(self.cl_feats)))
        # 余弦精化
        sim = t @ self.cl_emb[cands].t()
        return cands[int(sim.argmax(dim=1).item())]


def precompute_matches(task, split='Train', max_text_len=10):
    """预计算指定数据集的对比标签索引，保存 {img: idx}。

    Returns:
        dict {img_name: contrast_idx}
    """
    from utils import read_text
    base = f'./datasets/{task}/{split}_Folder/'
    # Covid19 的 train 文本合并为 Train_Val_text.xlsx
    text_file = 'Train_Val_text.xlsx' if (task in ['Covid19', 'QaTaCOV19'] and split == 'Train') \
        else f'{split}_text.xlsx'
    row_text = read_text(base + text_file)
    pre_path = base + 'precomputed_bert.pt'
    pre = torch.load(pre_path, map_location='cpu') if os.path.exists(pre_path) else None

    cm = ContrastMatch()
    result = {}
    for text_key, desc in row_text.items():
        # text_key 为 row_text 的键；Covid19 是 mask_ 前缀，需映射为图片名（与 batch names 一致）
        img_name = text_key.replace('mask_', '') if task in ['Covid19', 'QaTaCOV19'] else text_key
        emb = pre[text_key].numpy() if pre and text_key in pre else None
        if emb is None:
            continue
        emb = emb[:max_text_len]
        idx = cm.match_with_text(desc, emb)
        result[img_name] = idx

    os.makedirs(f'./semi_results/matches/', exist_ok=True)
    save_path = f'./semi_results/matches/{task}_{split}_contrast.pt'
    torch.save(result, save_path)
    print(f'Saved {len(result)} matches -> {save_path}')

    from collections import Counter
    c = Counter(result.values())
    total = len(result)
    top = max(c.values())
    print(f'分布: 用到 {len(c)}/14 对比标签, 最大集中 {top/total:.1%}')
    for k in sorted(c)[:6]:
        print(f'  fig{k+1}: {c[k]}')
    return result


if __name__ == '__main__':
    precompute_matches('MosMedDataPlus', 'Train')
