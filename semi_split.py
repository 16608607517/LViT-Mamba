# -*- coding: utf-8 -*-
"""
半监督数据划分器：按比例将有标注数据集分为「有标注」和「无标注」子集。

划分逻辑：
- labeled:  随机选取 ratio 比例（有 mask + 文本）
- unlabeled: 其余（只有图像 + 文本，无 mask 监督）

支持固定种子保证可复现。输出划分好的文件名列表，供 train_semi.py 使用。
"""
import os
import torch
import random


def _label_path(image_dir, name, task):
    """由图像名推出对应 mask 路径（Covid19 为 mask_ 前缀，其余同名 .png）。"""
    label_dir = os.path.join(os.path.dirname(image_dir.rstrip('/')), 'labelcol')
    if task in ('Covid19', 'QaTaCOV19'):
        return os.path.join(label_dir, 'mask_' + name)
    return os.path.join(label_dir, os.path.splitext(name)[0] + '.png')


def _foreground_ratio(image_dir, name, task):
    """前景面积占比（用于分层抽样）。读不到 mask 时返回 0。"""
    import cv2
    p = _label_path(image_dir, name, task)
    if not os.path.exists(p):
        return 0.0
    m = cv2.imread(p, 0)
    if m is None or m.size == 0:
        return 0.0
    return float((m > 0).sum()) / float(m.size)


def _stratified_sample(images, keys, ratio, seed, n_strata=10):
    """按 keys 的等频分位数分层，每层按比例抽样（层内随机）。"""
    n_labeled = max(1, int(len(images) * ratio))
    order = sorted(range(len(images)), key=lambda i: keys[i])
    n = len(order)
    strata = [[] for _ in range(n_strata)]
    for rank, idx in enumerate(order):
        strata[min(rank * n_strata // n, n_strata - 1)].append(images[idx])
    quota = [len(s) * ratio for s in strata]
    base = [int(q) for q in quota]
    rem = n_labeled - sum(base)
    for i in sorted(range(n_strata), key=lambda j: quota[j] - base[j], reverse=True)[:rem]:
        base[i] += 1
    labeled = []
    for s, k in zip(strata, base):
        if k > 0:
            labeled += random.sample(s, min(k, len(s)))
    labeled = sorted(set(labeled))
    if len(labeled) < n_labeled:                      # 极端情况补齐
        rest = [i for i in images if i not in labeled]
        labeled = sorted(labeled + random.sample(rest, n_labeled - len(labeled)))
    return labeled[:n_labeled]


def _text_key(image_dir, name, task, pcb):
    """用文本嵌入的均值向量作为语义分层依据（返回 numpy 向量）。"""
    import numpy as np
    if task in ('Covid19', 'QaTaCOV19'):
        key = 'mask_' + name
    else:
        key = os.path.splitext(name)[0] + '.png'
    v = pcb.get(key)
    if v is None:
        return np.zeros(768, dtype=np.float32)
    return v.mean(0).numpy().astype(np.float32)


def split_supervised(image_dir, ratio=0.1, seed=666, strategy='random',
                     task='Covid19', n_strata=10):
    """
    将有标注训练集划分为有标注/无标注两个子集。

    Args:
        image_dir: 训练集 img 目录路径
        ratio: 有标注样本比例 (0, 1]
        seed: 随机种子
        strategy:
            'random'     随机抽样（默认基线）
            'area_strat' 按病灶面积占比分层（保证标注覆盖病灶大小分布）
            'text_strat' 按文本语义分层（用预计算 BERT 嵌入的 PCA 投影分层，
                         保证标注覆盖不同的临床表述/病灶类型）
        task: 数据集名
        n_strata: 分层数

    Returns:
        labeled, unlabeled: 图像文件名列表
    """
    images = sorted(os.listdir(image_dir))
    random.seed(seed)
    n_labeled = max(1, int(len(images) * ratio))

    if strategy == 'random':
        labeled = sorted(random.sample(images, n_labeled))
    elif strategy == 'area_strat':
        keys = [_foreground_ratio(image_dir, im, task) for im in images]
        labeled = _stratified_sample(images, keys, ratio, seed, n_strata)
    elif strategy == 'text_strat':
        import numpy as np
        from sklearn.decomposition import PCA
        base = os.path.dirname(image_dir.rstrip('/'))
        pcb = torch.load(os.path.join(base, 'precomputed_bert.pt'), map_location='cpu')
        X = np.stack([_text_key(image_dir, im, task, pcb) for im in images])
        proj = PCA(n_components=1, random_state=seed).fit_transform(X).ravel()
        labeled = _stratified_sample(images, list(proj), ratio, seed, n_strata)
    else:
        raise ValueError('unknown strategy: %s' % strategy)

    unlabeled = sorted([i for i in images if i not in labeled])
    return labeled, unlabeled


def save_split(save_path, labeled, unlabeled, ratio):
    """保存划分结果到文件，便于复现与统计。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(f'# ratio={ratio} labeled={len(labeled)} unlabeled={len(unlabeled)}\n')
        f.write('# labeled:\n')
        for name in labeled:
            f.write(name + '\n')
        f.write('# unlabeled:\n')
        for name in unlabeled:
            f.write(name + '\n')
    return save_path


if __name__ == '__main__':
    import Config as config

    task = 'MosMedDataPlus'
    img_dir = f'./datasets/{task}/Train_Folder/img/'

    for ratio in [0.1, 0.2, 0.5]:
        labeled, unlabeled = split_supervised(img_dir, ratio=ratio)
        save_path = f'./semi_splits/{task}_ratio{int(ratio*100)}.txt'
        save_split(save_path, labeled, unlabeled, ratio)
        print(f'ratio={ratio:.0%}: labeled={len(labeled)}, unlabeled={len(unlabeled)} -> {save_path}')
