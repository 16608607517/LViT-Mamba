# -*- coding: utf-8 -*-
"""文本破坏矩阵评估: 在已训好的模型上系统性破坏文本, 测 Dice 变化。

回答的问题:
  - 模型是真的在用文本语义, 还是学了个捷径?
  - Mamba 骨干相比 Transformer 骨干, 对文本的依赖是更强还是更弱?

用法:
  python eval_text_corruption.py --task Covid19 --backbone transformer --session <session名> --use_text True [--limit N]

输出:
  - 每个破坏模式的 per-image Dice 均值/标准差/最差 + global Dice
  - results_corruption_<task>_<backbone>_<session>.csv  (汇总表)
  - corruption_dice_<task>_<backbone>_<session>.npz     (逐图 Dice, 供后续显著性检验)
"""
import os
import sys
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 选卡: 优先 LVIT_GPU 环境变量, 否则沿用 Config 里已设的值
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(
    "LVIT_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

import Config as config                                    # noqa: E402
from Load_Dataset import ValGenerator, ImageToImage2D      # noqa: E402
from nets.LVMamba import LVMamba                           # noqa: E402
from nets.LViT import LViT                                 # noqa: E402

MODES = ['full', 'zero', 'shuffle', 'trunc1', 'trunc3', 'trunc5', 'random', 'template']


class TextCorruptDataset(Dataset):
    """在基础数据集上, 对每条样本的文本嵌入施加指定破坏。"""

    def __init__(self, base, mode, all_texts=None, perm=None):
        self.base = base
        self.mode = mode
        self.all_texts = all_texts      # (N, 10, 768) 数据集全部文本, random 模式用
        self.perm = perm                # 随机配对索引
        self.mean_text = all_texts.mean(dim=0) if all_texts is not None else None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample, name = self.base[idx]
        text = sample['text']           # (10, 768) tensor
        if self.mode == 'zero':
            text = torch.zeros_like(text)
        elif self.mode == 'shuffle':
            p = torch.randperm(text.shape[0])
            text = text[p]
        elif self.mode.startswith('trunc'):
            k = int(self.mode[5:])
            text = text.clone()
            text[k:] = 0
        elif self.mode == 'random':
            text = self.all_texts[self.perm[idx]].clone()
        elif self.mode == 'template':
            text = self.mean_text.clone()   # (10, 768) 数据集平均文本
        sample['text'] = text
        return sample, name


def build_model(task, backbone, use_text, cond_target=''):
    config_vit = config.get_CTranS_config()
    if config.model_name == 'LViT':
        return LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
    return LVMamba(config_vit, n_channels=config.n_channels, n_classes=config.n_labels,
                   backbone=backbone, use_text=use_text,
                   mamba_depth=config.mamba_depth, scan_mode=config.scan_mode,
                   text_gate=config.text_gate,
                   text_cond=bool(cond_target), cond_target=cond_target or 'dtBC')


def collect_texts(test_path, task):
    """按数据集索引顺序取出全部文本嵌入 (与 ImageToImage2D 的排序一致)。"""
    pcb_path = os.path.join(test_path, 'precomputed_bert.pt')
    pcb = torch.load(pcb_path, map_location='cpu')
    if task in ['Covid19', 'QaTaCOV19']:
        names = sorted(os.listdir(os.path.join(test_path, 'labelcol')))
    else:
        imgs = sorted(os.listdir(os.path.join(test_path, 'img')))
        names = [im[:-3] + 'png' for im in imgs]
    texts = []
    for n in names:
        t = pcb[n][:10, :].float()
        if t.shape[0] < 10:                      # 补齐到 10 行
            t = torch.cat([t, torch.zeros(10 - t.shape[0], t.shape[1])], dim=0)
        texts.append(t)
    return torch.stack(texts)


def run_mode(model, loader, limit=None):
    """跑完一个破坏模式, 返回逐图 Dice 数组 + 全局 TP/FP/FN。"""
    dices = []
    tp = fp = fn = 0.0
    with torch.no_grad():
        for i, (sampled_batch, names) in enumerate(loader, 1):
            if limit and i > limit:
                break
            img = sampled_batch['image'].cuda()
            lab = sampled_batch['label'].cuda().float()
            txt = sampled_batch['text'].cuda()
            out = model(img, txt)
            pred = (out > 0.5).float()
            p = pred.reshape(-1)
            g = lab.reshape(-1)
            inter = (p * g).sum().item()
            dice = 2 * inter / (p.sum().item() + g.sum().item() + 1e-5)
            dices.append(dice)
            tp += inter
            fp += (p * (1 - g)).sum().item()
            fn += ((1 - p) * g).sum().item()
            torch.cuda.empty_cache()
    dices = np.array(dices)
    gdice = 2 * tp / (2 * tp + fp + fn + 1e-5)
    return dices, gdice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--backbone', default='transformer')
    ap.add_argument('--session', default='')
    ap.add_argument('--use_text', type=lambda x: x.lower() == 'true', default=True)
    ap.add_argument('--semi', action='store_true',
                    help='加载半监督模型 semi_results/{tag}/best_model.pth（配合 --ratio）')
    ap.add_argument('--ratio', type=float, default=0.25)
    ap.add_argument('--cond_target', default='',
                    help='非空则按文本调制模型构造（如 dtBC/B/C），用于创新点①的机制验证')
    ap.add_argument('--limit', type=int, default=None, help='只跑前 N 张(调试用)')
    ap.add_argument('--outdir', default=os.path.expanduser('~/LViT'))
    args = ap.parse_args()

    test_path = './datasets/%s/Test_Folder/' % args.task
    if args.semi:
        semi_tag = '%s_r%d_%s_text%s' % (args.task, int(args.ratio * 100),
                                         args.backbone, str(args.use_text)[0])
        model_path = './semi_results/%s/best_model.pth' % semi_tag
        tag = semi_tag + '_semi'
    else:
        model_path = './%s/LVMamba/%s/models/best_model-LVMamba.pth.tar' % (args.task, args.session)
        tag = '%s_%s_%s' % (args.task, args.backbone, args.session)
    if args.cond_target:
        tag += '_tc' + args.cond_target
    print('=== %s ===' % tag, flush=True)

    ckpt = torch.load(model_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model = build_model(args.task, args.backbone, args.use_text, args.cond_target).cuda()
    # 兼容 DataParallel 训练的 checkpoint(键名带 module. 前缀)
    sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_loaded = len(sd) - len(unexpected)
    print('model loaded: %s  [params %d/%d]' % (model_path, n_loaded, len(sd)), flush=True)
    assert n_loaded > 0.5 * len(sd), '参数加载失败(只匹配 %d/%d)! 检查键名' % (n_loaded, len(sd))
    model.eval()

    tf = ValGenerator(output_size=[config.img_size, config.img_size])
    test_text = {}
    test_ds = ImageToImage2D(test_path, args.task, test_text, tf, image_size=config.img_size)
    all_texts = collect_texts(test_path, args.task)
    n = len(test_ds)
    g = torch.Generator().manual_seed(666)
    perm = torch.randperm(n, generator=g).tolist()
    perm = [p if p != i else (p + 1) % n for i, p in enumerate(perm)]   # 避免自己配自己

    results = {}
    for mode in MODES:
        ds = TextCorruptDataset(test_ds, mode, all_texts=all_texts, perm=perm)
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        dices, gdice = run_mode(model, loader, limit=args.limit)
        results[mode] = dices
        base = results['full'].mean() if 'full' in results else dices.mean()
        print('%-9s n=%d  per-image Dice=%.4f ± %.4f  worst=%.4f  global=%.4f  Δ=%.4f' % (
            mode, len(dices), dices.mean(), dices.std(), dices.min(), gdice,
            dices.mean() - base), flush=True)

    csv_path = os.path.join(args.outdir, 'results_corruption_%s.csv' % tag)
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['task', 'backbone', 'session', 'mode', 'n', 'dice_mean', 'dice_std',
                    'dice_worst', 'delta_vs_full'])
        base = results['full'].mean()
        for mode in MODES:
            d = results[mode]
            w.writerow([args.task, args.backbone, args.session, mode, len(d),
                        '%.6f' % d.mean(), '%.6f' % d.std(), '%.6f' % d.min(),
                        '%.6f' % (d.mean() - base)])
    np.savez(os.path.join(args.outdir, 'corruption_dice_%s.npz' % tag),
             **{m: results[m] for m in MODES})
    print('saved: %s' % csv_path, flush=True)
    print('CORRUPTION_DONE', flush=True)


if __name__ == '__main__':
    main()
