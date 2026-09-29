# -*- coding: utf-8 -*-
"""分辨率-效率基准: 同一模型在多个输入分辨率下的 显存/速度/精度 对比。

用途: 支撑"Mamba 线性复杂度在高分辨率下的优势"这一论证（文献中 2D 医学领域几乎无人系统做过）。

用法:
  python bench_resolution.py --task Covid19 --backbone mamba --session <s> --gpu 0 \
      --res 224,448,672 --n 30 [--cond_target dtBC]

说明:
  - 模型按目标分辨率重建（patch 数与位置编码相应变化）
  - 位置编码做线性插值以适配新分辨率（标准做法）
  - 每档测: 峰值显存 / 单图延迟 / 吞吐 / 前 n 张的 Dice
输出: ~/LViT/results_resolution_bench.csv + 终端表格
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('LVIT_GPU', '0')

import Config as config                                              # noqa: E402
from Load_Dataset import ValGenerator, ImageToImage2D                # noqa: E402
from torch.utils.data import DataLoader                              # noqa: E402
from nets.LVMamba import LVMamba                                     # noqa: E402
from nets.LViT import LViT                                           # noqa: E402


def build(task, backbone, use_text, res, cond_target):
    cfg = config.get_CTranS_config()
    if backbone == 'transformer':
        return LViT(cfg, n_channels=3, n_classes=1, img_size=res)
    return LVMamba(cfg, n_channels=3, n_classes=1, img_size=res, backbone='mamba',
                   use_text=use_text, mamba_depth=1, scan_mode='bidirectional',
                   text_gate=False, text_cond=bool(cond_target),
                   cond_target=cond_target or 'dtBC')


def _resize_dim(t, dim, size):
    """沿单一维度做线性插值（用于分辨率变化时的权重适配）。"""
    if t.shape[dim] == size:
        return t
    perm = [i for i in range(t.dim()) if i != dim] + [dim]
    inv = [0] * t.dim()
    for i, p in enumerate(perm):
        inv[p] = i
    v = t.permute(*perm).float()                          # (..., L)
    L = v.shape[-1]
    if L == 1:                                            # 退化: 复制
        v = v.repeat(*([1] * (v.dim() - 1)), size)
    else:
        v2 = F.interpolate(v.reshape(-1, 1, L), size=size,
                           mode='linear', align_corners=False)   # (M,1,size)
        v = v2.reshape(*v.shape[:-1], size)
    return v.permute(*inv)


def load_ckpt(model, path, res):
    """加载权重; 分辨率变化导致的尺寸不匹配用线性插值适配（记录件数）。"""
    ck = torch.load(path, map_location='cpu')
    sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    tgt = dict(model.named_parameters())
    tgt.update(dict(model.named_buffers()))
    resized = []
    for k in list(sd):
        if k not in tgt or sd[k].shape == tgt[k].shape:
            continue
        diff = [i for i in range(sd[k].dim()) if sd[k].shape[i] != tgt[k].shape[i]]
        if len(diff) != 1:
            continue                                      # 多维多处不同 → 交给 load_state_dict 报错
        d = diff[0]
        sd[k] = _resize_dim(sd[k], d, tgt[k].shape[d])
        resized.append('%s(dim%d %d->%d)' % (k, d, ck and 0 or 0, tgt[k].shape[d]))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_mod_missing = len([k for k in missing if 't_dt' in k or 't_B' in k or 't_C' in k])
    return len(sd) - len(unexpected), len(sd), len(resized), n_mod_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='Covid19')
    ap.add_argument('--backbone', default='mamba', choices=['mamba', 'transformer'])
    ap.add_argument('--session', required=True)
    ap.add_argument('--use_text', type=lambda x: x.lower() == 'true', default=True)
    ap.add_argument('--cond_target', default='')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--res', default='224,448,672')
    ap.add_argument('--n', type=int, default=30, help='每档测 Dice 的图片数')
    ap.add_argument('--batch', type=int, default=4)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    res_list = [int(r) for r in args.res.split(',')]
    OUT = os.path.expanduser('~/LViT/results_resolution_bench.csv')
    rows = []

    for res in res_list:
        try:
            model = build(args.task, args.backbone, args.use_text, res, args.cond_target).cuda()
            path = './%s/LVMamba/%s/models/best_model-LVMamba.pth.tar' % (args.task, args.session)
            nload, ntot, nint, nmod = load_ckpt(model, path, res)
            model.eval()

            tf = ValGenerator(output_size=[res, res])
            ds = ImageToImage2D('./datasets/%s/Test_Folder/' % args.task, args.task,
                                {}, tf, image_size=res)
            loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # 预热
            dummy = torch.randn(args.batch, 3, res, res).cuda()
            dtxt = torch.randn(args.batch, 10, 768).cuda()
            with torch.no_grad():
                for _ in range(2):
                    model(dummy, dtxt if args.use_text else None)
            torch.cuda.synchronize()

            t0 = time.time()
            nt, tot, dice_sum = 0, 0, 0.0
            with torch.no_grad():
                for b, names in loader:
                    if nt >= args.n:
                        break
                    img, lab, txt = b['image'].cuda(), b['label'].cuda().float(), b['text'].cuda()
                    out = model(img, txt if args.use_text else None)
                    pred = (out > 0.5).float()
                    p, g = pred.reshape(pred.shape[0], -1), lab.reshape(lab.shape[0], -1)
                    inter = (p * g).sum(1)
                    dice_sum += (2 * inter / (p.sum(1) + g.sum(1) + 1e-5)).sum().item()
                    nt += img.shape[0]
                    tot += img.shape[0]
            torch.cuda.synchronize()
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2
            dice = dice_sum / max(nt, 1)
            rows.append([args.task, args.backbone, args.session, res, nt,
                         '%.1f' % peak, '%.1f' % (dt / max(nt, 1) * 1000),
                         '%.1f' % (nt / dt), '%.4f' % dice,
                         '%d/%d' % (nload, ntot), nint, nmod])
            print('[%s %s] res=%d  峰值显存=%.0fMB  延迟=%.0fms/img  吞吐=%.1f img/s  Dice=%.4f'
                  % (args.task, args.backbone, res, peak, dt / max(nt, 1) * 1000, nt / dt, dice), flush=True)
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print('[%s %s] res=%d 失败: %s' % (args.task, args.backbone, res, str(e)[:150]), flush=True)
            rows.append([args.task, args.backbone, args.session, res, 0, 'ERR', 'ERR', 'ERR', 'ERR',
                         str(e)[:60], 0, 0])

    with open(OUT, 'a', newline='') as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(['task', 'backbone', 'session', 'res', 'n', 'peak_mem_MB', 'latency_ms',
                        'throughput_img_s', 'dice', 'params_loaded', 'n_interp', 'mod_missing'])
        w.writerows(rows)
    print('BENCH_DONE -> %s' % OUT, flush=True)


if __name__ == '__main__':
    main()
