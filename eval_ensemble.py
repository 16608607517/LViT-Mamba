# -*- coding: utf-8 -*-
"""集成 + 测试时增强(TTA) + 阈值优化 的评测（三个零训练成本的性能提升策略）。

① 多模型集成：平均多个种子/多个变体的预测概率图
② TTA：翻转(水平/垂直/双翻转)，同一模型多视图平均
③ 阈值优化：★ 在**验证集**上扫描二值化阈值(0.30~0.70)，选出的阈值再应用到测试集
   —— 测试集上的阈值扫描仅作诊断输出，不得写入论文（属于测试集调参）

用法:
  python eval_ensemble.py --task Covid19 \
      --models "sess1:dtBC,sess2:dtBC,sess3:dtBC" --tta flip --thresh_search

输出: 单模型 / +TTA / 集成 / 阈值(验证集选定) 的 per-image Dice
环境变量 LVIT_GPU 指定显卡。
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('LVIT_GPU', '0')

import Config as config                                              # noqa: E402
from Load_Dataset import ValGenerator, ImageToImage2D                # noqa: E402
from nets.LVMamba import LVMamba                                     # noqa: E402

THRS = [round(0.30 + 0.05 * i, 2) for i in range(9)]                 # 0.30 .. 0.70


def build(cond_target, img_size):
    ct = cond_target if cond_target != 'none' else 'dtBC'
    return LVMamba(config.get_CTranS_config(), n_channels=3, n_classes=1, img_size=img_size,
                   backbone='mamba', use_text=True, mamba_depth=1, scan_mode='bidirectional',
                   text_gate=False, text_cond=(cond_target != 'none'), cond_target=ct)


def _resize_dim(t, dim, size):
    if t.shape[dim] == size:
        return t
    perm = [i for i in range(t.dim()) if i != dim] + [dim]
    inv = [0] * t.dim()
    for i, p in enumerate(perm):
        inv[p] = i
    v = t.permute(*perm).float()
    L = v.shape[-1]
    if L > 1:
        v = F.interpolate(v.reshape(-1, 1, L), size=size,
                          mode='linear', align_corners=False).reshape(*v.shape[:-1], size)
    return v.permute(*inv)


def load_weights(model, task, session):
    candidates = ['./%s/LVMamba/%s/models/best_model-LVMamba.pth.tar' % (task, session),
                  './%s/LVMamba/%s/best_model.pth' % (task, session)]
    path = next((p for p in candidates if os.path.exists(p)), None)
    assert path, 'checkpoint not found for session %s' % session
    ck = torch.load(path, map_location='cpu')
    sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    tgt = dict(model.state_dict())
    for k in list(sd):
        if k not in tgt or sd[k].shape == tgt[k].shape:
            continue
        diff = [i for i in range(sd[k].dim()) if sd[k].shape[i] != tgt[k].shape[i]]
        if len(diff) == 1:
            sd[k] = _resize_dim(sd[k], diff[0], tgt[k].shape[diff[0]])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # torchprofile 注入的统计 buffer 不算权重（total_ops/total_params），过滤后再校验
    def _is_prof(k):
        return k.endswith('.total_ops') or k.endswith('.total_params')
    real_unexpected = [k for k in unexpected if not _is_prof(k)]
    real_missing = [k for k in missing if not _is_prof(k)]
    n_mod = len([k for k in real_missing if '.t_dt' in k or '.t_B' in k or '.t_C' in k])
    return len(sd) - len(unexpected), len(sd), n_mod, real_missing, real_unexpected


def tta_forward(model, img, txt, tta):
    # 注意: LVMamba(n_classes=1) 的 last_activation 已是 Sigmoid，输出即概率，不可再套 sigmoid
    with torch.no_grad():
        outs = [model(img, txt)]
        if tta == 'flip':
            for dims in ([3], [2], [2, 3]):
                inv = torch.flip(img, dims=dims)
                outs.append(torch.flip(model(inv, txt), dims=dims))
    return torch.stack(outs, 0).mean(0)


def dice_at(prob, mask, thr):
    p = (prob > thr).float().reshape(prob.shape[0], -1)
    g = mask.float().reshape(mask.shape[0], -1)
    inter = (p * g).sum(1)
    return (2 * inter / (p.sum(1) + g.sum(1) + 1e-5)).cpu()


def collect(models, loader, tta, thrs, limit=None):
    """在给定 loader 上跑全部模型，返回各口径的 per-image Dice 均值。"""
    single = {i: [] for i in range(len(models))}
    tta_each = {i: [] for i in range(len(models))}
    ens, ens_thr = [], {t: [] for t in thrs}
    n_img = 0
    with torch.no_grad():
        for bi, (b, names) in enumerate(loader):
            if limit and bi * loader.batch_size >= limit:
                break
            img, lab, txt = b['image'].cuda(), b['label'].cuda().float(), b['text'].cuda()
            probs = []
            for i, m in enumerate(models):
                single[i].append(dice_at(m(img, txt), lab, 0.5))
                p_tta = tta_forward(m, img, txt, tta)
                if tta != 'none':
                    tta_each[i].append(dice_at(p_tta, lab, 0.5))
                probs.append(p_tta)
            p_ens = torch.stack(probs, 0).mean(0)
            ens.append(dice_at(p_ens, lab, 0.5))
            for t in thrs:
                ens_thr[t].append(dice_at(p_ens, lab, t))
            n_img += img.shape[0]
    cat = lambda L: torch.cat(L).numpy()                             # noqa: E731
    return dict(single={i: float(cat(v).mean()) for i, v in single.items()},
                tta={i: float(cat(v).mean()) for i, v in tta_each.items() if len(v)},
                ens=float(cat(ens).mean()),
                ens_thr={t: float(cat(v).mean()) for t, v in ens_thr.items()},
                n=n_img)


def make_loader(task, folder, img_size, batch):
    tf = ValGenerator(output_size=[img_size, img_size])
    ds = ImageToImage2D('./datasets/%s/%s/' % (task, folder), task, {}, tf, image_size=img_size)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='Covid19')
    ap.add_argument('--models', required=True, help='逗号分隔的 session:cond_target')
    ap.add_argument('--tta', default='flip', choices=['none', 'flip'])
    ap.add_argument('--thresh_search', action='store_true')
    ap.add_argument('--val_folder', default='Val_Folder', help='用于选阈值的验证集目录')
    ap.add_argument('--no_val', action='store_true', help='跳过验证集（仅调试用）')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--batch', type=int, default=4)
    args = ap.parse_args()

    specs = [s.split(':') for s in args.models.split(',')]
    img_size = config.img_size

    models = []
    for session, ct in specs:
        m = build(ct, img_size).cuda().eval()
        nl, nt, nm, miss, unexp = load_weights(m, args.task, session)
        print('loaded %s (%s): %d/%d 调制头missing=%d' % (session, ct, nl, nt, nm), flush=True)
        if miss:
            raise AssertionError('%s 有 %d 个真实权重未加载: %s' % (session, len(miss), miss[:8]))
        if unexp:
            print('  [warn] %s 有 %d 个多余key(忽略): %s' % (session, len(unexp), unexp[:5]), flush=True)
        models.append(m)

    # ---------- ① 验证集：选阈值 ----------
    best_t, val_res = 0.5, None
    if args.thresh_search and not args.no_val:
        print('\n--- 验证集 (%s) 选阈值 ---' % args.val_folder, flush=True)
        val_res = collect(models, make_loader(args.task, args.val_folder, img_size, args.batch),
                          args.tta, THRS, args.limit)
        best_t = max(THRS, key=lambda t: val_res['ens_thr'][t])
        print('  验证集集成 Dice@0.5 = %.4f' % val_res['ens'], flush=True)
        for t in THRS:
            print('    val thr=%.2f: %.4f%s' % (t, val_res['ens_thr'][t],
                                                ' ←选定' if t == best_t else ''), flush=True)
        print('  ★ 验证集选定阈值 = %.2f (val Dice %.4f)' % (best_t, val_res['ens_thr'][best_t]),
              flush=True)

    # ---------- ② 测试集：只用验证集选定的阈值 ----------
    print('\n--- 测试集评测 ---', flush=True)
    res = collect(models, make_loader(args.task, 'Test_Folder', img_size, args.batch),
                  args.tta, THRS, args.limit)

    print('\n=== %s (n=%d, TTA=%s) ===' % (args.task, res['n'], args.tta), flush=True)
    for i, (session, ct) in enumerate(specs):
        line = '  单模型[%s]: %.4f' % (session, res['single'][i])
        if res['tta']:
            line += '   +TTA: %.4f' % res['tta'][i]
        print(line, flush=True)
    print('  ★ 集成(%d 模型%s) @0.5: %.4f  (相对单模型 %+.4f)' % (
        len(models), '+TTA' if args.tta != 'none' else '', res['ens'],
        res['ens'] - res['single'][0]), flush=True)

    if args.thresh_search:
        if val_res is not None:
            print('  ★★ 集成 + TTA + 阈值(验证集选定 %.2f): %.4f  (相对 @0.5 %+.4f)' % (
                best_t, res['ens_thr'][best_t], res['ens_thr'][best_t] - res['ens']), flush=True)
            print('  --- 以下为诊断用：测试集阈值扫描，禁止写入论文（测试集调参）---', flush=True)
        for t in THRS:
            mark = ' ←验证集选定' if (val_res is not None and t == best_t) else ''
            print('    test thr=%.2f: %.4f%s' % (t, res['ens_thr'][t], mark), flush=True)
    print('ENSEMBLE_DONE', flush=True)


if __name__ == '__main__':
    main()
