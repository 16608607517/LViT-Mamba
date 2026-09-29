# -*- coding: utf-8 -*-
"""
论文原版半监督训练（LViT, IEEE TMI 2024）:
- EPI (Exponential Pseudo label Iteration): EMA (β=0.99) 模型生成稳定伪标签
- LV Loss: 利用对比标签的文本相似度引导无标注数据 (α=0.1)

损失:
  有标注: L_sup = WeightedDiceBCE(pred, mask)
  无标注: L_unsup = WeightedDiceBCE(pred, pseudo_label) + 0.1 * L_LV

用法:
  python train_semi.py --ratio 0.1 --backbone mamba --use_text True
"""
import argparse
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import Config as config
from Load_Dataset import ImageToImage2D, RandomGenerator, ValGenerator
from nets.LVMamba import LVMamba
from nets.LViT import LViT
from semi_split import split_supervised
from semi_dataset import build_semi_datasets
from semi_loss import ContrastiveLVLoss
from utils import read_text, WeightedDiceBCE
from torch.utils.data import DataLoader

MATCH_PATH = './semi_results/matches/{task}_{split}_contrast.pt'

warnings.filterwarnings('ignore')


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='MosMedDataPlus')
    p.add_argument('--ratio', type=float, default=0.1, help='labeled ratio')
    p.add_argument('--backbone', default='mamba', choices=['mamba', 'transformer'])
    p.add_argument('--use_text', type=lambda x: x.lower() == 'true', default=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--ema_beta', type=float, default=0.99)
    p.add_argument('--alpha', type=float, default=0.1, help='LV Loss weight')
    p.add_argument('--seed', type=int, default=666)
    p.add_argument('--acc_steps', type=int, default=6,
                   help='梯度累积步数（batch4×6=有效batch24，论文QaTa最优）')
    p.add_argument('--cons_lambda', type=float, default=0.0,
                   help='调制自一致性正则权重（>0 时启用：让带调制与无文本的预测保持一致）')
    p.add_argument('--tag_suffix', default='',
                   help='实验变体后缀（避免并行实验保存到同一模型路径）')
    # === 第二轮: 参数级正则（针对低标注下调制头过拟合）===
    p.add_argument('--mod_wd', type=float, default=0.0,
                   help='调制头单独分组的 weight decay（>0 启用）')
    p.add_argument('--mod_lr_scale', type=float, default=1.0,
                   help='调制头学习率倍率（<1 让调制变化更慢）')
    p.add_argument('--mod_warmup', type=int, default=0,
                   help='前 N 个 epoch 冻结调制头（零初始化 = 等价基线，之后逐步启用）')
    p.add_argument('--clip_grad', type=float, default=1.0,
                   help='梯度裁剪范数上限（0=关闭）；防止调制注入 SSM 时数值发散')
    p.add_argument('--split_strategy', default='random',
                   choices=['random', 'area_strat', 'text_strat'],
                   help='标注子集划分策略: random=随机(默认) / area_strat=按病灶面积分层 / '
                        'text_strat=按文本语义(PCA)分层')
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args):
    base = f'./datasets/{args.task}/Train_Folder/'
    # Covid19 的 train 文本合并为 Train_Val_text.xlsx
    text_file = 'Train_Val_text.xlsx' if args.task in ['Covid19', 'QaTaCOV19'] else 'Train_text.xlsx'
    train_text = read_text(base + text_file)
    pre_path = base + 'precomputed_bert.pt'
    img_dir = base + 'img/'
    labeled_list, unlabeled_list = split_supervised(img_dir, args.ratio, args.seed,
                                                    strategy=getattr(args, 'split_strategy', 'random'),
                                                    task=args.task)
    labeled_ds, unlabeled_ds = build_semi_datasets(
        base, args.task, train_text, labeled_list, unlabeled_list,
        precomputed_bert_path=pre_path, image_size=config.img_size)
    labeled_loader = DataLoader(labeled_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    return labeled_loader, unlabeled_loader


def build_model(args):
    config_vit = config.get_CTranS_config()
    if args.backbone == 'mamba':
        model = LVMamba(config_vit, n_channels=3, n_classes=1,
                        backbone='mamba', use_text=args.use_text,
                        mamba_depth=1, scan_mode='bidirectional', text_gate=False,
                        text_cond=getattr(config, 'text_cond', False),
                        cond_target=getattr(config, 'cond_target', 'dtBC'))
    else:
        model = LViT(config_vit, n_channels=3, n_classes=1)
    return model.cuda()


def update_ema(ema_model, model, beta):
    """EMA 权重更新：ema = beta*ema + (1-beta)*model"""
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(beta).add_(p.data, alpha=1 - beta)
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.data.copy_(b.data)


def train_one_epoch(labeled_loader, unlabeled_loader, model, ema_model,
                    criterion, lv_loss, optimizer, args, contrast_idx_map=None):
    model.train()
    optimizer.zero_grad()  # 梯度累积模式下，进入 epoch 前清零
    l_sum, s_sum, p_sum, lv_sum = 0, 0, 0, 0
    n_batches = 0
    unlabeled_iter = iter(unlabeled_loader)

    for i, (batch_l, _) in enumerate(labeled_loader):
        img_l = batch_l['image'].cuda()
        mask_l = batch_l['label'].cuda()
        text_l = batch_l['text'].cuda()
        if text_l.ndim == 3 and text_l.shape[1] > 10:
            text_l = text_l[:, :10, :]

        # 取一个无标注 batch（round-robin）
        try:
            batch_u, names_u = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            batch_u, names_u = next(unlabeled_iter)
        img_u = batch_u['image'].cuda()
        text_u = batch_u['text'].cuda()
        if text_u.ndim == 3 and text_u.shape[1] > 10:
            text_u = text_u[:, :10, :]

        # 预计算对比标签索引
        contrast_idx = None
        if contrast_idx_map is not None and args.use_text:
            idxs = []
            for name in names_u:
                name = os.path.basename(name)
                if name in contrast_idx_map:
                    idxs.append(contrast_idx_map[name])
            if len(idxs) == len(names_u):
                contrast_idx = torch.tensor(idxs, dtype=torch.long).cuda()

        acc_steps = getattr(args, 'acc_steps', 1)
        clip_grad = getattr(args, 'clip_grad', 0.0)

        # ===== 有标注：监督损失 =====
        pred_l = model(img_l, text_l if args.use_text else None)
        loss_sup = criterion(pred_l, mask_l.float())

        # ===== 无标注：EPI 伪标签 + LV Loss =====
        with torch.no_grad():
            # EMA 模型生成稳定伪标签（EPI 思想）
            pseudo = ema_model(img_u, text_u if args.use_text else None)
        pred_u = model(img_u, text_u if args.use_text else None)
        loss_pseudo = criterion(pred_u, pseudo.detach())
        loss_lv = lv_loss(pred_u, text_u, contrast_idx) if args.use_text else torch.zeros(1).cuda()

        # ===== ★ 调制自一致性正则 =====
        # 让"带文本调制"的预测与"无文本参考"的预测保持一致：
        # 调制只应补充语义信息, 不应改变模型的预测习惯 → 抑制半监督下的过度约束
        # 参考路径用 no_grad(省显存 + 避免模型"关掉调制"来取巧)
        cons_lambda = getattr(args, 'cons_lambda', 0.0)
        if cons_lambda > 0 and args.use_text:
            with torch.no_grad():
                pred_u_notext = model(img_u, None)
            loss_cons = F.l1_loss(torch.sigmoid(pred_u),
                                  torch.sigmoid(pred_u_notext).detach())
        else:
            loss_cons = torch.zeros(1).cuda()

        loss = (loss_sup + loss_pseudo + args.alpha * loss_lv
                + cons_lambda * loss_cons) / acc_steps
        loss.backward()

        # 梯度裁剪：防止调制注入 SSM 时数值发散（半监督下曾多次触发 CUDA device-side assert）
        if (i + 1) % acc_steps == 0 and clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        # 梯度累积：每 acc_steps 步更新一次（等效增大 batch_size）
        if (i + 1) % acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        l_sum += loss.item() * acc_steps
        s_sum += loss_sup.item()
        p_sum += loss_pseudo.item()
        lv_sum += loss_lv.item()
        n_batches += 1

    return l_sum / n_batches, s_sum / n_batches, p_sum / n_batches, lv_sum / n_batches


def validate(val_loader, model, criterion):
    model.eval()
    d_sum, n = 0, 0
    with torch.no_grad():
        for batch, _ in val_loader:
            img = batch['image'].cuda()
            mask = batch['label'].cuda()
            text = batch['text'].cuda()
            if text.shape[1] > 10:
                text = text[:, :10, :]
            preds = model(img, text)
            d_sum += criterion._show_dice(preds, mask.float())
            n += 1
    return d_sum / n


def main():
    args = get_args()
    set_seed(args.seed)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    tag = f'{args.task}_r{int(args.ratio*100)}_{args.backbone}_text{str(args.use_text)[0]}'
    suffix = getattr(args, 'tag_suffix', '')       # 实验变体后缀(避免并行实验互相覆盖模型)
    if suffix:
        tag = tag + '_' + suffix
    print(f'=== [paper semi-sup] {tag} ===')

    labeled_loader, unlabeled_loader = build_loaders(args)
    # Covid19 的 val 文本在 Train_Val_text.xlsx（val 是 train 划分的子集）
    if args.task in ['Covid19', 'QaTaCOV19']:
        val_text = read_text(f'./datasets/{args.task}/Train_Folder/Train_Val_text.xlsx')
    else:
        val_text = read_text(f'./datasets/{args.task}/Val_Folder/Val_text.xlsx')
    val_ds = ImageToImage2D(f'./datasets/{args.task}/Val_Folder/', args.task, val_text,
                            ValGenerator(output_size=[config.img_size, config.img_size]),
                            image_size=config.img_size)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    model = build_model(args)
    ema_model = build_model(args)
    ema_model.load_state_dict(model.state_dict())

    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    lv_loss = ContrastiveLVLoss()
    # 调制头单独分组：可施加更强 weight decay / 更小学习率（针对低标注下的过拟合）
    MOD_KEYS = ('.t_dt', '.t_B', '.t_C', 'text_summary')
    mod_params, base_params = [], []
    for n, p in model.named_parameters():
        (mod_params if any(k in n for k in MOD_KEYS) else base_params).append(p)
    groups = [{'params': base_params, 'lr': args.lr}]
    if mod_params:
        groups.append({'params': mod_params, 'lr': args.lr * args.mod_lr_scale,
                       'weight_decay': args.mod_wd})
        print('[mod] %d 组调制参数 | lr_scale=%.3f | wd=%.4f | warmup=%d'
              % (len(mod_params), args.mod_lr_scale, args.mod_wd, args.mod_warmup))
    optimizer = torch.optim.Adam(groups, lr=args.lr)

    # 加载预计算对比标签索引（混合匹配）
    contrast_idx_map = None
    if args.use_text:
        match_path = MATCH_PATH.format(task=args.task, split='Train')
        if os.path.exists(match_path):
            contrast_idx_map = torch.load(match_path, map_location='cpu')
            print(f'[match] loaded {len(contrast_idx_map)} contrast indices from {match_path}')
        else:
            print(f'[match] WARNING: {match_path} not found, falling back to TextSim')

    print(f'labeled={len(labeled_loader.dataset)}, unlabeled={len(unlabeled_loader.dataset)}, '
          f'ema_beta={args.ema_beta}, alpha={args.alpha}')

    best_dice = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        # 调制头 warmup: 前 N 个 epoch 冻结（零初始化 ≈ 基线），让基线表示先学好再加文本调制
        if args.mod_warmup > 0:
            freeze = epoch < args.mod_warmup
            if freeze != getattr(model, '_mod_frozen', None):
                for n, p in model.named_parameters():
                    if any(k in n for k in MOD_KEYS):
                        p.requires_grad = not freeze
                model._mod_frozen = freeze
                print('[mod] epoch %d: 调制头 %s' % (epoch + 1, '冻结' if freeze else '启用'))
        l, s, p, lv = train_one_epoch(labeled_loader, unlabeled_loader, model, ema_model,
                                      criterion, lv_loss, optimizer, args, contrast_idx_map)
        # 每 epoch 更新一次 EMA
        update_ema(ema_model, model, args.ema_beta)
        val_dice = validate(val_loader, model, criterion)
        t = time.time() - t0

        print(f'Epoch {epoch+1:3d}/{args.epochs} ({t:.0f}s): total={l:.4f} '
              f'sup={s:.4f} pseudo={p:.4f} lv={lv:.4f} | Val Dice={val_dice:.4f}')

        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch + 1
            os.makedirs(f'./semi_results/{tag}/', exist_ok=True)
            torch.save(model.state_dict(), f'./semi_results/{tag}/best_model.pth')
            print(f'  [save] best Val Dice={val_dice:.4f} (epoch {best_epoch})')

    print(f'=== Done: {tag}, best Val Dice={best_dice:.4f} (epoch {best_epoch}) ===')


if __name__ == '__main__':
    main()
