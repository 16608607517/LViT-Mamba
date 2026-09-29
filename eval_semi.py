# -*- coding: utf-8 -*-
"""半监督模型测试集评估: 加载 semi_results/*/best_model.pth 跑测试集, 输出 Dice/IoU。
用法: python eval_semi.py --task Covid19 --ratio 0.25 --backbone mamba --use_text True
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")
import torch
import numpy as np
import Config as config
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
from nets.LViT import LViT
from nets.LVMamba import LVMamba
from utils import read_text
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='Covid19')
    p.add_argument('--ratio', type=float, default=0.25)
    p.add_argument('--backbone', default='mamba', choices=['mamba', 'transformer'])
    p.add_argument('--use_text', type=lambda x: x.lower() == 'true', default=True)
    p.add_argument('--save_vis', action='store_true', help='保存预测图(默认只算指标)')
    p.add_argument('--tag_suffix', default='', help='实验变体后缀（与训练时一致）')
    p.add_argument('--cond_target', default=None,
                   help='显式指定调制目标(与训练一致): none=无调制, B/C/dtBC=调制; 缺省则读 Config')
    args = p.parse_args()

    tag = f'{args.task}_r{int(args.ratio * 100)}_{args.backbone}_text{str(args.use_text)[0]}'
    if getattr(args, 'tag_suffix', ''):
        tag = tag + '_' + args.tag_suffix
    ckpt = f'./semi_results/{tag}/best_model.pth'
    assert os.path.exists(ckpt), f'checkpoint not found: {ckpt}'

    # 模型构造与 train_semi.py 保持一致
    config_vit = config.get_CTranS_config()
    if args.cond_target is not None:          # 显式指定(与训练时一致) → 不依赖 Config 当前状态
        _tc = args.cond_target != 'none'
        _ct = args.cond_target if _tc else 'dtBC'
    else:
        _tc = getattr(config, 'text_cond', False)
        _ct = getattr(config, 'cond_target', 'dtBC')
    if args.backbone == 'mamba':
        model = LVMamba(config_vit, n_channels=3, n_classes=1,
                        backbone='mamba', use_text=args.use_text,
                        mamba_depth=1, scan_mode='bidirectional', text_gate=False,
                        text_cond=_tc, cond_target=_ct)
    else:
        model = LViT(config_vit, n_channels=3, n_classes=1)
    _sd = torch.load(ckpt, map_location='cpu')
    _sd = {k[7:] if k.startswith('module.') else k: v for k, v in _sd.items()}
    _missing, _unexpected = model.load_state_dict(_sd, strict=False)
    _real_missing = [k for k in _missing
                     if not (k.endswith('.total_ops') or k.endswith('.total_params'))]
    # strict=False 会静默吞掉加载失败（曾导致 Dice=0 的假结果）→ 这里强制校验
    assert not _real_missing, '权重未完全加载 (%d 个缺失): %s' % (len(_real_missing), _real_missing[:8])
    print('[load] %s: %d/%d 权重加载成功' % (ckpt, len(_sd) - len(_unexpected), len(_sd)))
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    # 测试集（路径按 task 独立构造，不依赖 Config.task_name）
    if args.task == 'MosMedDataPlus':
        test_dir = './datasets/MosMedDataPlus/Test_Folder/'
        test_num = 273
    elif args.task == 'MoNuSeg':
        test_dir = './datasets/MoNuSeg/Test_Folder/'
        test_num = 14
    else:
        test_dir = f'./datasets/{args.task}/Test_Folder/'
        test_num = 2113
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_text = read_text(test_dir + 'Test_text.xlsx')
    test_ds = ImageToImage2D(test_dir, args.task, test_text, tf_test, image_size=config.img_size)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    vis_path = f'./{args.task}_visualize_test_semi/'
    if args.save_vis:
        os.makedirs(vis_path, exist_ok=True)

    dice_sum = iou_sum = 0.0
    with torch.no_grad(), tqdm(total=test_num, desc=tag, unit='img', ncols=70) as pbar:
        for sampled_batch, names in test_loader:
            img = sampled_batch['image'].cuda()
            lab = sampled_batch['label'].data.numpy().reshape(config.img_size, config.img_size)
            if args.use_text:
                text = sampled_batch['text'].cuda()
                if text.ndim == 3 and text.shape[1] > 10:  # 与训练截断一致
                    text = text[:, :10, :]
                out = model(img, text)
            else:
                out = model(img, None)
            pred = (out > 0.5).float()[0, 0].cpu().numpy()
            dice_sum += 2 * (lab * pred).sum() / (lab.sum() + pred.sum() + 1e-5)
            iou_sum += (lab * pred).sum() / ((lab + pred > 0).sum() + 1e-5)
            if args.save_vis:
                import cv2
                cv2.imwrite(vis_path + str(names[0]), pred * 255)
            pbar.update()
    print(f'[{tag}] test Dice={dice_sum / test_num:.4f}  IoU={iou_sum / test_num:.4f}')


if __name__ == '__main__':
    main()
