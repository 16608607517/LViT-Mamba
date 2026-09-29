# -*- coding: utf-8 -*-
"""评估单个 text_cond 检查点（指定 session 与 cond_target）。

用法: python eval_textcond_ckpt.py <session> <cond_target> [gpu]
输出: EVAL_RESULT ... TEST_DICE <数值>
"""
import sys
import os
import torch

gpu = sys.argv[3] if len(sys.argv) > 3 else '0'
task = sys.argv[4] if len(sys.argv) > 4 else 'Covid19'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import Config as config                                          # noqa: E402
from Load_Dataset import ValGenerator, ImageToImage2D            # noqa: E402
from torch.utils.data import DataLoader                          # noqa: E402
from nets.LVMamba import LVMamba                                 # noqa: E402

sess, ct = sys.argv[1], sys.argv[2]
ckpt_path = './%s/LVMamba/%s/models/best_model-LVMamba.pth.tar' % (task, sess)

m = LVMamba(config.get_CTranS_config(), n_channels=3, n_classes=1, backbone='mamba',
            use_text=True, mamba_depth=1, scan_mode='bidirectional', text_gate=False,
            text_cond=(ct != 'none'), cond_target=ct if ct != 'none' else 'dtBC').cuda()
ck = torch.load(ckpt_path, map_location='cpu')
sd = {k[7:] if k.startswith('module.') else k: v for k, v in ck['state_dict'].items()}
missing, unexpected = m.load_state_dict(sd, strict=False)
n_mod_missing = len([k for k in missing if '.t_dt' in k or '.t_B' in k or '.t_C' in k])
print('session=%s cond_target=%s | loaded %d/%d | 调制头missing=%d | text_summary_missing=%d' % (
    sess, ct, len(sd) - len(unexpected), len(sd), n_mod_missing,
    len([k for k in missing if 'text_summary' in k])), flush=True)

m.eval()
tf = ValGenerator(output_size=[config.img_size, config.img_size])
ds = ImageToImage2D('./datasets/%s/Test_Folder/' % task, task, {}, tf,
                    image_size=config.img_size)
loader = DataLoader(ds, batch_size=1, shuffle=False)

tot, n = 0.0, 0
with torch.no_grad():
    for b, names in loader:
        img, lab, txt = b['image'].cuda(), b['label'].cuda().float(), b['text'].cuda()
        pred = (m(img, txt) > 0.5).float()
        p, g = pred.reshape(-1), lab.reshape(-1)
        tot += (2 * (p * g).sum() / (p.sum() + g.sum() + 1e-5)).item()
        n += 1
print('EVAL_RESULT task=%s session=%s cond_target=%s TEST_DICE %.6f' % (task, sess, ct, tot / n), flush=True)
