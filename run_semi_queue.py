# -*- coding: utf-8 -*-
"""batch24 半监督队列: 8组自动串行 (train_semi → eval_semi), 结果记入 results_semi_batch24.csv。
用法: python run_semi_queue.py [起始序号]
"""
import subprocess
import os
import re
import sys
import csv

BASE = os.path.expanduser('~/LViT/LViT-main')
LOG = os.path.expanduser('~/LViT/logs')
RESULTS = os.path.expanduser('~/LViT/results_semi_batch24.csv')
# 优先用 wrapper（修复 LD_LIBRARY_PATH），不存在则回退直接调用
PY = os.path.expanduser('~/LViT/run_py.sh')
if not os.path.exists(PY):
    PY = os.path.expanduser('~/miniconda3/envs/mamba_env/bin/python')

# (name, ratio, backbone, use_text)
EXPS = [
    ('s1_trans_t25', 0.25, 'transformer', True),
    ('s2_trans_f25', 0.25, 'transformer', False),
    ('s3_mamba_t25', 0.25, 'mamba', True),
    ('s4_mamba_f25', 0.25, 'mamba', False),
    ('s5_trans_t50', 0.50, 'transformer', True),
    ('s6_trans_f50', 0.50, 'transformer', False),
    ('s7_mamba_t50', 0.50, 'mamba', True),
    ('s8_mamba_f50', 0.50, 'mamba', False),
]


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for i, (name, ratio, bb, ut) in enumerate(EXPS):
        if i < start:
            continue
        print('=== %s ===' % name, flush=True)
        cmd = [PY, '-u', 'train_semi.py', '--task', 'Covid19', '--ratio', str(ratio),
               '--backbone', bb, '--use_text', str(ut), '--epochs', '200',
               '--batch_size', '24', '--lr', '1e-3', '--acc_steps', '1']
        # batch24 直接跑, 无需梯度累积; lr 1e-3 与论文 batch24 配套
        log = os.path.join(LOG, name + '.log')
        with open(log, 'w') as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=BASE)
        ok = r.returncode == 0
        dice = -1.0
        if ok:
            ev = subprocess.run([PY, '-u', 'eval_semi.py', '--task', 'Covid19',
                                 '--ratio', str(ratio), '--backbone', bb,
                                 '--use_text', str(ut)],
                                capture_output=True, text=True, cwd=BASE, timeout=7200)
            m = re.search(r'test Dice=([\d.]+)', ev.stdout)
            dice = float(m.group(1)) if m else -1.0
        with open(RESULTS, 'a', newline='') as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(['name', 'ratio', 'backbone', 'use_text', 'test_dice', 'train_ok'])
            w.writerow([name, ratio, bb, ut, dice, ok])
        print('%s: ok=%s dice=%.4f' % (name, ok, dice), flush=True)
    print('SEMI_ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
