# -*- coding: utf-8 -*-
"""31.26 双卡并行队列: 两张卡各跑一组实验(单卡 batch12 x acc2 = 等效 batch24)。

用法: python run_parallel.py <idx...>
  索引对应 run_queue.py 的 EXPS: 0=E2 1=E3 2=E4 3=E5 4=E6 5=E7 6=E8 7=E9 8=E10 9=E11
  例: python run_parallel.py 4 5   → E6(GPU0) + E7(GPU1) 并行

流程: 逐个 patch Config 并启动(等 session 目录出现确认归属, 间隔>60s 保证 session 名不撞)
      → 等全部训练结束 → 逐个评估写 CSV
"""
import subprocess
import os
import re
import sys
import csv
import time

BASE = os.path.expanduser('~/LViT/LViT-main')
LOG = os.path.expanduser('~/LViT/logs')
RESULTS = os.path.expanduser('~/LViT/results_batch24.csv')
# 优先用 wrapper(修复 LD_LIBRARY_PATH), 不存在则直接调用 env python
PY = os.path.expanduser('~/LViT/run_py.sh')
if not os.path.exists(PY):
    PY = os.path.expanduser('~/miniconda3/envs/mamba_env/bin/python')

# 与 run_queue.py 保持一致的实验表
EXPS = [
    ('e2_mamba_text_covid19',   'Covid19',        'mamba',       True,  '3e-4', 'bidirectional', False),
    ('e3_trans_notext_covid19', 'Covid19',        'transformer', False, '3e-4', 'bidirectional', False),
    ('e4_mamba_notext_covid19', 'Covid19',        'mamba',       False, '3e-4', 'bidirectional', False),
    ('e5_trans_text_mosmed',    'MosMedDataPlus', 'transformer', True,  '1e-3', 'bidirectional', False),
    ('e6_mamba_text_mosmed',    'MosMedDataPlus', 'mamba',       True,  '1e-3', 'bidirectional', False),
    ('e7_trans_notext_mosmed',  'MosMedDataPlus', 'transformer', False, '1e-3', 'bidirectional', False),
    ('e8_mamba_notext_mosmed',  'MosMedDataPlus', 'mamba',       False, '1e-3', 'bidirectional', False),
    ('e9_cross_mamba_covid19',  'Covid19',        'mamba',       True,  '3e-4', 'cross',         False),
    ('e10_cross_mamba_mosmed',  'MosMedDataPlus', 'mamba',       True,  '1e-3', 'cross',         False),
    ('e11_gate_mamba_covid19',  'Covid19',        'mamba',       True,  '3e-4', 'bidirectional', True),
]
BATCH = 12   # 单卡 12 张
ACC = 2      # 梯度累积 2 步 → 等效 batch 24


def patch_config(task, backbone, use_text, lr, scan, gate, gpus, test_session=None):
    path = os.path.join(BASE, 'Config.py')
    with open(path) as f:
        s = f.read()

    def rep(pattern, val):
        nonlocal s
        s, n = re.subn(pattern, val, s, flags=re.M)
        assert n >= 1, 'pattern not found: ' + pattern
    rep(r'^os\.environ\["CUDA_VISIBLE_DEVICES"\] = .*$',
        'os.environ["CUDA_VISIBLE_DEVICES"] = "%s"' % gpus)
    rep(r'^batch_size = .*$', 'batch_size = %s' % BATCH)
    rep(r'^task_name = .*$', "task_name = '%s'" % task)
    rep(r'^backbone = .*$', "backbone = '%s'" % backbone)
    rep(r'^use_text = .*$', 'use_text = %s' % use_text)
    rep(r'^learning_rate = .*$', 'learning_rate = %s' % lr)
    rep(r'^scan_mode = .*$', "scan_mode = '%s'" % scan)
    rep(r'^text_gate = .*$', 'text_gate = %s' % gate)
    if test_session:
        rep(r'^test_session = .*$', 'test_session = "%s"' % test_session)
    with open(path, 'w') as f:
        f.write(s)


def sessions(task):
    d = os.path.join(BASE, task, 'LVMamba')
    return set(os.listdir(d)) if os.path.isdir(d) else set()


def main():
    idxs = [int(x) for x in sys.argv[1:]] or [4, 5]
    jobs = []
    prev_sess = None
    for k, i in enumerate(idxs):
        name, task, bb, ut, lr, scan, gate = EXPS[i]
        gpu = str(k % 2)                      # 交替分卡: 0,1,0,1...
        before = sessions(task)
        patch_config(task, bb, ut, lr, scan, gate, gpus=gpu)
        if k > 0:
            time.sleep(75)                    # 保证 session 名跨分钟(不与上一组撞目录)
        log = open(os.path.join(LOG, name + '.log'), 'w')
        p = subprocess.Popen([PY, '-u', 'train_model.py', '--acc_steps', str(ACC)],
                             stdout=log, stderr=subprocess.STDOUT, cwd=BASE)
        # 等新 session 目录出现, 确认该组的 session 归属
        sess = None
        for _ in range(180):                  # 最多等 15 分钟
            new = sessions(task) - before
            if new:
                sess = sorted(new)[-1]
                break
            time.sleep(5)
        assert sess != prev_sess, 'session 名冲突(两组写同一目录)! %s' % sess
        prev_sess = sess
        print('%s: gpu=%s pid=%s session=%s' % (name, gpu, p.pid, sess), flush=True)
        jobs.append([name, task, bb, ut, lr, scan, gate, gpu, sess, p])

    for j in jobs:                            # 等全部训练结束
        j[9].wait()
    print('ALL_TRAIN_DONE', flush=True)

    for name, task, bb, ut, lr, scan, gate, gpu, sess, p in jobs:
        ok = p.returncode == 0
        dice = -1.0
        if ok and sess:
            patch_config(task, bb, ut, lr, scan, gate, gpus=gpu, test_session=sess)
            env = os.environ.copy()
            env['LVIT_GPU'] = gpu
            r = subprocess.run([PY, '-u', 'test_model.py'], capture_output=True,
                               text=True, cwd=BASE, timeout=7200, env=env)
            m = re.findall(r'dice_pred (\d\.\d+)', r.stdout)
            dice = float(m[-1]) if m else -1.0
            with open(os.path.join(LOG, 'eval_' + sess + '.txt'), 'w') as f:
                f.write(r.stdout[-3000:])
        with open(RESULTS, 'a', newline='') as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(['name', 'task', 'backbone', 'use_text', 'lr', 'scan',
                            'gate', 'session', 'test_dice', 'train_ok'])
            w.writerow([name, task, bb, ut, lr, scan, gate, sess, dice, ok])
        print('%s: train_ok=%s dice=%.4f' % (name, ok, dice), flush=True)
    print('PARALLEL_ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
