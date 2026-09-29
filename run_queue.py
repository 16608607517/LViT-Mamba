# -*- coding: utf-8 -*-
"""batch24 全监督实验队列: 自动改Config→训练→测试集评估→记录, 串行执行。
用法: python run_queue.py [起始序号]  (默认从0开始; E1完成后用 python run_queue.py 跑E2之后)
"""
import subprocess
import os
import re
import sys
import csv

BASE = os.path.expanduser('~/LViT/LViT-main')
LOG = os.path.expanduser('~/LViT/logs')
RESULTS = os.path.expanduser('~/LViT/results_batch24.csv')
# 优先用 wrapper（修复 LD_LIBRARY_PATH），不存在则回退直接调用
PY = os.path.expanduser('~/LViT/run_py.sh')
if not os.path.exists(PY):
    PY = os.path.expanduser('~/miniconda3/envs/mamba_env/bin/python')

# (name, task, backbone, use_text, lr, scan_mode, text_gate)
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


def patch_config(task=None, backbone=None, use_text=None, lr=None, scan=None, gate=None,
                 gpus=None, batch=None):
    path = os.path.join(BASE, 'Config.py')
    with open(path) as f:
        s = f.read()

    def rep(pattern, val):
        nonlocal s
        s, n = re.subn(pattern, val, s, flags=re.M)
        assert n >= 1, 'pattern not found: ' + pattern
    if gpus:
        rep(r'^os\.environ\["CUDA_VISIBLE_DEVICES"\] = .*$',
            'os.environ["CUDA_VISIBLE_DEVICES"] = "%s"' % gpus)
    if batch:
        rep(r'^batch_size = .*$', 'batch_size = %s' % batch)
    if task:
        rep(r'^task_name = .*$', "task_name = '%s'" % task)
    if backbone:
        rep(r'^backbone = .*$', "backbone = '%s'" % backbone)
    if use_text is not None:
        rep(r'^use_text = .*$', 'use_text = %s' % use_text)
    if lr:
        rep(r'^learning_rate = .*$', 'learning_rate = %s' % lr)
    if scan:
        rep(r'^scan_mode = .*$', "scan_mode = '%s'" % scan)
    if gate is not None:
        rep(r'^text_gate = .*$', 'text_gate = %s' % gate)
    with open(path, 'w') as f:
        f.write(s)


def latest_session(task):
    d = os.path.join(BASE, task, 'LVMamba')
    return sorted(os.listdir(d))[-1]


def evaluate(task):
    """设 test_session 为最新 session 并跑 test_model.py, 返回 (session, dice)"""
    path = os.path.join(BASE, 'Config.py')
    with open(path) as f:
        s = f.read()
    sess = latest_session(task)
    s = re.sub(r'^test_session = .*$', 'test_session = "%s"' % sess, s, flags=re.M)
    with open(path, 'w') as f:
        f.write(s)
    env = os.environ.copy()
    gpus = os.environ.get('RQ_GPUS')  # 评估用同一张卡(单卡分担时=1)
    if gpus:
        env['LVIT_GPU'] = gpus.split(',')[0]
    r = subprocess.run([PY, '-u', 'test_model.py'], capture_output=True,
                       text=True, cwd=BASE, timeout=7200, env=env)
    m = re.findall(r'dice_pred (\d\.\d+)', r.stdout)
    dice = float(m[-1]) if m else -1.0
    with open(os.path.join(LOG, 'eval_' + sess + '.txt'), 'w') as f:
        f.write(r.stdout[-3000:])
    return sess, dice


def main():
    # eval-only <name> <task> <backbone> <use_text> <lr> <scan> <gate>: 只评估最新session并记入CSV
    if len(sys.argv) >= 8 and sys.argv[1] == 'eval-only':
        name, task, bb, ut, lr, scan, gate = sys.argv[2:9]
        sess, dice = evaluate(task)
        with open(RESULTS, 'a', newline='') as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(['name', 'task', 'backbone', 'use_text', 'lr', 'scan',
                            'gate', 'session', 'test_dice', 'train_ok'])
            w.writerow([name, task, bb, ut, lr, scan, gate, sess, dice, True])
        print('EVAL_DONE dice=%.4f' % dice, flush=True)
        return
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else len(EXPS)
    for i, (name, task, bb, ut, lr, scan, gate) in enumerate(EXPS):
        if i < start:
            continue
        if i >= end:
            break
        print('=== %s ===' % name, flush=True)
        # 单卡分担GPU时: RQ_GPUS=1 RQ_BATCH=12 RQ_ACC=2 (12×2=24 等效论文batch)
        gpus = os.environ.get('RQ_GPUS')
        batch = os.environ.get('RQ_BATCH')
        acc = os.environ.get('RQ_ACC', '1')
        patch_config(task=task, backbone=bb, use_text=ut, lr=lr, scan=scan, gate=gate,
                     gpus=gpus, batch=batch)
        cmd = [PY, '-u', 'train_model.py']
        if acc != '1':
            cmd += ['--acc_steps', acc]
        log = os.path.join(LOG, name + '.log')
        with open(log, 'w') as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=BASE)
        ok = r.returncode == 0
        sess, dice = (None, -1.0)
        if ok:
            try:
                sess, dice = evaluate(task)
            except Exception as e:
                print('eval failed: %s' % e, flush=True)
        with open(RESULTS, 'a', newline='') as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(['name', 'task', 'backbone', 'use_text', 'lr', 'scan',
                            'gate', 'session', 'test_dice', 'train_ok'])
            w.writerow([name, task, bb, ut, lr, scan, gate, sess, dice, ok])
        print('%s: train_ok=%s dice=%.4f' % (name, ok, dice), flush=True)
    print('ALL_DONE', flush=True)
    # 自动衔接半监督队列 (run_semi_queue.py)；指定了 end 说明只要部分实验, 不衔接
    if len(sys.argv) <= 2:
        semi = os.path.join(BASE, 'run_semi_queue.py')
        if os.path.exists(semi):
            print('=== auto-start semi-sup queue ===', flush=True)
            with open(os.path.join(LOG, 'semi_queue.log'), 'w') as f:
                subprocess.run([PY, '-u', semi], stdout=f, stderr=subprocess.STDOUT, cwd=BASE)


if __name__ == '__main__':
    main()
