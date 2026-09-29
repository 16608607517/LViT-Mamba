# -*- coding: utf-8 -*-
"""半监督双卡并行队列: 每卡一组, 单卡 batch12 x acc2 = 等效 batch24。

用法: python run_semi_parallel.py <idx...>
  半监督组顺序(与 run_semi_queue.py 一致):
    0=s1_trans_t25  1=s2_trans_f25  2=s3_mamba_t25  3=s4_mamba_f25
    4=s5_trans_t50  5=s6_trans_f50  6=s7_mamba_t50  7=s8_mamba_f50
  例: python run_semi_parallel.py 2 3   → s3(GPU0) + s4(GPU1) 并行

流程: 按卡轮询启动(每组前 patch Config 的 CUDA_VISIBLE_DEVICES) → 等全部训练完 → 逐个评估写 CSV
"""
import os
import re
import sys
import csv
import time
import subprocess

BASE = os.path.expanduser('~/LViT/LViT-main')
LOG = os.path.expanduser('~/LViT/logs')
RESULTS = os.path.expanduser('~/LViT/results_semi_batch24.csv')
PY = os.path.expanduser('~/LViT/run_py.sh')
if not os.path.exists(PY):
    PY = os.path.expanduser('~/miniconda3/envs/mamba_env/bin/python')

BATCH, ACC, EPOCHS, LR = 12, 2, 200, '1e-3'
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


def patch_gpu(gpu):
    path = os.path.join(BASE, 'Config.py')
    with open(path) as f:
        s = f.read()
    s, n = re.subn(r'^os\.environ\["CUDA_VISIBLE_DEVICES"\] = .*$',
                   'os.environ["CUDA_VISIBLE_DEVICES"] = "%s"' % gpu, s, flags=re.M)
    assert n >= 1, 'CUDA_VISIBLE_DEVICES 行未找到'
    with open(path, 'w') as f:
        f.write(s)


def gpu_mem(idx):
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used',
                          '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    for line in out.strip().split('\n'):
        i, m = line.split(',')
        if i.strip() == str(idx):
            return int(m)
    return -1


def launch(name, ratio, bb, ut, gpu):
    """启动一组训练, 并确认它真的落在指定卡上(防止 Config 竞争导致串卡)。"""
    # 先等所有已存在的任务都稳定(其他卡显存已分配), 再 patch+启动本组
    patch_gpu(gpu)
    time.sleep(75)
    before = gpu_mem(gpu)
    log = open(os.path.join(LOG, name + '.log'), 'w')
    cmd = [PY, '-u', 'train_semi.py', '--task', 'Covid19', '--ratio', str(ratio),
           '--backbone', bb, '--use_text', str(ut), '--epochs', str(EPOCHS),
           '--batch_size', str(BATCH), '--lr', LR, '--acc_steps', str(ACC)]
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=BASE)
    # 轮询确认目标卡显存上涨(证明确实读到了正确的 CUDA_VISIBLE_DEVICES)
    ok = False
    for _ in range(60):                      # 最多等 5 分钟
        time.sleep(5)
        if p.poll() is not None:             # 进程已退出 → 失败
            break
        if gpu_mem(gpu) > before + 2000:
            ok = True
            break
    print('%s: gpu=%s pid=%s 显存确认=%s (%.0f→%.0f MiB)' % (
        name, gpu, p.pid, ok, before, gpu_mem(gpu)), flush=True)
    assert ok, '%s 未落到 GPU%s! 检查 Config 竞争' % (name, gpu)
    return [name, ratio, bb, ut, p]


def main():
    idxs = [int(x) for x in sys.argv[1:]]
    assert idxs, '用法: python run_semi_parallel.py <idx...>'

    # 机器可用卡数: 31.26 是 2 张; 31.82 只有 GPU1 可用(GPU0 同学占用) → SEMI_SLOTS=1 SEMI_GPU=1
    slots = int(os.environ.get('SEMI_SLOTS', '2'))
    gpu_fixed = os.environ.get('SEMI_GPU')

    running, pending = [], list(idxs)
    occupied = set()                     # 记录当前被占用的卡号
    while pending or running:
        # 先回收已结束的任务, 释放其占用的卡
        for j in list(running):
            if j[4].poll() is not None:
                occupied.discard(j[5])
                running.remove(j)
        # 只在真正空闲的卡上启动新任务(修复: 原来按 running 数量取模, 会两组挤一张卡)
        while pending:
            free = [g for g in range(slots) if g not in occupied]
            if not free:
                break
            gpu = gpu_fixed if gpu_fixed is not None else str(free[0])
            i = pending.pop(0)
            name, ratio, bb, ut = EXPS[i]
            running.append(launch(name, ratio, bb, ut, gpu) + [int(gpu)])   # 统一存 int
            occupied.add(int(gpu))
        time.sleep(60)
    print('SEMI_TRAIN_DONE', flush=True)

    # 训练全部结束后逐个评估(串行, 避免互相干扰)
    for i in idxs:
        name, ratio, bb, ut = EXPS[i]
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
            w.writerow([name, ratio, bb, ut, dice, True])
        print('%s: dice=%.4f' % (name, dice), flush=True)
    print('SEMI_PARALLEL_ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
