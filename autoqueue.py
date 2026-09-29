# -*- coding: utf-8 -*-
"""无人值守实验队列: 等 GPU 空闲 → 配置 Config → 训练 → 评估 → 记 CSV → 下一个。

用法:
  nohup python autoqueue.py <gpu> "<job1>" "<job2>" ... &
  job 格式: name|task|cond_target|seed|lr
    cond_target: none = 普通 Mamba(无文本调制); B/C/dtBC = 文本调制目标
  例: python autoqueue.py 0 "jobA|Covid19|dtBC|668|3e-4" "jobB|MosMedDataPlus|B|666|1e-3"

特性: 单任务失败(崩溃/评估失败)不影响后续任务; 全部过程记入 ~/LViT/logs/autoqueue.log
"""
import csv
import os
import re
import subprocess
import sys
import time

BASE = os.path.expanduser('~/LViT/LViT-main')
LOG = os.path.expanduser('~/LViT/logs')
RESULTS = os.path.expanduser('~/LViT/results_autoqueue.csv')
QLOG = os.path.join(LOG, 'autoqueue.log')
PY = os.path.expanduser('~/LViT/run_py.sh')
if not os.path.exists(PY):
    PY = os.path.expanduser('~/miniconda3/envs/mamba_env/bin/python')
PYTHON = os.path.expanduser('~/miniconda3/envs/mamba_env/bin/python')


def log(msg):
    line = '%s %s' % (time.strftime('%F %T'), msg)
    print(line, flush=True)
    with open(QLOG, 'a') as f:
        f.write(line + '\n')


def gpu_mem(gpu):
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used',
                          '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    for line in out.strip().split('\n'):
        i, m = line.split(',')
        if i.strip() == str(gpu):
            return int(m)
    return -1


def wait_free(gpu, thresh=1200, max_hours=72):
    """等 GPU 显存降到阈值以下(视为空闲)。"""
    for _ in range(int(max_hours * 60)):
        m = gpu_mem(gpu)
        if m >= 0 and m < thresh:
            return True
        time.sleep(60)
    return False


def patch_config(task, cond_target, seed, lr, gpu):
    path = os.path.join(BASE, 'Config.py')
    with open(path) as f:
        s = f.read()

    def rep(pat, val):
        nonlocal s
        s, n = re.subn(pat, val, s, flags=re.M)
        assert n >= 1, 'pattern not found: ' + pat
    rep(r'^task_name = .*$', "task_name = '%s'" % task)
    rep(r'^backbone = .*$', "backbone = 'mamba'")
    rep(r'^use_text = .*$', 'use_text = True')
    rep(r'^batch_size = .*$', 'batch_size = 12')
    rep(r'^epochs = .*$', 'epochs = 2000')
    rep(r'^learning_rate = .*$', 'learning_rate = %s' % lr)
    rep(r'^seed = .*$', 'seed = %s' % seed)
    if cond_target == 'none':
        rep(r'^text_cond = .*$', 'text_cond = False')
    else:
        rep(r'^text_cond = .*$', 'text_cond = True')
        rep(r'^cond_target = .*$', "cond_target = '%s'" % cond_target)
    rep(r'^os\.environ\["CUDA_VISIBLE_DEVICES"\] = .*$',
        'os.environ["CUDA_VISIBLE_DEVICES"] = "%s"' % gpu)
    with open(path, 'w') as f:
        f.write(s)


def wait_process(p, name, max_hours=12):
    """等训练进程结束(最多 max_hours 小时)。"""
    for _ in range(int(max_hours * 60)):
        if p.poll() is not None:
            return p.returncode
        time.sleep(60)
    p.kill()
    log('%s 超时被终止' % name)
    return -9


def evaluate(name, task, cond_target, session, gpu, seed='', lr=''):
    """评估并把 Dice 写入总表。返回 dice。"""
    dice = -1.0
    try:
        if cond_target == 'none':
            path = os.path.join(BASE, 'Config.py')
            with open(path) as f:
                s = f.read()
            s = re.sub(r'^test_session = .*$', 'test_session = "%s"' % session, s, flags=re.M)
            s = re.sub(r'^text_cond = .*$', 'text_cond = False', s, flags=re.M)
            with open(path, 'w') as f:
                f.write(s)
            env = os.environ.copy()
            env['LVIT_GPU'] = str(gpu)
            r = subprocess.run([PYTHON, '-u', 'test_model.py'], capture_output=True,
                               text=True, cwd=BASE, timeout=7200, env=env)
            m = re.findall(r'dice_pred (\d\.\d+)', r.stdout)
            dice = float(m[-1]) if m else -1.0
        else:
            r = subprocess.run([PYTHON, '-u', 'eval_textcond_ckpt.py', session,
                                cond_target, str(gpu), task],
                               capture_output=True, text=True, cwd=BASE, timeout=7200)
            m = re.search(r'TEST_DICE ([\d.]+)', r.stdout)
            dice = float(m.group(1)) if m else -1.0
            with open(os.path.join(LOG, 'eval_%s.txt' % name), 'w') as f:
                f.write(r.stdout[-3000:])
    except Exception as e:
        log('%s 评估异常: %s' % (name, str(e)[:120]))
    with open(RESULTS, 'a', newline='') as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(['name', 'task', 'cond_target', 'seed', 'lr', 'session', 'test_dice'])
        w.writerow([name, task, cond_target, seed, lr, session, dice])
    return dice


def main():
    gpu = sys.argv[1]
    jobs = []
    for spec in sys.argv[2:]:
        parts = spec.split('|')
        jobs.append(parts)          # name, task, cond_target, seed, lr

    log('=== autoqueue 启动 GPU%s, %d 个任务 ===' % (gpu, len(jobs)))
    for name, task, ct, seed, lr in jobs:
        try:
            log('[%s] 等待 GPU%s 空闲...' % (name, gpu))
            if not wait_free(gpu):
                log('[%s] 等待超时, 跳过' % name)
                continue
            # 加锁配置 Config(避免与其他 GPU 的队列竞争)
            lock = open('/tmp/config.lock', 'w')
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                patch_config(task, ct, seed, lr, gpu)
                time.sleep(5)
                logf = open(os.path.join(LOG, name + '.log'), 'w')
                p = subprocess.Popen([PY, '-u', 'train_model.py', '--acc_steps', '2'],
                                     stdout=logf, stderr=subprocess.STDOUT, cwd=BASE)
                time.sleep(60)
                log('[%s] 启动 pid=%s, GPU%s 显存=%s MiB' % (name, p.pid, gpu, gpu_mem(gpu)))
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()

            rc = wait_process(p, name)
            log('[%s] 训练结束 rc=%s' % (name, rc))
            # 找 session
            session = ''
            try:
                with open(os.path.join(LOG, name + '.log')) as f:
                    m = re.search(r'Test_session_\d+\.\d+_\d+h\d+', f.read())
                    session = m.group(0) if m else ''
            except Exception:
                pass
            if rc == 0 and session:
                d = evaluate(name, task, ct, session, gpu, seed=seed, lr=lr)
                log('[%s] 评估完成 dice=%.4f (session=%s)' % (name, d, session))
            else:
                log('[%s] 跳过评估(rc=%s session=%s)' % (name, rc, session))
        except Exception as e:
            log('[%s] 任务异常: %s' % (name, str(e)[:150]))
            time.sleep(120)
    log('=== autoqueue GPU%s 全部完成 ===' % gpu)


if __name__ == '__main__':
    main()
