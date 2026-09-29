#!/bin/bash
# 主控队列（31.26）：等任意一张卡空闲 → 依次执行所有待做实验
#   1) 集成 + TTA + 阈值优化评测（推理，约 20 分钟）
#   2) 划分策略实验：area_strat ×2 种子、text_strat ×2 种子（各约 2-3 小时）
# 全程自动评估并记录；日志 ~/LViT/logs/master_queue.log
#
# GPU 选择：默认只用 GPU1（GPU0 已让给同学，绝不占用）。
# 由 MASTER_GPU_ORDER 覆盖，例如 MASTER_GPU_ORDER="1 0" 才允许用 GPU0。
set -u
cd ~/LViT/LViT-main
PY=${MASTER_PY:-$HOME/miniconda3/envs/mamba_env/bin/python}
exec >> ~/LViT/logs/master_queue.log 2>&1

find_gpu() {
    for g in ${MASTER_GPU_ORDER:-1}; do
        bash ~/LViT/LViT-main/gpu_busy.sh $g || { echo "$g"; return; }
    done
    echo ""
}

wait_any() {                       # 等任意一张卡空闲（最多 48h）
    for i in $(seq 1 2880); do
        g=$(find_gpu)
        [ -n "$g" ] && { echo "$g"; return; }
        sleep 60
    done
}

echo ""
echo "$(date '+%F %T') === 主控队列启动 (GPU候选: ${MASTER_GPU_ORDER:-1}) ==="

# ---------- 1) 集成 + TTA + 阈值评测 ----------
G=$(wait_any)
if [ -z "$G" ]; then echo "$(date '+%F %T') 等待超时，退出"; exit 1; fi
echo "$(date '+%F %T') GPU$G 空闲 → 开始集成评测"
LVIT_GPU=$G $PY -u eval_ensemble.py --task Covid19 \
    --models "Test_session_09.14_18h52:dtBC,Test_session_09.15_01h22:dtBC,Test_session_09.15_14h55:dtBC" \
    --tta flip --thresh_search --batch 4 > ~/LViT/logs/ensemble_eval.log 2>&1
echo "$(date '+%F %T') 集成评测完成 → 结果见 ensemble_eval.log"

# ---------- 2) 数据源验证（重新评估历史半监督 checkpoint + e7 补评估）----------
# 追溯发现半监督配对数据的 10 个数字在服务器上无评估记录，checkpoint 都还在，
# 重评即可坐实。约 1 小时。
G=$(wait_any)
if [ -n "$G" ]; then
    bash ~/LViT/LViT-main/verify_data.sh "$G"
else
    echo "$(date '+%F %T') 等待超时，跳过数据源验证"
fi

# ---------- 3) 划分策略实验 ----------
for spec in \
  "0.25|dtBC|strat_area_s666|--split_strategy,area_strat,--seed,666" \
  "0.25|dtBC|strat_area_s668|--split_strategy,area_strat,--seed,668" \
  "0.25|dtBC|strat_text_s666|--split_strategy,text_strat,--seed,666" \
  "0.25|dtBC|strat_text_s668|--split_strategy,text_strat,--seed,668" ; do
    IFS='|' read -r RATIO CT SUFFIX EXTRA <<< "$spec"
    G=$(wait_any)
    [ -z "$G" ] && { echo "$(date '+%F %T') 等待超时，跳过 $SUFFIX"; continue; }
    echo "$(date '+%F %T') GPU$G → 训练 $SUFFIX"
    EXTRA_ARGS=$(echo "$EXTRA" | tr ',' ' ')
    LVIT_GPU=$G $PY -u train_semi.py --task Covid19 --ratio "$RATIO" --backbone mamba --use_text True \
        --epochs 200 --batch_size 6 --lr 1e-3 --acc_steps 4 --tag_suffix "$SUFFIX" $EXTRA_ARGS \
        > ~/LViT/logs/semi_$SUFFIX.log 2>&1
    echo "$(date '+%F %T') $SUFFIX 训练结束 → 评估"
    LVIT_GPU=$G $PY -u eval_semi.py --task Covid19 --ratio "$RATIO" --backbone mamba \
        --use_text True --tag_suffix "$SUFFIX" --cond_target "$CT" \
        > ~/LViT/logs/eval_$SUFFIX.txt 2>&1
    D=$(grep -oE 'test Dice=[0-9.]+' ~/LViT/logs/eval_$SUFFIX.txt | tail -1)
    echo "$(date '+%F %T') $SUFFIX 评估结果: $D"
    echo "$SUFFIX,$RATIO,$CT,$D" >> ~/LViT/results_split_strategy.csv
done

echo "$(date '+%F %T') === 主控队列全部完成 ==="
