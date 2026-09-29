# LViT-Mamba：医学图像分割的复现与改进（文本条件化状态空间模型）

> 基于 [LViT（Language meets Vision Transformer in Medical Image Segmentation, IEEE TMI）](https://github.com/HUANGLIZI/LViT)
> 官方实现的**复现 + 改进**工作。
> 核心贡献：让文本向量直接调制 Mamba 选择性扫描的状态参数（Δ / B / C），
> 在**零推理开销**的前提下提升医学图像分割性能，并通过多种子消融与文本破坏矩阵验证机制。

---

## 主要工作

1. **复现基线** —— 复现 LViT 在 QaTa-COV19 / MosMedData+ 上的结果，与论文差距 ≤ 0.6%
2. **文本条件化状态空间（核心方法）** —— 文本向量以零初始化方式调制 Mamba 的 Δ/B/C 参数；
   训练起点与基线等价（输出差 8.94e-08），增益 +0.87%（约为种子波动的 3.5 倍）
3. **消融与机制验证** —— 参数量-性能消融（B / C / Δ / Δ+B+C 四组）+ 文本破坏矩阵
   （8 种文本破坏 × 多模型），验证有效性来自「让文本进入状态动力学」
4. **半监督框架** —— 对比标签 + LV Loss，支持 10% / 25% / 50% 标注比例
5. **工程化** —— 无人值守实验队列（多卡多任务调度、失败续跑）、集成 + TTA + 阈值寻优评测流程、
   Dice 评测口径核查

## 实验结果

### 复现基线（per-image Dice）

| 数据集 | 本仓库复现 | LViT 论文 |
|---|---|---|
| QaTa-COV19 | 83.03% | 83.66% |
| MosMedData+ | 74.67% | 74.57% |

### 核心方法：文本调制 SSM 状态参数（QaTa-COV19，多种子）

| 配置 | 参数量 | Dice (mean ± std) | n | vs baseline |
|---|---|---|---|---|
| baseline | — | 0.8278 ± 0.25% | 4 | — |
| B-only | 0.2M | 0.8323 | 4 | +0.45% |
| C-only | 0.2M | 0.8338 | 3 | +0.60% |
| Δ-only | 1.1M | 0.8347 | 3 | +0.69% |
| **Δ+B+C（本文方法）** | 1.5M | **0.8365 ± 0.19%** | 5 | **+0.87%** |

- 增益 ≈ 种子波动的 **3.5 倍** → 统计显著
- **零初始化等价性**：初始化时与官方 Mamba 输出差 **8.94e-08**
- **零推理开销**：1.61 vs 1.66 ms/step（与官方 `selective_scan_fn` 持平，比纯 PyTorch 实现快 2.6×）

### 机制验证：文本破坏矩阵（QaTa-COV19）

| 破坏方式 | baseline | 本文方法 |
|---|---|---|
| 换成其他样本的文本 | −13.71% | **−16.89%** |
| 空文本 | −9.42% | −11.44% |
| 只保留 1 / 5 个词 | −9.43% / −6.84% | −10.84% / −7.45% |

→ 本文方法让模型**更依赖文本语义**，而非单纯增加参数或做文本融合。

### 方法论发现：Dice 评测口径

LViT 报告的 83.66% 为 **per-image** 口径；近年部分文献报告的 89–92% 为 **global / micro** 口径
（同一模型两种口径差异可达 7 分以上）。本仓库统一采用 **per-image** 口径，以便与官方结果直接对照。
> 参考实证：TIRNet（arXiv:2606.27794）同一 LViT 模型列出 m-Dice 83.40 / g-Dice 90.63。

## 快速开始

```bash
pip install -r requirements.txt

python train_model.py      # 全监督训练
python train_semi.py       # 半监督训练
python eval_semi.py        # 半监督评估
python eval_ensemble.py    # 集成 + TTA + 阈值寻优评测
python eval_text_corruption.py   # 文本破坏矩阵
python bench_resolution.py       # 效率分析
```

数据集（QaTa-COV19 / MosMedData+ / MoNuSeg）的下载与目录格式与
[官方仓库](https://github.com/HUANGLIZI/LViT) 保持一致，详见其 README。

## 目录结构

```
├── nets/                 # 网络结构：LViT、文本条件 Mamba（Mamba.py / LVMamba.py）、UNet、ViT
├── LV_loss/              # 半监督对比标签资源
├── train_model.py        # 全监督训练入口
├── train_semi.py         # 半监督训练入口
├── semi_loss.py          # LV Loss（对比标签）
├── semi_dataset.py       # 半监督数据管线
├── semi_split.py         # 标注比例划分
├── eval_semi.py          # 半监督评估
├── eval_ensemble.py      # 集成 + TTA + 阈值寻优
├── eval_text_corruption.py / eval_textcond_ckpt.py
├── bench_resolution.py   # 效率（分辨率）分析
├── autoqueue.py / run_*.py / master_queue.sh   # 无人值守实验队列
├── contrast_match.py     # 对比标签匹配
├── Config.py / Load_Dataset.py / Train_one_epoch.py / utils.py / test_model.py
└── requirements.txt
```

## 致谢与引用

本仓库基于 LViT 官方实现（MIT License, Copyright © 2022 Zihan Li）进行复现与改进，
网络结构与数据处理部分沿用了原实现，改进部分（文本条件化状态空间、半监督框架、评测与队列工具）为本仓库新增。

```bibtex
@article{li2023lvit,
  title={LViT: Language meets Vision Transformer in Medical Image Segmentation},
  author={Li, Zihan and Li, Yuchen and Li, Qingde and Wang, Pengfei and Guo, Debin and Lu, Lu and Jin, Dakai and Zhang, Yanchun and Hong, Qingqi},
  journal={IEEE Transactions on Medical Imaging},
  year={2023}
}
```

## License

MIT License（保留原作者版权声明，见 [LICENSE](LICENSE)）

---

## English Abstract

This repository reproduces [LViT](https://github.com/HUANGLIZI/LViT) (IEEE TMI) and extends it with a
**text-conditioned state-space model**: text embeddings modulate the Δ/B/C parameters of Mamba's selective
scan with zero-initialization, yielding **+0.87% Dice** (≈3.5× seed noise, n=5) at **zero inference overhead**
(8.94e-08 equivalence to vanilla Mamba at init). Ablations (B/C/Δ/Δ+B+C) and a text-corruption matrix
confirm the gain comes from *letting text enter the state dynamics*, not from extra parameters.
A semi-supervised framework (contrastive labels + LV Loss, 10%/25%/50% label ratios) and an unattended
multi-GPU experiment queue are also included.
