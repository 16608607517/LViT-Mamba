# -*- coding: utf-8 -*-
import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
# 由环境变量 LVIT_GPU 指定显卡（队列脚本用），未设置时退回 0；避免硬编码误占他人卡
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("LVIT_GPU", "0")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "d:/huggingface_cache"
os.environ["TORCH_HOME"] = "d:/torch_cache"
os.environ["CUDA_CACHE_PATH"] = "d:/cuda_cache"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True  # Use cosineLR or not
n_channels = 3
n_labels = 1  # MoNuSeg & Covid19
epochs = 2000
img_size = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 50
early_stopping_patience = 50

pretrain = False

# === 实验配置开关 ===
# backbone: 'transformer' (原LViT) 或 'mamba' (LVMamba)
# use_text: True (文本引导) 或 False (纯视觉消融)
backbone = 'transformer'
use_text = False     # 消融组: Trans 无文本

# === Mamba 改进开关（C1: cross-scan 四向扫描）===
mamba_depth = 1              # 1=原版深度
scan_mode = 'bidirectional'  # 消融基线: 双向扫描
text_gate = False            # 门控关闭

# === 文本条件化选择性扫描（创新点①）===
text_cond = False            # True: 文本向量直接调制扫描参数 Δ/B/C（强制走纯 PyTorch Mamba）
cond_target = 'dtBC'         # 可消融: 'dt' / 'B' / 'C' / 'dtBC'

# === 数据集 ===
# task_name = 'MoNuSeg'
task_name = 'Covid19'
# task_name = 'Covid19'
# task_name = 'MoNuSeg'
learning_rate = 3e-4  # 实验结论: batch4 下 lr3e-4 最优（论文1e-3是batch24配套）
batch_size = 4  # 适度增大加速训练

model_name = 'LVMamba'  # C1: cross-scan 实验

if task_name == 'MosMedDataPlus':
    train_dataset = './datasets/MosMedDataPlus/Train_Folder/'
    val_dataset = './datasets/MosMedDataPlus/Val_Folder/'
    test_dataset = './datasets/MosMedDataPlus/Test_Folder/'
    task_dataset = './datasets/MosMedDataPlus/Train_Folder/'
else:
    train_dataset = './datasets/' + task_name + '/Train_Folder/'
    val_dataset = './datasets/' + task_name + '/Val_Folder/'
    test_dataset = './datasets/' + task_name + '/Test_Folder/'
    task_dataset = './datasets/' + task_name + '/Train_Folder/'
session_name = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path = task_name + '/' + model_name + '/' + session_name + '/'
model_path = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path = save_path + session_name + ".log"
visualize_path = save_path + 'visualize_val/'


##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.expand_ratio = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 64  # base channel of U-Net
    config.n_classes = 1
    return config


# used in testing phase, copy the session name in training phase
test_session = "Test_session_08.31_06h58"  # C2: cross-scan Covid19