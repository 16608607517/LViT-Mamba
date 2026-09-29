# -*- coding: utf-8 -*-
"""
半监督数据集：
- SemiSupLabeled: 有标注子集（图像 + mask + 文本）
- SemiSupUnlabeled: 无标注子集（图像 + 文本，无 mask）

复用 ImageToImage2D 的数据读取逻辑，但按划分好的文件列表过滤。
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from Load_Dataset import correct_dims, to_long_tensor


class SemiSupDataset(Dataset):
    """半监督数据集基类：按文件名列表加载指定子集。"""

    def __init__(self, dataset_path, task_name, row_text, file_list,
                 has_mask=True, precomputed_bert=None, image_size=224):
        self.dataset_path = dataset_path
        self.task_name = task_name
        self.row_text = row_text
        self.has_mask = has_mask
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.precomputed_bert = precomputed_bert

        # 按文件列表过滤（mask 命名：Covid19 有 mask_ 前缀，其余与图片同名）
        self.file_list = []
        for fname in file_list:
            if task_name in ['Covid19', 'QaTaCOV19']:
                mask_name = 'mask_' + fname
            else:
                mask_name = fname
            if os.path.exists(os.path.join(self.input_path, fname)):
                self.file_list.append((fname, mask_name))

    def __len__(self):
        return len(self.file_list)

    def _load_bert(self, img_name, mask_name):
        """读取 BERT 嵌入。Covid19 的 precomputed_bert/row_text 键带 mask_ 前缀，其余用图片名。"""
        key = mask_name if self.task_name in ['Covid19', 'QaTaCOV19'] else img_name
        if self.precomputed_bert is not None and key in self.precomputed_bert:
            text = self.precomputed_bert[key].float().numpy()
            if text.shape[0] > 10:
                text = text[:10, :]
            return text
        # fallback：实时 BERT（若未预计算）
        text = self.row_text[key]
        from Load_Dataset import BertEmbedding
        be = BertEmbedding()
        tokens = be([text.split('\n')[0]])
        emb = np.array(tokens[0][1])
        return emb[:10] if emb.shape[0] > 10 else emb

    def __getitem__(self, idx):
        img_name, mask_name = self.file_list[idx]
        image = cv2.imread(os.path.join(self.input_path, img_name))
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0  # 归一化到 [0,1]
        image = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)

        text = self._load_bert(img_name, mask_name)

        sample = {'image': image, 'text': text}

        if self.has_mask:
            mask = cv2.imread(os.path.join(self.output_path, mask_name), 0)
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            mask[mask <= 0] = 0
            mask[mask > 0] = 1
            sample['label'] = torch.from_numpy(mask.astype(np.uint8)).long()

        return sample, img_name


def build_semi_datasets(dataset_path, task_name, row_text, labeled_list, unlabeled_list,
                        precomputed_bert_path=None, image_size=224):
    """构建有/无标注数据集。

    Returns:
        labeled_ds, unlabeled_ds
    """
    precomputed = None
    if precomputed_bert_path and os.path.exists(precomputed_bert_path):
        precomputed = torch.load(precomputed_bert_path, map_location='cpu')

    labeled_ds = SemiSupDataset(dataset_path, task_name, row_text, labeled_list,
                                has_mask=True, precomputed_bert=precomputed,
                                image_size=image_size)
    unlabeled_ds = SemiSupDataset(dataset_path, task_name, row_text, unlabeled_list,
                                  has_mask=False, precomputed_bert=precomputed,
                                  image_size=image_size)
    return labeled_ds, unlabeled_ds
