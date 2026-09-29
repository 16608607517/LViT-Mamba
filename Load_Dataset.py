# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage
# from bert_embedding import BertEmbedding  # replaced with HuggingFace transformers
import os
import warnings
warnings.filterwarnings('ignore')

from transformers import BertTokenizer, BertModel
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)


class BertEmbedding:
    """Compatible wrapper matching bert_embedding.BertEmbedding API using HuggingFace transformers.
    Uses HF-Mirror for China mainland users when HF_ENDPOINT is set, falls back to random embeddings
    when the model is not available.
    """
    def __init__(self, model_name='bert-base-uncased'):
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self._use_mock = False
            print(f'[BertEmbedding] Loaded {model_name} successfully.')
        except Exception as e:
            print(f'[BertEmbedding] Cannot load BERT model ({e}), using random embeddings as fallback.')
            print(f'[BertEmbedding] Set HF_ENDPOINT=https://hf-mirror.com for China mainland mirror.')
            self._use_mock = True

    def __call__(self, sentences):
        if self._use_mock:
            results = []
            for sentence in sentences:
                # Generate random embeddings matching bert-base-uncased output shape
                words = sentence.split()
                seq_len = min(len(words) + 2, 20)  # approximate: words + [CLS] + [SEP]
                embedding = np.random.randn(seq_len, 768).astype(np.float32)
                tokens = ['[CLS]'] + words + ['[SEP]']
                tokens = tokens[:seq_len]
                results.append((tokens, embedding))
            return results
        else:
            results = []
            for sentence in sentences:
                inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = outputs.last_hidden_state.squeeze(0).cpu().numpy()
                tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                results.append((tokens, embedding))
            return results


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {'image': image, 'label': label, 'text': text}
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text = sample['image'], sample['label'], sample['text']
        image, label = image.astype(np.uint8), label.astype(np.uint8)  # OSIC
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        text = torch.Tensor(text)
        sample = {'image': image, 'label': label, 'text': text}
        return sample


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class LV2D(Dataset):
    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.output_path = os.path.join(dataset_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.bert_embedding = BertEmbedding()

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.output_path))

    def __getitem__(self, idx):

        mask_filename = self.mask_list[idx]  # Co
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1
        mask = correct_dims(mask)
        text = self.rowtext[mask_filename]
        text = text.split('\n')
        text_token = self.bert_embedding(text)
        text = np.array(text_token[0][1])
        if text.shape[0] > 14:
            text = text[:14, :]
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'label': mask, 'text': text}

        return sample, mask_filename


class ImageToImage2D(Dataset):

    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = sorted(os.listdir(self.input_path))
        self.mask_list = sorted(os.listdir(self.output_path))
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name

        # Try to load precomputed BERT embeddings
        precomputed_path = os.path.join(dataset_path, 'precomputed_bert.pt')
        if os.path.exists(precomputed_path):
            print(f'[ImageToImage2D] Loading precomputed BERT embeddings from {precomputed_path}')
            self.precomputed_bert = torch.load(precomputed_path, map_location='cpu')
            self.bert_embedding = None  # 不再需要实时BERT
        else:
            print(f'[ImageToImage2D] No precomputed embeddings found, using real-time BERT')
            self.precomputed_bert = None
            self.bert_embedding = BertEmbedding()

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):

        if self.task_name in ['Covid19', 'QaTaCOV19']:
            # Covid19 naming: mask file in labelcol has 'mask_' prefix
            mask_filename = self.mask_list[idx]
            image_filename = mask_filename.replace('mask_', '')
        else:
            # MoNuSeg naming: image and mask have same basename, different extensions
            image_filename = self.images_list[idx]
            mask_filename = image_filename[: -3] + "png"
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        image = cv2.resize(image, (self.image_size, self.image_size))

        # read mask image
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)

        # 文本嵌入：优先使用预计算缓存
        if self.precomputed_bert is not None:
            text = self.precomputed_bert[mask_filename]  # torch tensor [seq_len, 768]
            if text.shape[0] > 10:
                text = text[:10, :]
            text = text.float().numpy()
        else:
            text = self.rowtext[mask_filename]
            text = text.split('\n')
            text_token = self.bert_embedding(text)
            text = np.array(text_token[0][1])
            if text.shape[0] > 10:
                text = text[:10, :]

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'image': image, 'label': mask, 'text': text}

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, image_filename
