# -*- coding: utf-8 -*-
"""
数据集与数据增强：训练阶段随机裁剪/翻转/颜色抖动/轻度旋转 + ImageNet 归一化；
验证与测试阶段仅缩放 + 中心裁剪，保证评估口径一致。
"""
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config


class FoodDataset(Dataset):
    """目录结构：root/{split}/{class}/xxx.jpg，class 子目录名即标签。"""

    def __init__(self, root, classes, transform=None):
        self.root = root
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform
        self.samples = []
        for c in self.classes:
            cdir = os.path.join(root, c)
            if not os.path.isdir(cdir):
                continue
            for fn in sorted(os.listdir(cdir)):
                if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    self.samples.append((os.path.join(cdir, fn), self.class_to_idx[c]))
        if not self.samples:
            raise RuntimeError(f"在 {root} 下未找到任何图片，请先运行 prepare_data.py")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=config.AUG_HFLIP_PROB),
        transforms.ColorJitter(**config.AUG_COLOR_JITTER),
        transforms.RandomRotation(degrees=config.AUG_ROTATION_DEG),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize(config.RESIZE_SIZE),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def build_dataloaders(data_dir, classes=None, batch_size=None, num_workers=None):
    """构建 train/val/test 三个 DataLoader。"""
    classes = list(classes or config.FOOD_CLASSES)
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = config.NUM_WORKERS if num_workers is None else num_workers

    train_ds = FoodDataset(os.path.join(data_dir, "train"), classes, get_train_transform())
    val_ds = FoodDataset(os.path.join(data_dir, "val"), classes, get_eval_transform())
    test_ds = FoodDataset(os.path.join(data_dir, "test"), classes, get_eval_transform())

    def _loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return (_loader(train_ds, True), _loader(val_ds, False), _loader(test_ds, False),
            len(train_ds), len(val_ds), len(test_ds))
