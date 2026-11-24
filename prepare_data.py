# -*- coding: utf-8 -*-
"""
数据准备：下载 Food-101，筛选 20—30 类常见食物，按 70/15/15 分层划分为
训练/验证/测试集，写入 data/{train,val,test}/{class}/ 目录。

用法：
    python prepare_data.py --classes-download        # 首次需下载 Food-101(约5GB)
    python prepare_data.py --seed 42 --link          # 用硬链接代替复制以省磁盘
"""
import argparse
import glob
import os
import random
import shutil

import config


def parse_args():
    p = argparse.ArgumentParser(description="Food-101 子集下载与划分")
    p.add_argument("--download", action="store_true", help="触发 Food-101 下载（约5GB）")
    p.add_argument("--data-root", default=config.DATA_DIR, help="输出目录（默认 data/）")
    p.add_argument("--food101-root", default=os.path.join(config.DATA_DIR, "food-101"),
                   help="torchvision 数据集根目录")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--link", action="store_true", help="用硬链接而非复制文件")
    return p.parse_args()


def ensure_download(root, download):
    """确保 Food-101 已下载解压，返回 images 目录。"""
    from torchvision.datasets import Food101
    images_dir = os.path.join(root, "food-101", "images")
    if os.path.isdir(images_dir) and glob.glob(os.path.join(images_dir, "*", "*.jpg")):
        print("Food-101 已存在，跳过下载。")
        return images_dir
    if not download:
        raise SystemExit("未发现已解压的 Food-101。请加 --download 触发下载（约5GB）。")
    print("下载 Food-101（约5GB，耗时取决于网络）...")
    Food101(root, split="train", download=True)
    return images_dir


def split_and_copy(images_dir, out_dir, classes, seed, use_link):
    rng = random.Random(seed)
    copy_fn = _link if use_link else _copy
    stats = {}
    for c in classes:
        src_dir = os.path.join(images_dir, c)
        if not os.path.isdir(src_dir):
            print(f"警告: 未找到类别目录 {src_dir}，跳过。")
            continue
        files = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
        if not files:
            print(f"警告: 类别 {c} 无图片，跳过。")
            continue
        rng.shuffle(files)
        n = len(files)
        n_train = int(round(n * config.TRAIN_SPLIT))
        n_val = int(round(n * config.VAL_SPLIT))
        train = files[:n_train]
        val = files[n_train:n_train + n_val]
        test = files[n_train + n_val:]
        for split, subset in [("train", train), ("val", val), ("test", test)]:
            dst_dir = os.path.join(out_dir, split, c)
            os.makedirs(dst_dir, exist_ok=True)
            for f in subset:
                copy_fn(f, os.path.join(dst_dir, os.path.basename(f)))
        stats[c] = (len(train), len(val), len(test))
        print(f"{c:<24s} train={len(train):4d} val={len(val):4d} test={len(test):4d}")
    total = [sum(s[i] for s in stats.values()) for i in range(3)]
    print(f"合计: train={total[0]} val={total[1]} test={total[2]} (共 {sum(total)})")
    return stats


def _copy(src, dst):
    if not os.path.exists(dst):
        shutil.copy2(src, dst)


def _link(src, dst):
    if not os.path.exists(dst):
        try:
            os.link(src, dst)
        except OSError:  # 硬链接失败时退回复制
            shutil.copy2(src, dst)


def main():
    args = parse_args()
    images_dir = ensure_download(args.food101_root, args.download)
    os.makedirs(args.data_root, exist_ok=True)
    split_and_copy(images_dir, args.data_root, config.FOOD_CLASSES, args.seed, args.link)


if __name__ == "__main__":
    main()
