# -*- coding: utf-8 -*-
"""
把 HuggingFace `ethz/food101` 的 parquet 分片转换成训练脚本使用的 ImageFolder 结构。

原始 Food-101 官方下载（5 GB tar.gz）在国内网络下速度不足 100 KB/s，
而 HF 镜像（hf-mirror.com）可以稳定下载，因此本脚本直接从 parquet 分片
解码图片、按本项目 30 类筛选、按类别重新分层划分为 70%/15%/15%。

注意：parquet 里的 `label` 是 ClassLabel 的整数下标，类别名保存在
schema metadata（huggingface 字段）的 features.label.names 里，需先解析映射。

用法：
    python fetch_food101.py --parquet-dir D:/foodcal/data --out D:/foodcal/dataset
"""
import argparse
import csv
import io
import json
import os
import time

import pyarrow.parquet as pq
from PIL import Image

import config

TARGET = set(config.FOOD_CLASSES)
SPLIT = (("train", 0.70), ("val", 0.15), ("test", 0.15))
SHORT_SIDE = 380
JPEG_QUALITY = 92


def list_parquet(root):
    files = sorted(f for f in os.listdir(root) if f.endswith(".parquet"))
    return [os.path.join(root, f) for f in files]


def class_names(path):
    """从 parquet 的 huggingface 元数据里取出 101 个类别名（按 label 下标排列）。"""
    meta = json.loads(pq.ParquetFile(path).schema_arrow.metadata[b"huggingface"].decode("utf-8"))
    return meta["info"]["features"]["label"]["names"]


def count_targets(files, index2name, target_idx):
    """第一遍：只读 label 列，统计每个目标类别的样本数。"""
    counts = {c: 0 for c in config.FOOD_CLASSES}
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=16384, columns=["label"]):
            for idx in batch.column(0).to_pylist():
                if idx in target_idx:
                    counts[index2name[idx]] += 1
    return counts


def split_bounds(count):
    n_train = int(round(count * SPLIT[0][1]))
    n_val = int(round(count * SPLIT[1][1]))
    n_test = count - n_train - n_val
    return n_train, n_val, n_test


def convert(files, out_dir, counts, index2name, target_idx):
    split_of = {}
    for c, n in counts.items():
        n_train, n_val, _ = split_bounds(n)
        split_of[c] = (n_train, n_train + n_val)

    for split, _ in SPLIT:
        for c in config.FOOD_CLASSES:
            os.makedirs(os.path.join(out_dir, split, c), exist_ok=True)

    seen = {c: 0 for c in config.FOOD_CLASSES}
    manifest = []
    t0 = time.time()
    for path in files:
        pf = pq.ParquetFile(path)
        name = os.path.basename(path)
        for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
            labels = batch.column(1).to_pylist()
            if not any(l in target_idx for l in labels):
                continue
            images = batch.column(0)
            for i, idx in enumerate(labels):
                if idx not in target_idx:
                    continue
                label = index2name[idx]
                k = seen[label]
                seen[label] += 1
                n_train, n_val_end = split_of[label]
                split = "train" if k < n_train else ("val" if k < n_val_end else "test")
                raw = images[i].as_py()
                if isinstance(raw, dict):  # HF Image 特征存成 {"bytes":..,"path":..}
                    raw = raw.get("bytes")
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                w, h = img.size
                scale = SHORT_SIDE / min(w, h)
                if scale < 1.0:
                    img = img.resize((max(1, int(round(w * scale))),
                                      max(1, int(round(h * scale)))), Image.LANCZOS)
                dst_name = f"{label}_{k:05d}.jpg"
                img.save(os.path.join(out_dir, split, label, dst_name),
                         "JPEG", quality=JPEG_QUALITY, optimize=False)
                manifest.append([split, label, dst_name])
            if sum(seen.values()) % 2000 < 256:
                print(f"  已处理 {sum(seen.values())} 张 | {time.time() - t0:.0f}s", flush=True)
        print(f"[{name}] 累计 {sum(seen.values())} 张 | 用时 {time.time() - t0:.0f}s", flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", default=r"D:/foodcal/data")
    p.add_argument("--out", default=r"D:/foodcal/dataset")
    args = p.parse_args()

    files = list_parquet(args.parquet_dir)
    if not files:
        raise SystemExit(f"未找到 parquet 文件：{args.parquet_dir}")
    names = class_names(files[0])
    index2name = {i: n for i, n in enumerate(names)}
    target_idx = {i for i, n in enumerate(names) if n in TARGET}
    missing = TARGET - set(names)
    print(f"parquet 分片 {len(files)} 个 | 类别表 {len(names)} 类 | 命中目标 {len(target_idx)} 类", flush=True)
    if missing:
        raise SystemExit(f"数据集中缺少这些类别：{sorted(missing)}")

    counts = count_targets(files, index2name, target_idx)
    total = sum(counts.values())
    print(f"30 类目标样本总数 = {total}", flush=True)
    for c in config.FOOD_CLASSES:
        n_train, n_val, n_test = split_bounds(counts[c])
        print(f"  {c:<22} 共 {counts[c]:>5} | train {n_train:>5} val {n_val:>4} test {n_test:>4}",
              flush=True)

    manifest = convert(files, args.out, counts, index2name, target_idx)
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "label", "file"])
        w.writerows(manifest)

    for split, _ in SPLIT:
        n = sum(len(os.listdir(os.path.join(args.out, split, c))) for c in config.FOOD_CLASSES)
        print(f"{split}: {n} 张")
    print("完成 ->", args.out)


if __name__ == "__main__":
    main()
