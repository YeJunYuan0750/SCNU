# -*- coding: utf-8 -*-
"""
推理脚本：单图与批量推理，输出 Top-K 类别概率、热量估计与推理时延。
时延口径：224×224 单张图片在固定设备上预热后重复 100 次的平均与中位数耗时。

用法示例：
    python predict.py --image demo.jpg --backbone resnet50 \
        --weight output/best_model_resnet50.pth
    python predict.py --dir data/test/pizza --backbone mobilenet_v3_large \
        --weight output/best_model_mobilenet_v3_large.pth --portion medium
"""
import argparse
import glob
import json
import os
import statistics
import time

import torch
from PIL import Image

import config
from calorie import estimate_calories, load_calorie_table
from dataset import get_eval_transform
from models import load_classifier


def parse_args():
    p = argparse.ArgumentParser(description="食物识别与热量估算推理")
    p.add_argument("--image", default=None, help="单张图片路径")
    p.add_argument("--dir", default=None, help="图片目录（批量推理）")
    p.add_argument("--backbone", default="resnet50", choices=config.BACKBONES)
    p.add_argument("--weight", required=True, help="训练好的权重 .pth")
    p.add_argument("--calorie-csv", default=config.CALORIE_CSV)
    p.add_argument("--portion", default=config.DEFAULT_PORTION,
                   choices=list(config.PORTION_COEFFS))
    p.add_argument("--grams", type=float, default=None, help="显式克重（覆盖份量档位）")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--device", default=None)
    p.add_argument("--json", action="store_true", help="以 JSON 输出")
    return p.parse_args()


def preprocess(path):
    img = Image.open(path).convert("RGB")
    return get_eval_transform()(img).unsqueeze(0)


def infer(model, tensor, device):
    with torch.no_grad():
        logits = model(tensor.to(device))
    probs = torch.softmax(logits, dim=1)[0].cpu()
    return probs


def measure_latency(model, tensor, device, runs=100):
    """预热后重复 runs 次，返回平均与中位数时延（秒）。"""
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(tensor.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(tensor.to(device))
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return statistics.mean(times), statistics.median(times)


def predict_one(model, device, path, table, args):
    tensor = preprocess(path)
    probs = infer(model, tensor, device)
    topk = probs.topk(args.topk)
    top_names = [config.FOOD_CLASSES[i] for i in topk.indices.tolist()]
    top_probs = topk.values.tolist()

    # 完整概率向量用于热量区间估计
    full_probs = probs.tolist()
    top1 = top_names[0]
    cal = estimate_calories(top1, full_probs, config.FOOD_CLASSES,
                            portion=args.portion, grams=args.grams, table=table)

    avg_ms, med_ms = measure_latency(model, tensor, device)
    result = {
        "image": path,
        "topk": [{"class": n, "prob": round(p, 4)} for n, p in zip(top_names, top_probs)],
        "calorie": cal,
        "latency": {"avg_ms": round(avg_ms * 1000, 2), "median_ms": round(med_ms * 1000, 2)},
    }
    return result


def pretty(res):
    print(f"图片: {res['image']}")
    print(f"Top-{len(res['topk'])} 预测:")
    for i, item in enumerate(res["topk"], 1):
        print(f"  {i}. {item['class']:<24s} {item['prob']*100:5.2f}%")
    c = res["calorie"]
    print(f"热量估计: {c['kcal']} kcal ({c['kcal_low']} ~ {c['kcal_high']} kcal)")
    print(f"  份量: {c['portion']} (≈{c['grams_effective']}g) | 蛋白 {c['protein']}g "
          f"脂肪 {c['fat']}g 碳水 {c['carb']}g")
    print(f"推理时延: 平均 {res['latency']['avg_ms']}ms / 中位 {res['latency']['median_ms']}ms")


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_classifier(args.backbone, args.weight, config.NUM_CLASSES, device)
    table = load_calorie_table(args.calorie_csv)

    if args.image:
        results = [predict_one(model, device, args.image, table, args)]
    elif args.dir:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
        paths = sorted(p for e in exts for p in glob.glob(os.path.join(args.dir, e)))
        results = [predict_one(model, device, p, table, args) for p in paths]
    else:
        raise SystemExit("请提供 --image 或 --dir")

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for r in results:
            pretty(r)
            print()


if __name__ == "__main__":
    main()
