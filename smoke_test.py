# -*- coding: utf-8 -*-
"""
端到端冒烟测试：用合成小数据集验证整条链路（数据加载 → 两阶段训练 →
推理 → 热量映射 → Gradio 界面构建）可运行，无需下载 Food-101。

用法：
    python smoke_test.py
"""
import os
import shutil
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import config

DATA = os.path.join(config.BASE_DIR, "_smoke_data")
OUT = os.path.join(config.BASE_DIR, "_smoke_output")


def gen_synthetic_data():
    """为全部 30 类生成 96×96 的彩色噪声图片，带类别颜色偏置。"""
    if os.path.isdir(DATA):
        shutil.rmtree(DATA)
    rng = np.random.default_rng(0)
    for split, n in [("train", 6), ("val", 2), ("test", 2)]:
        for ci, c in enumerate(config.FOOD_CLASSES):
            d = os.path.join(DATA, split, c)
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                base = np.zeros((96, 96, 3), dtype=np.uint8)
                base[:, :, 0] = (ci * 37) % 256
                base[:, :, 1] = (ci * 71 + 50) % 256
                base[:, :, 2] = (ci * 103 + 20) % 256
                noise = rng.integers(0, 40, (96, 96, 3)).astype(np.uint8)
                arr = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(os.path.join(d, f"{i:02d}.jpg"))
    print(f"合成数据已生成: {DATA}")


def train_smoke():
    from dataset import build_dataloaders
    from models import FoodClassifier, load_classifier
    from train import train_stage

    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cpu")
    args = SimpleNamespace(
        epochs_stage1=1, epochs_stage2=1, lr_stage1=1e-3, lr_stage2=1e-4,
        weight_decay=1e-4, class_weights=False, label_smoothing=0.0,
        output_dir=OUT, backbone="mobilenet_v3_large",
    )
    loaders = build_dataloaders(DATA, batch_size=8, num_workers=0)
    _, _, test_loader, n_tr, n_va, n_te = loaders
    print(f"样本数: train={n_tr} val={n_va} test={n_te}")

    model = FoodClassifier("mobilenet_v3_large", config.NUM_CLASSES, pretrained=False).to(device)
    model.set_backbone_trainable(False)
    train_stage(model, loaders, args, device, stage=1)
    model.set_backbone_trainable(True, stage_only=True)
    train_stage(model, loaders, args, device, stage=2)

    best = os.path.join(OUT, "best_model_mobilenet_v3_large.pth")
    assert os.path.exists(best), "未生成最优权重"
    final = load_classifier("mobilenet_v3_large", best, config.NUM_CLASSES, device)
    from train import evaluate, make_criterion
    crit = make_criterion(False, 0.0, device, test_loader)
    te_loss, te_acc, te_top5 = evaluate(final, test_loader, crit, device)
    print(f"测试集: Top-1={te_acc:.3f} Top-5={te_top5:.3f} loss={te_loss:.3f}")
    return best


def predict_smoke(best):
    from calorie import estimate_calories, load_calorie_table
    from dataset import get_eval_transform
    from models import load_classifier
    import time

    device = torch.device("cpu")
    model = load_classifier("mobilenet_v3_large", best, config.NUM_CLASSES, device)
    table = load_calorie_table()
    sample = os.path.join(DATA, "test", config.FOOD_CLASSES[0], "00.jpg")
    tensor = get_eval_transform()(Image.open(sample).convert("RGB")).unsqueeze(0).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu()
    dt = (time.perf_counter() - t0) * 1000
    top1 = config.FOOD_CLASSES[probs.argmax().item()]
    cal = estimate_calories(top1, probs.tolist(), config.FOOD_CLASSES,
                            portion="medium", grams=None, table=table)
    print(f"预测 Top-1: {top1}  概率: {cal['top1_prob']}")
    print(f"热量: {cal['kcal']} kcal ({cal['kcal_low']}~{cal['kcal_high']}) "
          f"蛋白{cal['protein']}g 脂肪{cal['fat']}g 碳水{cal['carb']}g")
    print(f"推理时延: {dt:.1f} ms")


def app_smoke():
    from types import SimpleNamespace
    from app import build_demo
    args = SimpleNamespace(backbone="mobilenet_v3_large", weight=os.path.join(OUT, "best_model_mobilenet_v3_large.pth"),
                           calorie_csv=config.CALORIE_CSV, device="cpu",
                           server_name="127.0.0.1", port=7860, share=False)
    demo = build_demo(args)
    print("Gradio 界面构建成功（未启动服务）。")


def main():
    gen_synthetic_data()
    best = train_smoke()
    predict_smoke(best)
    app_smoke()
    print("\n全部冒烟测试通过 ✅")


if __name__ == "__main__":
    main()
