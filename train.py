# -*- coding: utf-8 -*-
"""
迁移学习训练脚本：两阶段训练。

阶段一：冻结骨干，仅训练新分类头（lr 较大，快速收敛）。
阶段二：解冻骨干最后一个 stage，以小学习率微调。
优化器 AdamW + 余弦退火学习率 + 交叉熵（可选标签平滑/类别权重），
按验证集 Top-1 准确率保存最优权重，并输出损失/准确率曲线与分类报告。

用法示例：
    python train.py --backbone resnet50 --data-dir data
    python train.py --backbone efficientnet_b0 --epochs-stage2 20
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

import config
from dataset import build_dataloaders
from models import FoodClassifier


def parse_args():
    p = argparse.ArgumentParser(description="两阶段迁移学习训练")
    p.add_argument("--backbone", default="resnet50", choices=config.BACKBONES)
    p.add_argument("--data-dir", default=config.DATA_DIR)
    p.add_argument("--output-dir", default=config.OUTPUT_DIR)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--epochs-stage1", type=int, default=config.STAGE1_EPOCHS)
    p.add_argument("--epochs-stage2", type=int, default=config.STAGE2_EPOCHS)
    p.add_argument("--lr-stage1", type=float, default=config.STAGE1_LR)
    p.add_argument("--lr-stage2", type=float, default=config.STAGE2_LR)
    p.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--label-smoothing", type=float, default=config.LABEL_SMOOTHING)
    p.add_argument("--class-weights", action="store_true", default=config.USE_CLASS_WEIGHTS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda/cpu，默认自动")
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_optimizer(model, lr, weight_decay):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )


def make_criterion(use_class_weights, label_smoothing, device, loader):
    if use_class_weights:
        counts = torch.zeros(config.NUM_CLASSES)
        for _, labels in loader:
            for l in labels.tolist():
                counts[l] += 1
        counts = counts.clamp(min=1.0)
        weights = counts.sum() / (config.NUM_CLASSES * counts)
        weights = weights.to(device)
        print("类别权重:", weights.tolist())
    else:
        weights = None
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, top5_correct, n = 0.0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        _, top5 = logits.topk(5, dim=1)
        correct += (logits.argmax(1) == y).sum().item()
        top5_correct += (top5 == y.view(-1, 1)).any(dim=1).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n, top5_correct / n


def run_epoch(model, loader, optimizer, criterion, scheduler, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    if scheduler is not None and train:
        scheduler.step()
    return total_loss / n, correct / n


def train_stage(model, loaders, args, device, stage):
    train_loader, val_loader, test_loader, *_ = loaders
    epochs = args.epochs_stage1 if stage == 1 else args.epochs_stage2
    lr = args.lr_stage1 if stage == 1 else args.lr_stage2
    criterion = make_criterion(args.class_weights, args.label_smoothing, device, train_loader)
    optimizer = make_optimizer(model, lr, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_path = os.path.join(args.output_dir, f"best_model_{args.backbone}.pth")
    history = {"stage": stage, "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "epoch_time": []}

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, scheduler, device, train=True)
        va_loss, va_acc, va_top5 = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["epoch_time"].append(dt)
        print(f"[stage{stage}] ep {ep:02d}/{epochs} | tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} "
              f"| val_loss {va_loss:.4f} val_acc {va_acc:.4f} top5 {va_top5:.4f} | {dt:.1f}s")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"state_dict": model.state_dict(), "backbone": args.backbone,
                        "classes": config.FOOD_CLASSES, "val_acc": va_acc}, best_path)

    print(f"[stage{stage}] 完成，最优验证 Top-1 = {best_acc:.4f} -> {best_path}")
    return history, best_path


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"设备: {device} | 骨干: {args.backbone} | 类别数: {config.NUM_CLASSES}")
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    loaders = build_dataloaders(args.data_dir, batch_size=args.batch_size,
                                num_workers=args.num_workers)
    _, _, test_loader, n_train, n_val, n_test = loaders
    print(f"样本数: train={n_train} val={n_val} test={n_test}")

    model = FoodClassifier(args.backbone, config.NUM_CLASSES).to(device)

    # 阶段一：冻结骨干，仅训练分类头
    model.set_backbone_trainable(False)
    h1, _ = train_stage(model, loaders, args, device, stage=1)

    # 阶段二：解冻最后一个 stage 微调
    model.set_backbone_trainable(True, stage_only=True)
    h2, best_path = train_stage(model, loaders, args, device, stage=2)

    # 用最优权重在测试集上评估
    from models import load_classifier
    final = load_classifier(args.backbone, best_path, config.NUM_CLASSES, device)
    criterion = make_criterion(False, 0.0, device, test_loader)
    te_loss, te_acc, te_top5 = evaluate(final, test_loader, criterion, device)
    print(f"测试集: Top-1 = {te_acc:.4f}  Top-5 = {te_top5:.4f}  损失 = {te_loss:.4f}")

    # 分类报告 + 混淆矩阵数据
    final.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y_pred.extend(final(x).argmax(1).cpu().tolist())
            y_true.extend(y.tolist())
    report = classification_report(y_true, y_pred, target_names=config.FOOD_CLASSES,
                                   zero_division=0, output_dict=True)
    result = {"backbone": args.backbone, "test_acc": te_acc, "test_top5": te_top5,
              "history": {"stage1": h1, "stage2": h2}, "classification_report": report}
    with open(os.path.join(args.output_dir, f"results_{args.backbone}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    plot_curves(h1, h2, args.backbone, args.output_dir)
    print(f"结果已写入 {args.output_dir}")


def plot_curves(h1, h2, backbone, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        stages = [("stage1", h1), ("stage2", h2)]
        offset = 0
        for tag, h in stages:
            xs = list(range(offset + 1, offset + len(h["train_loss"]) + 1))
            axes[0].plot(xs, h["train_loss"], label=f"{tag} train")
            axes[0].plot(xs, h["val_loss"], label=f"{tag} val")
            axes[1].plot(xs, h["train_acc"], label=f"{tag} train")
            axes[1].plot(xs, h["val_acc"], label=f"{tag} val")
            offset += len(h["train_loss"])
        axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
        axes[1].set_title("Top-1 Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"curves_{backbone}.png"), dpi=150)
        plt.close(fig)
    except Exception as e:  # 绘图失败不阻断训练
        print(f"绘图失败: {e}")


if __name__ == "__main__":
    main()
