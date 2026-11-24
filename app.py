# -*- coding: utf-8 -*-
"""
Gradio 演示：图片上传 → Top-5 结果条形图 + 热量卡片 + 份量选择，并显示推理时延。
实现"上传—推理—展示"的端到端闭环。

用法：
    python app.py --backbone resnet50 --weight output/best_model_resnet50.pth
"""
import argparse
import functools
import os
import time

import gradio as gr
import torch

import config
from calorie import estimate_calories, load_calorie_table
from dataset import get_eval_transform
from models import load_classifier

# 全局模型资源，首次推理时懒加载
_MODEL = {"model": None, "device": None, "table": None}


def parse_args():
    p = argparse.ArgumentParser(description="Gradio 食物识别与热量估算演示")
    p.add_argument("--backbone", default="resnet50", choices=config.BACKBONES)
    p.add_argument("--weight", default="", help="权重路径，默认 output/best_model_{backbone}.pth")
    p.add_argument("--calorie-csv", default=config.CALORIE_CSV)
    p.add_argument("--device", default=None)
    p.add_argument("--server-name", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    return p.parse_args()


def ensure_model(args):
    if _MODEL["model"] is None:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weight = args.weight or os.path.join(config.OUTPUT_DIR, f"best_model_{args.backbone}.pth")
        if not os.path.exists(weight):
            raise FileNotFoundError(f"未找到权重 {weight}，请先运行 train.py")
        model = load_classifier(args.backbone, weight, config.NUM_CLASSES, device)
        _MODEL.update(model=model, device=device, table=load_calorie_table(args.calorie_csv))
    return _MODEL["model"], _MODEL["device"], _MODEL["table"]


def make_topk_chart(names, probs, name_cn_map):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{n}" + (f" ({name_cn_map.get(n, '')})" if name_cn_map.get(n) else "") for n in names]
    labels, probs = labels[::-1], probs[::-1]  # 让最高概率在最上方
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(labels, probs, color="#4C8CFF")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("概率")
    ax.set_title("Top-5 类别预测")
    for i, v in enumerate(probs):
        ax.text(v + 0.01, i, f"{v*100:.1f}%", va="center")
    fig.tight_layout()
    return fig


def run_demo(image, portion, grams, args):
    if image is None:
        return None, "请先上传一张食物图片。", ""
    model, device, table = ensure_model(args)
    t0 = time.perf_counter()
    tensor = get_eval_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu()
    topk = probs.topk(5)
    names = [config.FOOD_CLASSES[i] for i in topk.indices.tolist()]
    top_probs = topk.values.tolist()
    top1 = names[0]
    cal = estimate_calories(top1, probs.tolist(), config.FOOD_CLASSES,
                            portion=portion, grams=grams, table=table)
    latency = (time.perf_counter() - t0) * 1000
    name_cn_map = {c: table[c]["name_cn"] for c in config.FOOD_CLASSES}

    chart = make_topk_chart(names, top_probs, name_cn_map)
    text = (
        f"识别结果：{top1}（{name_cn_map[top1]}）\n"
        f"热量估计：{cal['kcal']} kcal（区间 {cal['kcal_low']} ~ {cal['kcal_high']} kcal）\n"
        f"份量：{cal['portion']}（约 {cal['grams_effective']} g）\n"
        f"蛋白 {cal['protein']} g · 脂肪 {cal['fat']} g · 碳水 {cal['carb']} g\n"
        f"数据来源：{cal['source']}"
    )
    return chart, text, f"{latency:.1f} ms"


def build_demo(args):
    with gr.Blocks(title="食物识别与卡路里估算系统") as demo:
        gr.Markdown("# 基于深度学习的食物识别与卡路里估算系统\n"
                    "上传一张食物图片，系统识别类别并估算热量。")
        with gr.Row():
            image_input = gr.Image(type="pil", label="食物图片", height=320)
            with gr.Column():
                portion = gr.Radio(list(config.PORTION_COEFFS), value=config.DEFAULT_PORTION,
                                   label="份量")
                grams = gr.Number(label="或输入克重（留空则用份量档位）", value=None, precision=1)
                btn = gr.Button("识别并估算热量", variant="primary")
        with gr.Row():
            chart = gr.Plot(label="Top-5 类别概率")
            calorie = gr.Textbox(label="热量估算结果", lines=6)
        latency = gr.Textbox(label="推理时延")
        handler = functools.partial(run_demo, args=args)
        btn.click(handler, [image_input, portion, grams], [chart, calorie, latency])
    return demo


if __name__ == "__main__":
    args = parse_args()
    demo = build_demo(args)
    demo.launch(server_name=args.server_name, server_port=args.port, share=args.share)
