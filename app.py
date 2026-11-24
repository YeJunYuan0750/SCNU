# -*- coding: utf-8 -*-
"""
食物识别与卡路里估算系统 —— Gradio 网页端。

启动：
    python app.py --backbone mobilenet_v3_large --weight D:/foodcal/models/best_model_mobilenet_v3_large.pth

页面结构：
    标签页 1「开始识别」：上传/拍照 → 份量选择 → Top-5 概率图 + 热量卡片 + 营养素 + 时延
    标签页 2「模型与指标」：骨干、验证/测试精度、混淆分析与训练曲线
    标签页 3「使用说明」：三步上手与误差来源说明
"""
import argparse
import functools
import json
import os
import time

import gradio as gr
import torch

import config
from calorie import estimate_calories, load_calorie_table
from dataset import get_eval_transform
from models import load_classifier

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.join(HERE, "examples")
CURVE_CANDIDATES = ("curves_{backbone}.png",)

_M = {"model": None, "device": None, "table": None, "args": None, "metrics": None}

CUSTOM_CSS = """
.gradio-container {max-width: 1180px !important; margin: 0 auto;}
#hero {background: linear-gradient(120deg,#1f3864 0%,#2e5c9a 100%); color:#fff;
       border-radius:14px; padding:22px 26px;}
#hero h1 {margin:0 0 6px 0; font-size:26px; color:#fff;}
#hero p {margin:0; opacity:.9; font-size:14px;}
.kcal-card {border:1px solid #e3e8f0; border-radius:12px; padding:16px 18px; background:#fbfcfe;}
.kcal-big {font-size:40px; font-weight:700; color:#1f3864; line-height:1.1;}
.kcal-sub {color:#5b6b86; font-size:13px;}
.tag {display:inline-block; background:#eef3fc; color:#2e5c9a; border-radius:999px;
      padding:3px 10px; font-size:12px; margin-right:6px;}
.muted {color:#7a869a; font-size:12px;}
"""

# Gradio 6 起 css / theme 由 Blocks 构造器移到 launch()，这里按版本兼容
GRADIO_MAJOR = int(gr.__version__.split(".")[0])
BLOCK_KW = {} if GRADIO_MAJOR >= 6 else {"css": CUSTOM_CSS, "theme": gr.themes.Soft()}
LAUNCH_KW = {"css": CUSTOM_CSS, "theme": gr.themes.Soft()} if GRADIO_MAJOR >= 6 else {}


def parse_args():
    p = argparse.ArgumentParser(description="Gradio 食物识别与热量估算网页端")
    p.add_argument("--backbone", default=os.environ.get("FOOD_BACKBONE", "mobilenet_v3_large"),
                   choices=config.BACKBONES)
    p.add_argument("--weight", default=os.environ.get("FOOD_WEIGHT", ""))
    p.add_argument("--calorie-csv", default=config.CALORIE_CSV)
    p.add_argument("--device", default=None)
    p.add_argument("--server-name", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    p.add_argument("--no-browser", action="store_true")
    return p.parse_args()


def weight_path(args):
    if args.weight:
        return args.weight
    return os.path.join(config.OUTPUT_DIR, f"best_model_{args.backbone}.pth")


def load_metrics(args):
    # 权重文件所在目录优先（训练时 --output-dir 通常与权重同目录），
    # 其次是 config.OUTPUT_DIR，这样即使把权重复制到别处也能读到对应指标。
    cands = []
    try:
        cands.append(os.path.join(os.path.dirname(os.path.abspath(weight_path(args))),
                                  f"results_{args.backbone}.json"))
    except Exception:
        pass
    cands.append(os.path.join(config.OUTPUT_DIR, f"results_{args.backbone}.json"))
    for path in cands:
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return None


def ensure_model(args):
    if _M["model"] is None:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        path = weight_path(args)
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到权重文件：{path}\n请先运行 train.py 训练模型。")
        _M["model"] = load_classifier(args.backbone, path, config.NUM_CLASSES, device)
        _M["device"] = device
        _M["table"] = load_calorie_table(args.calorie_csv)
        _M["args"] = args
        _M["metrics"] = load_metrics(args)
    return _M["model"], _M["device"], _M["table"]


def make_topk_chart(names, probs, name_cn_map):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    labels = [f"{name_cn_map.get(n, n)}" for n in names]
    labels, probs = labels[::-1], probs[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=110)
    bars = ax.barh(labels, probs, color="#4C8CFF", height=0.62)
    top = bars[-1]
    top.set_color("#1f3864")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("预测概率", fontsize=9)
    ax.tick_params(labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for i, v in enumerate(probs):
        ax.text(min(v + 0.015, 0.92), i, f"{v * 100:.1f}%", va="center", fontsize=9, color="#33475b")
    fig.tight_layout()
    return fig


def card_html(top1, top1_cn, top1_prob, cal):
    tags = "".join(f'<span class="tag">{t}</span>' for t in
                   [f"{top1_prob * 100:.1f}% 置信", f"份量 {cal['portion']}", f"约 {cal['grams_effective']} g"])
    return f"""
    <div class="kcal-card">
      <div class="muted">识别结果</div>
      <div style="font-size:22px;font-weight:600;color:#1f3864;margin:2px 0 8px;">{top1_cn}
        <span class="muted" style="font-weight:400;">（{top1}）</span></div>
      <div class="kcal-big">{cal['kcal']} <span style="font-size:16px;font-weight:500;">kcal</span></div>
      <div class="kcal-sub">区间 {cal['kcal_low']} ~ {cal['kcal_high']} kcal（概率分布加权，约 95%）</div>
      <div style="margin-top:12px;font-size:14px;color:#33475b;">
        蛋白质 <b>{cal['protein']} g</b> · 脂肪 <b>{cal['fat']} g</b> · 碳水 <b>{cal['carb']} g</b>
      </div>
      <div style="margin-top:10px;">{tags}</div>
      <div class="muted" style="margin-top:10px;">营养成分来源：{cal['source']}；数值为参考估算，不作为医学或临床营养依据。</div>
    </div>
    """


def run(image, portion, grams, args=None):
    if image is None:
        return None, "<div class='kcal-card'><div class='muted'>请先上传一张食物图片，或从示例中选一张。</div></div>", "—"
    # 克重输入框在 Gradio 6 里默认值是 0 而不是 None，必须自己判空，
    # 否则 0/100 会把份量系数算成 0，热量与营养素全变成 0。
    try:
        grams = float(grams) if grams not in (None, "") else None
    except (TypeError, ValueError):
        grams = None
    if grams is not None and grams <= 0:
        grams = None
    args = args or _M.get("args") or _M["pending_args"]
    model, device, table = ensure_model(args)
    t0 = time.perf_counter()
    tensor = get_eval_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu()
    topk = probs.topk(5)
    names = [config.FOOD_CLASSES[i] for i in topk.indices.tolist()]
    values = topk.values.tolist()
    latency = (time.perf_counter() - t0) * 1000

    cal = estimate_calories(names[0], probs.tolist(), config.FOOD_CLASSES,
                            portion=portion, grams=grams, table=table)
    name_cn = {c: table[c]["name_cn"] for c in config.FOOD_CLASSES}
    chart = make_topk_chart(names, values, name_cn)
    html = card_html(names[0], name_cn[names[0]], values[0], cal)
    return chart, html, f"{latency:.0f} ms（单张，{device.type.upper()}）"


def metrics_markdown(args):
    m = _M["metrics"] or load_metrics(args)
    if not m:
        return "_训练结果尚未生成：运行 `python train.py` 后此处会显示真实指标。_"
    rep = m.get("classification_report", {})
    acc = m.get("test_acc", 0) * 100
    top5 = m.get("test_top5", 0) * 100
    macro_f1 = rep.get("macro avg", {}).get("f1-score", 0) * 100
    lines = [
        f"**骨干网络**：`{m.get('backbone', args.backbone)}`　**类别数**：{config.NUM_CLASSES}",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 测试集 Top-1 准确率 | {acc:.2f}% |",
        f"| 测试集 Top-5 准确率 | {top5:.2f}% |",
        f"| 宏平均 F1 | {macro_f1:.2f}% |",
    ]
    return "\n".join(lines)


def examples_list():
    if not os.path.isdir(EXAMPLE_DIR):
        return None
    files = sorted(f for f in os.listdir(EXAMPLE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return [os.path.join(EXAMPLE_DIR, f) for f in files][:12] or None


def build_demo(args):
    _M["pending_args"] = args
    _M["metrics"] = load_metrics(args)
    weight_exists = os.path.exists(weight_path(args))
    status = ("模型已就绪" if weight_exists else "未找到权重：请先训练")

    with gr.Blocks(title="食物识别与卡路里估算系统", fill_width=True, **BLOCK_KW) as demo:
        gr.HTML(f"""
        <div id="hero">
          <h1>食物识别与卡路里估算系统</h1>
          <p>上传一张食物照片 · 自动识别类别 · 估算热量与营养素 · 给出概率区间（{status}）</p>
        </div>
        """)

        with gr.Tab("开始识别"):
            with gr.Row():
                with gr.Column(scale=5):
                    image_input = gr.Image(type="pil", label="食物图片（支持上传或摄像头）",
                                           sources=["upload", "webcam"], height=330)
                    with gr.Row():
                        portion = gr.Radio(list(config.PORTION_COEFFS), value=config.DEFAULT_PORTION,
                                           label="份量档位", scale=3)
                        grams = gr.Number(label="或直接输入克重（选填）", value=None, precision=0, scale=2)
                    btn = gr.Button("识别并估算热量", variant="primary", size="lg")
                    ex = examples_list()
                    if ex:
                        gr.Examples(examples=ex, inputs=image_input, label="示例图片（来自测试集）")
                with gr.Column(scale=6):
                    result_html = gr.HTML("<div class='kcal-card'><div class='muted'>结果会显示在这里</div></div>")
                    chart = gr.Plot(label="Top-5 类别概率", show_label=False)
                    latency = gr.Textbox(label="推理时延", interactive=False)
            handler = functools.partial(run)
            btn.click(handler, [image_input, portion, grams], [chart, result_html, latency])
            image_input.change(handler, [image_input, portion, grams], [chart, result_html, latency])

        with gr.Tab("模型与指标"):
            gr.Markdown("### 实验指标")
            metrics_box = gr.Markdown(metrics_markdown(args))
            curve_png = os.path.join(config.OUTPUT_DIR, f"curves_{args.backbone}.png")
            if os.path.exists(curve_png):
                gr.Markdown("### 训练曲线")
                gr.Image(curve_png, label="损失与准确率曲线", show_label=False)
            gr.Markdown(
                "### 评测口径\n"
                "- 训练/验证/测试 = 70% / 15% / 15%，按类别分层抽样，随机种子 42\n"
                "- 两阶段迁移学习：先冻结骨干训练分类头，再解冻最后一个 stage 微调\n"
                "- 模型选择只看验证集 Top-1，测试集仅在最后评估一次"
            )

        with gr.Tab("使用说明"):
            gr.Markdown(
                """
### 三步上手
1. **上传图片**：单张、只含一种主食物，尽量正面拍摄、光线均匀；也可以点「示例图片」直接体验。
2. **选择份量**：小份 / 中份 / 大份三档（系数 0.7 / 1.0 / 1.3），或直接输入估计克重。
3. **读取结果**：左侧是 Top-5 类别概率，右侧是热量与蛋白质、脂肪、碳水。

### 结果怎么理解
- **热量**是 30 个类别概率的加权期望，比只看 Top-1 更稳；**区间**是概率分布的加权标准差 ×1.96。
- **置信度低于 60%** 时，说明图片可能不在 30 类范围内，或拍摄角度、遮挡影响了判断，建议换一张。
- 营养素按 Top-1 类别换算：营养结构属于类别的固有属性，按概率加权反而会混掉不同菜品。

### 已知限制
- 只处理**单张图片中的单一主食物**，不做多目标检测与食材分解；
- 份量靠人工选择，热量误差同时包含"识别错误"和"份量假设误差"两部分；
- 热量表取自 USDA FoodData Central 与《中国食物成分表（第 6 版）》，属参考性估算。
                """
            )

    return demo


if __name__ == "__main__":
    args = parse_args()
    demo = build_demo(args)
    demo.queue().launch(server_name=args.server_name, server_port=args.port,
                        share=args.share, inbrowser=not args.no_browser,
                        show_error=True, allowed_paths=[config.OUTPUT_DIR], **LAUNCH_KW)
