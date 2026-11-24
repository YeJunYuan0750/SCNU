# 基于深度学习的食物识别与卡路里估算系统

《人工智能导论》课程项目。系统实现“食物图片 → 类别识别 → 热量估算”的端到端闭环：
从 Food-101 选取 30 类常见食物，用迁移学习训练分类模型，结合自建热量表与份量选择输出
带区间的热量估计，并通过 Gradio 网页提供可交互演示。

## 功能特性

- **数据准备**：Food-101 子集（30 类）、70/15/15 分层划分、数据增强
- **迁移学习**：ResNet50 / EfficientNet-B0 / MobileNetV3-Large 三骨干、两阶段训练（冻结骨干→解冻微调）
- **热量映射**：概率分布加权期望 + 加权标准差区间、份量三档（0.7/1.0/1.3）或显式克重
- **系统集成**：`predict.py` 单图/批量推理、`app.py` Gradio 演示（Top-5 条形图 + 热量卡片 + 时延）
- **评估指标**：Top-1/Top-5 准确率、推理时延、热量 MAE、混淆矩阵与分类报告

## 当前状态

| 环节 | 状态 |
| --- | --- |
| 代码框架（10 个模块、约 1 000 行） | 已完成 |
| 数据管线（下载/筛选/分层划分/增强） | 已完成 |
| 热量知识库（30 类、逐条注明来源） | 已完成 |
| 训练与评估脚本（两阶段、曲线与指标导出） | 已完成 |
| 推理、热量映射与 Gradio 界面 | 已完成 |
| 端到端自检（`smoke_test.py`，合成数据） | 通过 |
| 真实数据训练与 E1—E7 对比实验 | 进行中，指标待补充 |

> 仓库当前不含训练好的权重与数据集（体积大）。下方“快速开始”可在本地完整复现；
> 实验结果表将在训练完成后补充到本文件。

## 目录结构

```
food-calorie-system/
├── config.py            全局配置（类别、路径、超参数、份量系数）
├── calorie_table.csv    热量表：30 类食物每 100g 能量/蛋白/脂肪/碳水（逐条注明来源）
├── dataset.py           数据集与数据增强
├── models.py            骨干网络封装（替换分类头、冻结/解冻）
├── calorie.py           热量映射模块（期望/区间/MAE）
├── train.py             两阶段迁移学习训练 + 评估
├── predict.py           单图/批量推理 + 热量 + 时延
├── app.py               Gradio 演示
├── prepare_data.py      Food-101 子集下载与划分
├── smoke_test.py        端到端自检（合成数据，数分钟跑通全链路）
├── docs/                开题报告、进展报告与技术路线图
├── requirements.txt
└── LICENSE
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载 Food-101 并划分 30 类子集（约 5GB，仅首次需要）
python prepare_data.py --download

# 3. 训练（以 MobileNetV3 为例，CPU 亦可；GPU 更快）
python train.py --backbone mobilenet_v3_large

# 4. 单图推理
python predict.py --image data/test/pizza/xxx.jpg \
    --backbone mobilenet_v3_large --weight output/best_model_mobilenet_v3_large.pth

# 5. 启动网页演示
python app.py --backbone mobilenet_v3_large \
    --weight output/best_model_mobilenet_v3_large.pth
```

浏览器打开 `http://127.0.0.1:7860`，上传图片即可查看识别结果与热量估算。

改完代码后建议先跑一次端到端自检（不需要真实数据，CPU 上数分钟完成）：

```bash
python smoke_test.py
```

## 文档

| 文档 | 说明 |
| --- | --- |
| [开题报告](docs/基于深度学习的食物识别与卡路里估算系统-开题报告.docx) | 选题背景、技术路线、16 周进度与验收指标 |
| [进展报告](docs/基于深度学习的食物识别与卡路里估算系统-进展报告.docx) | 技术方案、实验方案（E1—E7）、目前进展与待解决问题 |
| [系统架构图](docs/figures/system_arch.png) | 训练链路与推理链路 |

## 关键配置

- 类别列表与份量系数见 `config.py`（`FOOD_CLASSES`、`PORTION_COEFFS`）
- 训练超参数（两阶段 epoch/学习率/批大小）见 `config.py`
- 热量表数值来源见 `calorie_table.csv` 的 `source` 列

## 说明

- 训练默认在 GPU 可用时自动使用 GPU，否则退化为 CPU（MobileNetV3 在 CPU 上单张推理为
  几十毫秒量级，可满足网页交互）。
- 热量区间由分类概率分布下的加权标准差 × 1.96 给出（约 95% 置信），蛋白/脂肪/碳水取自
  Top-1 类别按份量系数换算后的数值。
- 若本地无独立显卡，建议使用 MobileNetV3 并适当减小输入分辨率或使用云端免费 GPU。
- 热量数值为参考性估算，不作为医学或临床营养依据；热量表仅引用公开数据（USDA FoodData
  Central、《中国食物成分表（第 6 版）》）的数值，未复制其原表版式。

## 致谢

- 数据集：[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)（Bossard et al., ECCV 2014）
- 预训练骨干：[torchvision](https://pytorch.org/vision/stable/index.html)
- 交互界面：[Gradio](https://www.gradio.app/)
- 营养数据：USDA FoodData Central、《中国食物成分表（第 6 版）》

## 许可协议

本项目采用 [MIT License](LICENSE) 开源。数据集与预训练权重遵循其原始许可。
