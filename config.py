# -*- coding: utf-8 -*-
"""
全局配置：类别、路径、训练超参数、热量映射常量。
所有脚本统一从这里读取配置，保证口径一致。
"""
import os

# ---------------------------------------------------------------
# 路径
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")          # 训练/验证/测试图片目录
OUTPUT_DIR = os.path.join(BASE_DIR, "output")      # 权重、曲线、日志
CALORIE_CSV = os.path.join(BASE_DIR, "calorie_table.csv")

# ---------------------------------------------------------------
# 类别：Food-101 中的 30 个常见类别（按居民膳食频次 + 可获取性筛选）
# ---------------------------------------------------------------
FOOD_CLASSES = [
    "pizza", "hamburger", "sushi", "fried_rice", "dumplings",
    "ramen", "steak", "chicken_wings", "french_fries", "hot_dog",
    "spaghetti_bolognese", "spaghetti_carbonara", "lasagna", "pancakes", "waffles",
    "donuts", "ice_cream", "chocolate_cake", "cheesecake", "apple_pie",
    "caesar_salad", "greek_salad", "grilled_salmon", "peking_duck", "spring_rolls",
    "bibimbap", "pad_thai", "tacos", "gyoza", "omelette",
]
NUM_CLASSES = len(FOOD_CLASSES)

# ---------------------------------------------------------------
# 可选骨干网络（torchvision 名称）
# ---------------------------------------------------------------
BACKBONES = ["resnet50", "efficientnet_b0", "mobilenet_v3_large"]

# ---------------------------------------------------------------
# 图像与数据增强
# ---------------------------------------------------------------
IMAGE_SIZE = 224          # 训练/推理输入尺寸
RESIZE_SIZE = 256         # 评估阶段的先缩放尺寸
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 训练阶段增强强度（轻度，避免破坏食物细粒度纹理）
AUG_HFLIP_PROB = 0.5
AUG_COLOR_JITTER = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
AUG_ROTATION_DEG = 10

# ---------------------------------------------------------------
# 训练超参数（两阶段迁移学习）
# ---------------------------------------------------------------
TRAIN_SPLIT = 0.70        # 70/15/15 划分
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

STAGE1_EPOCHS = 5         # 阶段一：冻结骨干，仅训练分类头
STAGE2_EPOCHS = 15        # 阶段二：解冻最后阶段，小学习率微调
STAGE1_LR = 1e-3
STAGE2_LR = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
NUM_WORKERS = 2
LABEL_SMOOTHING = 0.0     # 可选标签平滑
USE_CLASS_WEIGHTS = False # 可选类别权重

# ---------------------------------------------------------------
# 热量映射
# ---------------------------------------------------------------
# 份量三档系数（相对标准份量 100g）
PORTION_COEFFS = {"small": 0.7, "medium": 1.0, "large": 1.3}
DEFAULT_PORTION = "medium"
DEFAULT_GRAMS = None       # None 表示使用份量系数；否则按输入克重换算
STD_INTERVAL_K = 1.96      # 热量区间用加权标准差 × k 给出（约 95% 置信）
