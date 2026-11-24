# -*- coding: utf-8 -*-
"""
骨干网络封装：加载 ImageNet 预训练权重，替换分类头，
支持冻结/解冻骨干（阶段二可只解冻最后一个 stage）。
"""
import torch
import torch.nn as nn
import torchvision.models as tvmodels

import config

# 各骨干最后一个 stage 的参数名前缀（用于阶段二选择性解冻）
_BACKBONE_STAGE_PREFIX = {
    "resnet50": "layer4",
    "efficientnet_b0": "features.8",  # EfficientNet 最后的 MBConv 阶段
    "mobilenet_v3_large": "features.16",
}


def get_backbone(name, num_classes=None, pretrained=True):
    """返回 (backbone, in_features)。in_features 为原分类头输入维度。"""
    num_classes = num_classes or config.NUM_CLASSES
    if name == "resnet50":
        model = tvmodels.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        return model, in_features
    if name == "efficientnet_b0":
        model = tvmodels.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, in_features
    if name == "mobilenet_v3_large":
        model = tvmodels.mobilenet_v3_large(weights="IMAGENET1K_V1" if pretrained else None)
        # classifier = Linear(960,1280) -> Hardswish -> Dropout -> Linear(1280,1000)
        # 替换整个 classifier 会同时去掉 960->1280 投影，故特征维度取第一层 Linear 的输入 960
        in_features = model.classifier[0].in_features
        model.classifier = nn.Identity()
        return model, in_features
    raise ValueError(f"未知骨干网络: {name}，可选 {config.BACKBONES}")


class FoodClassifier(nn.Module):
    """backbone + 新分类头（含 Dropout）。"""

    def __init__(self, backbone_name, num_classes=None, pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes or config.NUM_CLASSES
        self.backbone, in_features = get_backbone(backbone_name, self.num_classes, pretrained)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, self.num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

    def set_backbone_trainable(self, trainable, stage_only=False):
        """冻结或解冻骨干。stage_only=True 时仅解冻最后一个 stage（阶段二微调）。"""
        prefix = _BACKBONE_STAGE_PREFIX[self.backbone_name]
        for name, p in self.backbone.named_parameters():
            if not trainable:
                p.requires_grad = False
            else:
                p.requires_grad = stage_only and name.startswith(prefix)
        return self


def load_classifier(backbone_name, weight_path, num_classes=None, device="cpu"):
    """加载保存的分类模型权重。"""
    model = FoodClassifier(backbone_name, num_classes)
    state = torch.load(weight_path, map_location=device)
    # 兼容保存 model 整体或仅 state_dict 两种格式
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
