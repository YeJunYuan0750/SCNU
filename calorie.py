# -*- coding: utf-8 -*-
"""
热量映射模块。

识别结果（类别概率分布）与份量假设被转换为带区间的热量估计：
    E[热量] = Σ_i p_i · kcal_i · 份量系数 · (克重/100)
区间由概率分布下的加权标准差给出（约 95% 置信），蛋白/脂肪/碳水
取自 Top-1 类别按份量系数换算后的数值。
"""
import csv
import math
import os

import config


def load_calorie_table(path=None):
    """读取 calorie_table.csv，返回 {class: {energy, protein, fat, carb, name_cn, source}}。"""
    path = path or config.CALORIE_CSV
    table = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            table[row["class"]] = {
                "energy": float(row["energy_kcal_100g"]),
                "protein": float(row["protein_g_100g"]),
                "fat": float(row["fat_g_100g"]),
                "carb": float(row["carb_g_100g"]),
                "name_cn": row["name_cn"],
                "source": row["source"],
            }
    return table


def _portion_factor(portion, grams):
    """返回份量系数：优先使用显式克重，否则使用三档系数（相对 100g）。"""
    if grams is not None:
        return grams / 100.0
    coeffs = config.PORTION_COEFFS
    if portion not in coeffs:
        raise ValueError(f"未知份量档位: {portion}，可选 {list(coeffs)}")
    return coeffs[portion]


def estimate_calories(class_name, probs, class_names, portion=None, grams=None,
                      table=None):
    """
    计算热量估计。

    Args:
        class_name: Top-1 类别名。
        probs: 与 class_names 对齐的完整概率向量（softmax 输出）。
        class_names: 概率向量对应的类别名列表。
        portion: 份量档位（small/medium/large），grams 为 None 时生效。
        grams: 显式克重；若提供则覆盖份量系数。
        table: 热量表 dict，默认从 CSV 加载。

    Returns:
        dict: kcal, kcal_low, kcal_high, protein, fat, carb, top1, top1_prob,
              portion, grams_effective, unit。
    """
    portion = portion or config.DEFAULT_PORTION
    table = table or load_calorie_table()
    factor = _portion_factor(portion, grams)

    # 概率分布下的期望热量（每 100g 基准）与加权标准差
    energies = [table[c]["energy"] for c in class_names]
    probs = list(probs)
    exp_100g = sum(p * e for p, e in zip(probs, energies))
    var_100g = sum(p * (e - exp_100g) ** 2 for p, e in zip(probs, energies))
    std_100g = math.sqrt(max(var_100g, 0.0))

    # 带区间的主估计（以概率加权期望为准，区间由标准差给出）
    kcal = exp_100g * factor
    k = config.STD_INTERVAL_K
    kcal_low = max(0.0, (exp_100g - k * std_100g) * factor)
    kcal_high = (exp_100g + k * std_100g) * factor

    top1 = table[class_name]
    return {
        "kcal": round(kcal, 1),
        "kcal_low": round(kcal_low, 1),
        "kcal_high": round(kcal_high, 1),
        "protein": round(top1["protein"] * factor, 1),
        "fat": round(top1["fat"] * factor, 1),
        "carb": round(top1["carb"] * factor, 1),
        "top1": class_name,
        "top1_name_cn": top1["name_cn"],
        "top1_prob": round(float(probs[class_names.index(class_name)]), 4),
        "portion": portion,
        "grams_effective": round(factor * 100.0, 1),
        "source": top1["source"],
    }


def mean_absolute_error(estimated_kcal, reference_kcal):
    """平均绝对误差 MAE = (1/N)Σ|ŷᵢ − yᵢ|。"""
    if not estimated_kcal:
        return float("nan")
    return sum(abs(e - r) for e, r in zip(estimated_kcal, reference_kcal)) / len(estimated_kcal)
