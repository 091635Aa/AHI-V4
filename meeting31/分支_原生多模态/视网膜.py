# -*- coding: utf-8 -*-
"""分支_原生多模态/视网膜.py —— #279 原生视觉通路（无 OCR）。

像素→HDC 向量：帧降采样→16×12 补丁灰度→随机投影 HDC（确定性 seed）
帧差→变化度（运动/页面响应检测）——机器人同构的感知前庭。
不解析文字（那是文本模态的事）：视觉向量与页面标题共现绑定=『看到』。
"""
from __future__ import annotations

import numpy as np

# 随机投影矩阵：确定性 seed 777，模块级只建一次（实时性：每帧仅一次矩阵乘）
# RGB 三通道补丁（32×24=768 补丁×3=2304 维）——64×48 太粗：标签栏/文字级变化被补丁均值抹掉
_rng = np.random.default_rng(777)
投影 = _rng.standard_normal((32 * 24 * 3, 256)).astype(np.float32)


def 网格(img) -> np.ndarray:
    """PIL Image → 32×24 RGB 补丁均值网格（24×32×3=2304 维，保留空间结构）。"""
    a = np.asarray(img.convert("RGB").resize((128, 96)), dtype=np.float32) / 255.0
    return a.reshape(96 // 4, 4, 128 // 4, 4, 3).mean(axis=(1, 3))


def 帧向量(img) -> np.ndarray:
    """PIL Image → 补丁网格 → 随机投影 HDC 向量（256 维浮点）。"""
    return 网格(img).flatten() @ 投影


def 变化度(v1: np.ndarray, v2: np.ndarray) -> float:
    """帧间余弦距离——页面响应/运动检测（实时反馈信号）。"""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(1.0 - float(np.dot(v1, v2)) / (n1 * n2))


def 相似度(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def 区域差分(g1: np.ndarray, g2: np.ndarray):
    """两个补丁网格（24×32×3）逐区求差，返回 (变化最大的补丁行,列, 变化量)。

    『哪里变了』无需额外编码——网格本身保留空间结构。
    """
    d = np.linalg.norm(g1 - g2, axis=2)
    r, c = np.unravel_index(int(np.argmax(d)), d.shape)
    return int(r), int(c), float(d[r, c])
