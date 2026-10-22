# -*- coding: utf-8 -*-
"""meeting38/实施/形状束.py —— 步8：图标形状描述子（原生视觉多模态的"向量团"）

━━━ 定位（所有者 2026-10-07 判定）━━━
「同色相近并不正确……向量团不是绝对的红色，而是一个**形状**」——图标身份 ≠ 颜色 = **结构**。

━━━ 结构统计式描述子（确定性、零训练、零 OCR、非 Transformer）━━━
  patch（元素框对齐裁剪）→ 缩放 64×64 灰度 → 自适应阈值（patch 自身 μ±0.6σ）切出
  **亮结构图/中结构图/暗结构图** 三张二值结构图（底色/亮度自适应——颜色不再是身份，
  图形/纹样才是）→ 每张做两档**平均池化**：
      8×8 全局池化（64 维：结构的细布局）＋ 2×2 象限池化（4 维：结构在哪个方位）；
  三层拼接 204 维结构统计 → 编码.特征码 → **形状束**（D 维双极）。
  平均池化天生容忍 ±数px 平移/框抖动；自适应阈值天生容忍底色/亮度漂移。
相似 = 编码.相似（余弦）。同一图标（平移/换底色/框微抖）→ 高相似；不同图标 → 低相似。

铁律：C1 禁 Transformer ｜ C9 禁 OCR（不读字符，只看结构统计）｜ C13 确定性。
"""
from __future__ import annotations

import cv2
import numpy as np

缩放 = 64                   # patch 统一缩放到 64×64（尺度归一）


def _池化(图, n: int) -> list:
    """n×n 平均池化（自适应容抖动的关键：池化后 ±2px 只动小数位）。"""
    h = 图.shape[0] // n
    return 图[:h * n, :h * n].reshape(n, h, n, h).mean(axis=(1, 3)).ravel().tolist()


class 形状描述子:
    def __init__(self, 编码):
        self.编码 = 编码

    def 形状束(self, patch) -> np.ndarray:
        """RGB uint8 patch（元素框对齐裁剪，任意尺寸）→ 形状束（D 维双极）。无结构 → None。"""
        p = np.asarray(patch)
        if p.ndim == 3 and p.shape[2] >= 3:
            灰 = cv2.cvtColor(p[:, :, :3], cv2.COLOR_RGB2GRAY)
        else:
            灰 = p
        if 灰.shape[0] < 12 or 灰.shape[1] < 12:
            return None
        灰 = cv2.resize(灰, (缩放, 缩放), interpolation=cv2.INTER_AREA).astype(np.float32)
        μ, σ = float(灰.mean()), float(灰.std())
        if σ < 4.0:
            return None                                     # 全平（无结构）
        特征 = []
        for 阈 in (μ - 0.6 * σ, μ, μ + 0.6 * σ):            # 暗结构 / 中间域 / 亮结构 三层
            二值 = (灰 > 阈).astype(np.float32)
            for n in (16, 8, 2):                             # 细布局 / 中布局 / 方位
                v = np.asarray(_池化(二值, n), dtype=np.float32)
                v = v - v.mean()                             # 层内去共模：消"暗底亮纹"的
                特征 += v.ravel().tolist()                   #  粗轮廓共性，纯留布局差异
        return self.编码.特征码(np.asarray(特征, dtype=np.float32))

    def 相似(self, a, b) -> float:
        """形状相似（余弦）。任一为 None → -1（未识别，不冒充相似）。"""
        if a is None or b is None:
            return -1.0
        return self.编码.相似(a, b)
