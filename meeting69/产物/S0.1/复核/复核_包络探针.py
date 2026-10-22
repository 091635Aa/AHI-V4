# -*- coding: utf-8 -*-
"""附加探针 2：包络缺陷的频率后果（1 kHz 纯音的带扩散）+ 频带独立性检验。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

M69 = Path(r"<workspace>/03_AHI\AHI4\meeting69")
复核 = M69 / "产物" / "S0.1" / "复核"

s = importlib.util.spec_from_file_location("前", M69 / "实施" / "耳蜗前端_S0_1.py")
前 = importlib.util.module_from_spec(s); s.loader.exec_module(前)

α = 1.0 / (前.SR * 0.001)
n = int(0.4 * 前.SR)
t = np.arange(n) / 前.SR
音 = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
样本, fs = 前.滤波组(音)
包实 = 前.包络(样本)
包正 = lfilter([α], [1.0, -(1.0 - α)], np.maximum(样本, 0.0), axis=1)


def 廓(包):
    n1 = len(音) // 前.包络步
    b1 = 包[:, : n1 * 前.包络步].reshape(包.shape[0], n1, 前.包络步).mean(axis=2)
    c = np.power(np.maximum(b1, 0.0), 0.4)
    return c[:, 100:180].mean(axis=1)      # 稳态段（n1≈400）


廓实 = 廓(包实); 廓正 = 廓(包正)
出 = dict(
    频带中心Hz=[round(float(v), 1) for v in fs],
    实现_带廓=[round(float(v), 6) for v in 廓实],
    正确_带廓=[round(float(v), 6) for v in 廓正],
    实现_超10pct带数=int((廓实 > 0.1 * 廓实.max()).sum()),
    正确_超10pct带数=int((廓正 > 0.1 * 廓正.max()).sum()),
    实现_峰值带=int(np.argmax(廓实)), 正确_峰值带=int(np.argmax(廓正)),
    实现_峰值中心Hz=round(float(fs[int(np.argmax(廓实))]), 1),
    正确_峰值中心Hz=round(float(fs[int(np.argmax(廓正))]), 1),
    说明="1 kHz 纯音：正确实现应只点亮 ~1 kHz 邻近少数带；若实现下顶部带也高响应即为跨带泄漏",
)

# 定量：带 b 对带 b-k 的依赖（冲激式检验：只在带 40 放 1，其余 0）
测 = np.zeros((64, 400), np.float32); 测[40, :] = 1.0
y = 前.包络(测)
出["单带冲激响应"] = dict(
    从带40向上响应=[round(float(y[b].mean()), 6) for b in (40, 41, 45, 50, 55, 60, 63)],
    从带40向下响应=[round(float(y[b].mean()), 6) for b in (40, 39, 35, 30, 20, 0)],
    说明="若向上响应非零、向下为零，则确认递推沿带轴单向（低频→高频泄漏）",
)

# 频带间相关（稳态元音 场未适）：实现 vs 正确
g适 = next(x for x in json.loads((M69 / "产物" / "S0" / "manifest.json").read_text(encoding="utf-8"))["真值"]
          if x["类"] == "适应")
x2 = 前.读wav(g适["路径"])
样2, _ = 前.滤波组(x2)
包2实 = 前.包络(样2)
包2正 = lfilter([α], [1.0, -(1.0 - α)], np.maximum(样2, 0.0), axis=1)
n1 = len(x2) // 前.包络步


def 压缩(包):
    b1 = 包[:, : n1 * 前.包络步].reshape(包.shape[0], n1, 前.包络步).mean(axis=2)
    return np.power(np.maximum(b1, 0.0), 0.4)[:, 200:2200]


C实 = 压缩(包2实); C正 = 压缩(包2正)
出["稳态元音_带间相关"] = dict(
    实现_相邻带相关均值=round(float(np.mean([np.corrcoef(C实[i], C实[i + 1])[0, 1] for i in range(63)])), 4),
    正确_相邻带相关均值=round(float(np.mean([np.corrcoef(C正[i], C正[i + 1])[0, 1] for i in range(63)])), 4),
    实现_隔10带相关均值=round(float(np.mean([np.corrcoef(C实[i], C实[i + 10])[0, 1] for i in range(54)])), 4),
    正确_隔10带相关均值=round(float(np.mean([np.corrcoef(C正[i], C正[i + 10])[0, 1] for i in range(54)])), 4),
)

(复核 / "复核_包络探针.json").write_text(json.dumps(出, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in 出.items() if k not in ("实现_带廓", "正确_带廓", "频带中心Hz")},
                 ensure_ascii=False, indent=1))
