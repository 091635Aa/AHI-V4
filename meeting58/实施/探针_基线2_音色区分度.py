# -*- coding: utf-8 -*-
"""meeting58 · Lead 基线探针 2：音色区分度到底来自哪里（16 带包络，细粒度）。

基线 1 已发现：现有 4 档的 7 带包络余弦 0.89–0.99（近似不可分）。本探针继续拆：
  A. 16 带 log 包络（0–8k）两两余弦：4 档档案 + 共振峰缩放 k∈{0.7,0.85,1.0,1.15,1.3}；
  B. 单参数敏感度：只改 气声度 / 紧喉度 / 声区（基频固定），看 16 带包络变化多大；
  C. 只改 F0（±20%）看包络变化 —— 判断「音色」是否其实只是「音高」；
  D. 结论口径：哪些量能当「声纹代理」，哪些不能。

只读既有件；产物落 meeting58/产物/基线2。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AHI4 = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(AHI4, "meeting55", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting32"))

import 说_音色 as SY  # noqa: E402
import 声带 as SD  # noqa: E402

FS = 16000
产物 = os.path.join(ROOT, "产物", "基线2")
os.makedirs(产物, exist_ok=True)
SY.产物 = 产物

文本 = "你好，我是天依。今天想听我唱歌吗？"
带 = [(0, 125), (125, 250), (250, 375), (375, 500), (500, 750), (750, 1000),
      (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000), (3000, 3500),
      (3500, 4000), (4000, 5000), (5000, 6000), (6000, 7000), (7000, 8000)]


def 包络16(x: np.ndarray, sr: int = FS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    频 = np.fft.rfftfreq(len(x), 1.0 / sr)
    e = []
    for lo, hi in 带:
        m = (频 >= lo) & (频 < hi)
        e.append(float(np.log(np.mean(spec[m] ** 2) + 1e-12)))
    e = np.asarray(e)
    return e - e.mean()


def 余弦(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def 矩阵(名到包络: dict) -> dict:
    名 = list(名到包络)
    对 = {}
    for i in range(len(名)):
        for j in range(i + 1, len(名)):
            对["%s|%s" % (名[i], 名[j])] = round(余弦(名到包络[名[i]], 名到包络[名[j]]), 4)
    return 对


class 缩放声带(SD.声带器官):
    """共振峰缩放 + 带宽缩放（只覆盖 音节帧/谐振器系数不碰既有件；带宽缩放用 3 段近似）。"""

    def __init__(self, k: float = 1.0, 律: dict = None, **kw):
        super().__init__(目录=os.path.join(产物, "发声"),
                         台账路径=os.path.join(产物, "发声台账.jsonl"), **kw)
        self.k = float(k)
        self._律 = 律 or dict(基频=215.0, 语速=3.6, 幅度=0.55, 气声度=0.15,
                              紧喉度=0.28, 粗糙度=0.05, 送气量=0.20, 声区=0.05)

    def 韵律(self, 遥测: dict) -> dict:
        return dict(self._律)

    def 音节帧(self, 文本_: str, 律: dict, qs: np.ndarray) -> dict:
        d = super().音节帧(文本_, 律, qs)
        for i in range(len(d["f1"])):
            d["f1"][i] *= self.k
            d["f2"][i] *= self.k
            d["f3"][i] *= self.k
        return d


def 发(器, 向量=None):
    vec = np.asarray(向量 if 向量 is not None else
                     np.sign(np.random.default_rng(55).standard_normal(1024)),
                     dtype=np.float32)
    return 器.发声(vec, 文本, dict(能量=0.5, 新异=0.5, 消化债=0.0, 饱足=0.5))


def 跑() -> dict:
    # A. 4 档案（走既有 说_音色，含情绪偏移=平静）
    档 = {}
    for 名 in SY.音色档案:
        rec = SY.说(文本, 音色=名, 情绪="平静")
        档[名] = 包络16(rec["波形"])
    # A2. 共振峰缩放（同一 8 参数=女声-明亮）
    缩放 = {}
    律 = dict(SY.音色档案["女声-明亮"]); 律["_ñ"] = 0.5
    for k in (0.7, 0.85, 1.0, 1.15, 1.3):
        器 = 缩放声带(k=k, 律=律, seed=300, D=1024)
        rec = 发(器)
        缩放["k=%.2f" % k] = 包络16(rec["波形"])
    # B. 单参数敏感度（基频固定 215）
    敏感 = {}
    for 参, 值列 in (("气声度", (0.10, 0.35, 0.60)), ("紧喉度", (0.10, 0.35, 0.60)),
                     ("声区", (0.02, 0.25, 0.50)), ("基频", (172.0, 215.0, 258.0))):
        for v in 值列:
            律 = dict(SY.音色档案["女声-明亮"]); 律[参] = v; 律["_ñ"] = 0.5
            器 = 缩放声带(k=1.0, 律=律, seed=300, D=1024)
            rec = 发(器)
            敏感["%s=%.2f" % (参, v)] = 包络16(rec["波形"])
    return dict(档案=dict(矩阵=矩阵(档), 包络={k: v.round(4).tolist() for k, v in 档.items()}),
                共振峰缩放=dict(矩阵=矩阵(缩放), 包络={k: v.round(4).tolist() for k, v in 缩放.items()}),
                单参数=dict(矩阵=矩阵(敏感), 包络={k: v.round(4).tolist() for k, v in 敏感.items()}))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    结 = 跑()
    路径 = os.path.join(产物, "基线2结果.json")
    with open(路径, "w", encoding="utf-8") as f:
        json.dump(结, f, ensure_ascii=False, indent=1)
    for 段 in ("档案", "共振峰缩放", "单参数"):
        print("=" * 70)
        print("段：", 段)
        for k, v in 结[段]["矩阵"].items():
            print("  %-32s cos=%s" % (k, v))
    print("落盘：", 路径)
