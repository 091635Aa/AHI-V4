# -*- coding: utf-8 -*-
"""meeting58 · Lead 基线探针：现状「角色复刻」的量化起点（只读既有件，产物落 meeting58）。

测什么：
  A. 现有 4 个音色档案（说_音色.音色档案）的 F0 / 谱重心 / 7 带 log 包络 / 声纹代理余弦；
  B. 两趟确定性（同输入同 seed 是否逐位同）；
  C. 一个**共振峰缩放子类**实验：不改 `声带.py`，只覆盖 `音节帧()`，看 7 带包络能否被拉开（E2 提案的最小验证）；
  D. 句子级「文本-音色错配」演示：洛天依的文本用男声-低沉念（现状调用点没有任何门禁）。

输出：`产物/基线/基线结果.json` + 控制台摘要。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # meeting58
AHI4 = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(AHI4, "meeting55", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting32"))

import 说_音色 as SY  # noqa: E402
import 声带 as SD  # noqa: E402

FS = 16000
产物 = os.path.join(ROOT, "产物", "基线")
os.makedirs(产物, exist_ok=True)
SY.产物 = 产物                                   # 把既有模块的落盘重定向到 meeting58（不写 meeting55）

文本 = "你好，我是天依。今天想听我唱歌吗？"
档案名 = list(SY.音色档案.keys())


def 特征(x: np.ndarray, sr: int = FS) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x - float(np.mean(x))
    能量 = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    F0 = SY._F0轨迹(x, sr)
    F0v = F0[np.isfinite(F0) & (F0 > 60)]
    f0 = float(np.median(F0v)) if len(F0v) else 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    频 = np.fft.rfftfreq(len(x), 1.0 / sr)
    重心 = float(np.sum(频 * spec) / (np.sum(spec) + 1e-12))
    带 = [(0, 250), (250, 500), (500, 1000), (1000, 2000),
          (2000, 3000), (3000, 4000), (4000, 8000)]
    包络 = []
    for lo, hi in 带:
        m = (频 >= lo) & (频 < hi)
        包络.append(float(np.log(np.mean(spec[m] ** 2) + 1e-12)))
    包络 = np.asarray(包络)
    包络 = 包络 - 包络.mean()
    return dict(F0中位=round(f0, 1), 谱重心=round(重心, 1),
                能量=round(能量, 5), 时长秒=round(len(x) / sr, 3),
                包络=包络)


def 余弦(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def 跑档案() -> dict:
    行 = {}
    for 名 in 档案名:
        t0 = time.perf_counter()
        rec = SY.说(文本, 音色=名, 情绪="平静")
        耗 = time.perf_counter() - t0
        f = 特征(rec["波形"])
        行[名] = dict(参数=rec["参数"], **{k: v for k, v in f.items() if k != "包络"},
                      包络=f["包络"].round(4).tolist(),
                      实时率=rec["实时率"], 探针耗时秒=round(耗, 3),
                      自检通过=rec["通过"])
    # 声纹代理余弦（7 带包络）
    对 = {}
    for i in range(len(档案名)):
        for j in range(i + 1, len(档案名)):
            a, b = 档案名[i], 档案名[j]
            对["%s|%s" % (a, b)] = round(余弦(np.array(行[a]["包络"]), np.array(行[b]["包络"])), 4)
    return dict(行=行, 两两包络余弦=对)


class 缩放声带(SD.声带器官):
    """共振峰缩放：只覆盖 音节帧()，不动既有件（C14）。"""

    def __init__(self, k: float = 1.0, **kw):
        super().__init__(目录=os.path.join(产物, "缩放发声"),
                         台账路径=os.path.join(产物, "缩放发声台账.jsonl"), **kw)
        self.k = float(k)

    def 韵律(self, 遥测: dict) -> dict:
        return dict(基频=215.0, 语速=3.6, 幅度=0.55, 气声度=0.15,
                    紧喉度=0.28, 粗糙度=0.05, 送气量=0.20, 声区=0.05)

    def 音节帧(self, 文本_: str, 律: dict, qs: np.ndarray) -> dict:
        d = super().音节帧(文本_, 律, qs)
        for i in range(len(d["f1"])):
            d["f1"][i] *= self.k
            d["f2"][i] *= self.k
            d["f3"][i] *= self.k
        return d


def 跑缩放() -> dict:
    结 = {}
    for k in (0.85, 1.0, 1.15):
        器 = 缩放声带(k=k, seed=300, D=1024)
        vec = np.sign(np.random.default_rng(55).standard_normal(1024)).astype(np.float32)
        rec = 器.发声(vec, 文本, dict(能量=0.5, 新异=0.5, 消化债=0.0, 饱足=0.5))
        f = 特征(rec["波形"])
        结["k=%.2f" % k] = dict(F0中位=f["F0中位"], 谱重心=f["谱重心"],
                                包络=f["包络"].round(4).tolist(),
                                自检=rec["自检"], 通过=rec["通过"],
                                时长秒=rec["时长秒"], 实时率=rec["实时率"])
    # 两两包络余弦
    对 = {}
    for a in 结:
        for b in 结:
            if a < b:
                对[a + "|" + b] = round(余弦(np.array(结[a]["包络"]), np.array(结[b]["包络"])), 4)
    return dict(结=结, 两两包络余弦=对)


def 跑确定性() -> dict:
    r1 = SY.说(文本, 音色="女声-明亮", 情绪="平静")
    r2 = SY.说(文本, 音色="女声-明亮", 情绪="平静")
    x1 = np.asarray(r1["波形"], dtype=np.float32)
    x2 = np.asarray(r2["波形"], dtype=np.float32)
    return dict(逐位同=bool(x1.shape == x2.shape and np.array_equal(x1, x2)),
                长度=len(x1), 发声号=[r1["发声号"], r2["发声号"]])


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    结果 = dict(文本=文本, 档案=跑档案(), 确定性=跑确定性(), 缩放实验=跑缩放())
    路径 = os.path.join(产物, "基线结果.json")
    with open(路径, "w", encoding="utf-8") as f:
        json.dump(结果, f, ensure_ascii=False, indent=1, default=str)
    print("=" * 60)
    print("基线：现有 4 音色档案（同文本）")
    for 名, r in 结果["档案"]["行"].items():
        print("  %-8s F0=%6.1fHz 谱重心=%7.1f 能量=%.4f 时长=%.2fs 实时率=%.3f 自检=%s"
              % (名, r["F0中位"], r["谱重心"], r["能量"], r["时长秒"], r["实时率"], r["自检通过"]))
    print("  声纹代理（7带包络）两两余弦：", 结果["档案"]["两两包络余弦"])
    print("-" * 60)
    print("确定性（两趟逐位同）：", 结果["确定性"]["逐位同"], "长度", 结果["确定性"]["长度"])
    print("-" * 60)
    print("共振峰缩放实验（子类覆盖 音节帧，不改既有件）")
    for k, r in 结果["缩放实验"]["结"].items():
        print("  %s F0=%6.1f 谱重心=%7.1f 时长=%.2fs 通过=%s" % (k, r["F0中位"], r["谱重心"], r["时长秒"], r["通过"]))
    print("  缩放后两两包络余弦：", 结果["缩放实验"]["两两包络余弦"])
    print("=" * 60)
    print("落盘：", 路径)
