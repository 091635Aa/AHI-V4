# -*- coding: utf-8 -*-
"""meeting58/实施/探针_深度优化.py —— 声带.py 接口改造 + 深度优化的实测（改造后）。

测：
  A. 构造耗时（数轴缓存）与实时率（分块 IIR）——改造前后对比；
  B. 新音色路径的确定性；
  C. 死参数接线：紧喉/粗糙/幅度/送气 是否真的改变波形（旧路径全部无效或弱）；
  D. 共振峰缩放/带宽缩放/谱倾斜 是否可测；
  E. 快路径 vs 旧路径的差异（不应相同；旧路径保持原样）。
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M58 = os.path.dirname(HERE)
AHI4 = os.path.dirname(M58)
sys.path.insert(0, os.path.join(AHI4, "meeting32"))
sys.path.insert(0, HERE)

import 声带 as SD  # noqa: E402

FS = SD.FS
产物 = os.path.join(M58, "产物", "深度优化")
os.makedirs(产物, exist_ok=True)
文本 = "你好，我是天依。今天想听我唱歌吗？"
带 = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 3000),
      (3000, 4000), (4000, 8000)]


def 包络7(x):
    x = np.asarray(x, float)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    频 = np.fft.rfftfreq(len(x), 1.0 / FS)
    e = [np.log(np.mean(spec[(频 >= lo) & (频 < hi)] ** 2) + 1e-12) for lo, hi in 带]
    e = np.asarray(e)
    return e - e.mean()


def 余弦(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def F0中位(x):
    x = np.asarray(x, float)
    f0 = []
    n = max(1, (len(x) - int(0.04 * FS)) // int(0.01 * FS))
    for i in range(n):
        seg = x[i * int(0.01 * FS): i * int(0.01 * FS) + int(0.04 * FS)]
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
        lo, hi = int(FS / 500), int(FS / 70)
        if hi >= len(ac):
            continue
        段 = ac[lo:hi]
        mx = float(np.max(段))
        cand = [j for j in range(1, len(段) - 1)
                if 段[j] >= 0.88 * mx and 段[j] >= 段[j - 1] and 段[j] >= 段[j + 1]]
        if cand:
            f0.append(FS / (lo + cand[0]))
    return float(np.median(f0)) if f0 else 0.0


def 基线音色():
    return dict(基频=265.0, 语速=3.7, 幅度=0.55, 气声度=0.20, 紧喉度=0.26,
                粗糙度=0.04, 送气量=0.20, 声区=0.10,
                共振峰缩放=(1.14, 1.18, 1.22), 带宽缩放=1.0, 谱倾斜=0.30,
                送气增益=0.0, 增益=1.0)


def 发(器, 音色, 遥测=None, 文本_=文本, 写wav=False):
    vec = np.sign(np.random.default_rng(55).standard_normal(1024)).astype(np.float32)
    rec = 器.发声(vec, 文本_, 遥测 or dict(能量=0.5, 新异=0.5, 消化债=0.0, 饱足=0.5),
                  音色=音色)
    return rec


def 跑():
    结 = {}
    # A 构造耗时 / 实时率
    t0 = time.perf_counter()
    for _ in range(3):
        器 = SD.声带器官(seed=300, D=1024, 目录=os.path.join(产物, "发声"),
                         台账路径=os.path.join(产物, "台账.jsonl"))
    构造 = (time.perf_counter() - t0) / 3 * 1000
    rec = 发(器, 基线音色())
    x = np.asarray(rec["波形"], float)
    结["构造毫秒"] = round(构造, 2)
    结["新路径实时率"] = rec["实时率"]
    结["新路径时长"] = rec["时长秒"]
    结["新路径自检"] = rec["自检"]
    结["新路径通过"] = rec["通过"]
    # 旧路径（不传音色）实时率与波形（保持）
    t0 = time.perf_counter()
    rec旧 = 器.发声(np.sign(np.random.default_rng(55).standard_normal(1024)).astype(np.float32),
                    文本, dict(能量=0.5, 新异=0.5, 消化债=0.0, 饱足=0.5))
    结["旧路径实时率"] = rec旧["实时率"]
    结["新旧波形相关"] = round(float(np.corrcoef(
        x[:min(len(x), len(rec旧["波形"]))], np.asarray(rec旧["波形"])[:min(len(x), len(rec旧["波形"]))])[0, 1]), 4)
    # B 确定性
    r1 = 发(器, 基线音色()); r2 = 发(器, 基线音色())
    结["新路径确定性"] = bool(np.array_equal(np.asarray(r1["波形"]), np.asarray(r2["波形"])))
    # C 单参数接线（16 带包络 + RMS + F0）
    基 = 发(器, 基线音色())["波形"]
    基rms = float(np.sqrt(np.mean(np.asarray(基, float) ** 2)))
    参臂 = {}
    for 名, 值列 in (("紧喉度", (0.05, 0.55, 0.95)), ("粗糙度", (0.0, 0.5, 1.0)),
                     ("幅度", (0.2, 0.5, 0.9)), ("送气量", (0.0, 0.5, 1.0))):
        for v in 值列:
            a = 基线音色(); a[名] = v
            if 名 == "送气量":
                a["送气增益"] = 0.6
            y = np.asarray(发(器, a)["波形"], float)
            rms = float(np.sqrt(np.mean(y ** 2)))
            n = min(len(y), len(基))
            corr = float(np.corrcoef(y[:n], 基[:n])[0, 1]) if n > 8 else 1.0
            参臂["%s=%.1f" % (名, v)] = dict(
                包络余弦=round(余弦(包络7(基), 包络7(y)), 4),
                波形相关=round(corr, 4),
                逐位同=bool(np.array_equal(y, 基)),
                相对RMS=round(rms / max(基rms, 1e-9), 3),
                F0=round(F0中位(y), 1))
    结["参数接线"] = 参臂
    # D 共振峰/带宽/谱倾斜
    D = {}
    基spec = np.abs(np.fft.rfft(基 * np.hanning(len(基))))
    基频 = np.fft.rfftfreq(len(基), 1.0 / FS)
    基重心 = float(np.sum(基频 * 基spec) / (np.sum(基spec) + 1e-12))
    for 名, 改, 标签 in (("共振峰缩放", (0.85, 0.90, 0.95), "共振峰×0.85"),
                        ("带宽缩放", 1.6, "带宽×1.6"),
                        ("谱倾斜", -0.35, "谱倾斜-0.35")):
        a = 基线音色(); a[名] = 改
        y = np.asarray(发(器, a)["波形"], float)
        spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        频 = np.fft.rfftfreq(len(y), 1.0 / FS)
        c = float(np.sum(频 * spec) / (np.sum(spec) + 1e-12))
        D[标签] = dict(包络余弦=round(余弦(包络7(基), 包络7(y)), 4),
                      谱重心=round(c, 1), 基谱重心=round(基重心, 1))
    结["扩展维度"] = D
    # E 保存 wav
    for 名, 音 in (("基线", 基线音色()), ("紧喉0.95", dict(基线音色(), 紧喉度=0.95)),
                   ("粗糙1.0", dict(基线音色(), 粗糙度=1.0)),
                   ("共振峰0.85", dict(基线音色(), 共振峰缩放=(0.85, 0.90, 0.95)))):
        rec = 发(器, 音)
        器._写wav(os.path.join(产物, "%s.wav" % 名), np.asarray(rec["波形"], float))
    return 结


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    结 = 跑()
    p = os.path.join(产物, "深度优化结果.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(结, f, ensure_ascii=False, indent=1)
    for k, v in 结.items():
        print(k, ":", v)
    print("落盘:", p)
