# -*- coding: utf-8 -*-
"""meeting61/实施/实验_共振峰架构.py —— 声道滤波架构对照（决定 v6 走哪条路）。

诊断_元音_v5.py 已证明：v5 的元音共振峰没有到位（120Hz 声线峰位误差中位 23%），
F1 被系统性拉低 50–140Hz，/ao/ 的 F1–F2 谷只有 0.2dB（两峰糊成一团）。

候选架构（同一「声门流 −12dB/oct + 单次唇辐射 +6dB/oct」声源）：
  A 并联（v5 现状）：5 个 DC 增益=1 谐振器并联，增益 (1.0,0.55,0.20,0.09,0.04)
  B 串联（Klatt cascade）：5 个 DC 增益=1 谐振器串联
  C 串联 + 逐级增益 (1.0,1.0,0.7,0.5,0.35)：把 F3–F5 收回来
  D 并联 + 中心归一：每个谐振器归一化到自身中心频率处增益=1（消掉低频平台）

量测（120Hz 声线、稳态元音）：峰位/峰高/峰谷调制度/谐波斜率/谱带占比。
用法：``python 实验_共振峰架构.py``
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import 引擎_v4 as E4  # noqa: E402
import 引擎_v5 as E5  # noqa: E402

FS = 44100
帧率 = 100
元音 = {
    "a": (825.0, 1305.0, 2714.0, 3654.0, 4698.0),
    "o": (528.0, 887.0, 2610.0, 3445.0, 4385.0),
    "e": (572.0, 1305.0, 2610.0, 3602.0, 4646.0),
    "i": (330.0, 2349.0, 3132.0, 3967.0, 5011.0),
    "u": (352.0, 752.0, 2610.0, 3550.0, 4489.0),
}
默认带宽 = E4.默认带宽


def _源(f0=120.0, 秒=0.30, 紧喉=0.3, 种子=300):
    n = int(秒 * FS)
    表 = E5.声门流表(1024, 紧喉)
    相 = np.mod(np.cumsum(np.full(n, f0 / FS)), 1.0)
    idx = np.minimum((相 * 1024).astype(np.int64), 1023)
    x = 表[idx]
    b, a = E4.butter(2, min(6000.0, FS * 0.45) / (FS * 0.5), btype="low")
    x = E4.lfilter(b, a, x)
    return x


def _唇辐射(x):
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - x[:-1]
    return y


def 走_并联(x, Fs, 增益):
    y = np.zeros_like(x)
    for j in range(5):
        A, B, C = E4.谐振器系数(Fs[j], 默认带宽[j])
        yj = E4.lfilter([A], [1.0, -B, -C], x)
        y += 增益[j] * yj
    return y


def 走_串联(x, Fs, 增益=(1.0,) * 5):
    y = x
    for j in range(5):
        A, B, C = E4.谐振器系数(Fs[j], 默认带宽[j])
        y = E4.lfilter([A * 增益[j]], [1.0, -B, -C], y)
    return y


def 走_并联中心归一(x, Fs, 增益):
    """每个谐振器先归一到自身中心频率处 |H|=1（消掉低频 DC 平台）。"""
    y = np.zeros_like(x)
    for j in range(5):
        A, B, C = E4.谐振器系数(Fs[j], 默认带宽[j])
        A0, B0, C0 = E4.谐振器系数(Fs[j], 默认带宽[j])
        z = np.exp(1j * 2.0 * math.pi * Fs[j] / FS)
        H = A0 / (1.0 - B0 * z ** -1 - C0 * z ** -2)
        g = 1.0 / max(abs(H), 1e-9)
        yj = E4.lfilter([A * g], [1.0, -B, -C], x)
        y += 增益[j] * yj
    return y


架构 = {
    "A_并联_v5现状": lambda x, Fs: 走_并联(x, Fs, (1.0, 0.55, 0.20, 0.09, 0.04)),
    "B_串联": lambda x, Fs: 走_串联(x, Fs),
    "C_串联+逐级增益": lambda x, Fs: 走_串联(x, Fs, (1.0, 1.0, 0.70, 0.50, 0.35)),
    "D_并联中心归一": lambda x, Fs: 走_并联中心归一(x, Fs, (1.0, 0.55, 0.30, 0.18, 0.10)),
}


def _倒谱包络(x, 提升秒=0.0022):
    x = np.asarray(x, dtype=np.float64)
    x = x - float(np.mean(x))
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    cep = np.fft.irfft(np.log(np.abs(X) + 1e-12))
    L = max(4, int(提升秒 * FS))
    cep[L:-L] = 0.0
    env = np.exp(np.fft.rfft(cep).real)
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    return f, env


def _峰(env, f, lo, hi):
    m = (f >= lo) & (f <= hi)
    if not np.any(m):
        return 0.0, 0.0
    i = int(np.argmax(env[m]))
    return float(f[m][i]), float(env[m][i])


def _谷(env, f, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.min(env[m])) if np.any(m) else 0.0


def 量(x, Fs):
    f, env = _倒谱包络(x)
    底m = (f >= 150.0) & (f <= 5000.0)
    for t in Fs:
        底m &= ~((f >= t * 0.74) & (f <= t * 1.26))
    底 = float(np.median(env[底m])) if np.any(底m) else 1e-9
    峰s = []
    for t in Fs:
        c, v = _峰(env, f, t * 0.78, t * 1.22)
        峰s.append(dict(目标=t, 实测=round(c, 1), 误差=round(abs(c - t) / t, 3),
                        高dB=round(20 * math.log10(max(v, 1e-12) / 底), 1)))
    谷 = {}
    for j in range(len(Fs) - 1):
        if 峰s[j]["实测"] and 峰s[j + 1]["实测"] and 峰s[j + 1]["实测"] > 峰s[j]["实测"]:
            g = _谷(env, f, 峰s[j]["实测"] * 1.12, 峰s[j + 1]["实测"] * 0.88)
            if g > 0:
                谷["F%dF%d" % (j + 1, j + 2)] = round(
                    20 * math.log10(max(峰s[j]["高dB"] and 10 ** (峰s[j]["高dB"] / 20), 1e-12) / g), 1)
    P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    fr = np.fft.rfftfreq(len(x), 1.0 / FS)
    P[0] = 0.0
    T = float(P.sum()) or 1.0
    带 = {k: round(float(P[(fr >= a) & (fr < b)].sum() / T), 4)
         for k, (a, b) in {"0-500": (0, 500), "500-1k": (500, 1000),
                          "1-2k": (1000, 2000), "2-4k": (2000, 4000),
                          "4-8k": (4000, 8000)}.items()}
    重心 = float((fr * P).sum() / T)
    return dict(峰=峰s, 谷dB=谷, 带=带, 谱重心Hz=round(重心, 0),
                最大峰位误差=max(p["误差"] for p in 峰s),
                峰高序=[p["高dB"] for p in 峰s])


def 谐波斜率(x, f0=120.0):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    pts = []
    k = 1
    while k * f0 < 4500.0:
        c = k * f0
        m = (f >= c * 0.95) & (f <= c * 1.05)
        if np.any(m):
            pts.append((c, 20 * math.log10(max(float(np.max(X[m])), 1e-12))))
        k += 1
    xs = np.log2(np.array([p[0] for p in pts]) / 1000.0)
    ys = np.array([p[1] for p in pts]) - float(np.mean([p[1] for p in pts]))
    w = np.polyfit(xs, ys, 1)
    return round(float(w[0]), 2)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    报 = {}
    print("=" * 110)
    print("谐波斜率（dB/oct，理想 ≈ −6；参考原声实测 −9.3/−13.3/−10.7）")
    ts = np.arange(int(0.30 * FS), dtype=np.float64) / FS
    src = _源()
    for 名, 走 in 架构.items():
        x = _唇辐射(走(src, 元音["a"]))
        x = x / max(float(np.sqrt(np.mean(x ** 2))), 1e-12)
        报.setdefault(名, {})["谐波斜率"] = 谐波斜率(x)
        print("  %-16s %s" % (名, 报[名]["谐波斜率"]))
    print("=" * 110)
    for 元 in 元音:
        Fs = 元音[元]
        print("[%s] 目标 F1–F5 = %s" % (元, " ".join("%.0f" % t for t in Fs)))
        for 名, 走 in 架构.items():
            x = _唇辐射(走(src, Fs))
            x = x / max(float(np.sqrt(np.mean(x ** 2))), 1e-12)
            r = 量(x, Fs)
            报.setdefault(名, {})[元] = r
            print("   %-16s 峰位 %s | 误差 %.2f | 峰高 %s | 谷 %s | 重心 %.0f"
                  % (名,
                     " ".join("%5.0f" % p["实测"] for p in r["峰"]),
                     r["最大峰位误差"],
                     " ".join("%5.1f" % h for h in r["峰高序"]),
                     json.dumps(r["谷dB"], ensure_ascii=False),
                     r["谱重心Hz"]))
        print("-" * 110)
    out = os.path.join(os.path.dirname(HERE), "产物", "临时", "实验_共振峰架构.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    print("JSON:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
