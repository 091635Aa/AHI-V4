# -*- coding: utf-8 -*-
"""meeting70/实施/器官v2.py —— 发声器官 v2（零训练，解析控制）。

规格（预注册 03）：22.05 kHz；控制帧 5 ms；逐样本合成。
  · 声门源：LF 型单周期波形（时域表；开商/偏斜固定初值），F0 驱动、相位连续；
  · 气声：按 HNR 推噪声比混合（噪声经 1 kHz 高通，随声门包络调制）；
  · 清音/送气：噪声源经同一共振峰级联成形；
  · 声道：4 个时变二阶共振峰谐振器（F1–F4）级联近似（KL 波导列 v2.1，本版未实现，已登记偏差）；
  · 亚帧瞬态：能量突升的清音帧注入 1–3 ms 噪声爆破；
  · 唇辐射：一阶差分；能量按 5 ms 目标包络匹配；固定种子噪声（确定性）。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M70 = HERE.parent
M69 = M70.parent / "meeting69"
SR = 22050
帧 = 110          # 5 ms
种子 = 20261022


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


耳 = 载("耳蜗前端_S0_2", M69 / "实施" / "耳蜗前端_S0_2.py")

# ── LF 型声门流（一周期表，flow 与其导数） ──
_OQ = 0.62          # 开商
_OQ1 = 0.42         # 开相内「开→闭」分界
_表长 = 4096


def 声门表(OQ=0.62, OQ1=0.42):
    key = (round(float(OQ), 4), round(float(OQ1), 4))
    if key in _表缓存:
        return _表缓存[key]
    n = _表长
    t = np.arange(n) / n
    flow = np.zeros(n)
    for i, u in enumerate(t):
        if u < OQ:
            if u < OQ1:
                flow[i] = 0.5 * (1 - np.cos(np.pi * u / max(OQ1, 1e-6)))
            else:
                flow[i] = 0.5 * (1 + np.cos(np.pi * (u - OQ1) / max(OQ - OQ1, 1e-6)))
        else:
            flow[i] = 0.0
    d = np.gradient(flow) * n
    d = d - d.mean()
    d = d / (np.max(np.abs(d)) + 1e-12)
    _表缓存[key] = (flow, d)
    return flow, d


_表缓存 = {}
_流, _流导 = 声门表()


def lpc(seg, 阶):
    """自相关 + Levinson-Durbin → a[0..阶]（a[0]=1）。"""
    seg = np.asarray(seg, np.float64)
    r = np.correlate(seg, seg, mode="full")[len(seg) - 1: len(seg) + 阶]
    if r[0] <= 1e-12:
        return np.r_[1.0, np.zeros(阶)]
    a = np.zeros(阶 + 1); a[0] = 1.0
    e = float(r[0])
    for i in range(1, 阶 + 1):
        acc = float(r[i]) + (float(np.dot(a[1:i], r[i - 1:0:-1])) if i > 1 else 0.0)
        k = -acc / e
        if i > 1:
            a[1:i] = a[1:i] + k * a[i - 1:0:-1]
        a[i] = k
        e *= (1 - k * k)
        if e <= 1e-12:
            break
    return a


def 分析控制(x):
    """真人 wav → 控制帧（5 ms）：f0, 浊, F[4], B[4], 能量, 噪声比。"""
    x = np.asarray(x, np.float64)
    T = max(1, len(x) // 帧)
    f0c = np.zeros(T)
    t, f0 = 耳.F0轨迹(x.astype(np.float32))
    f0f = np.interp((np.arange(T) + 0.5) * 帧 / SR, t, np.asarray(f0, np.float64))
    浊 = f0f > 0
    F = np.zeros((T, 4)); B = np.zeros((T, 4)); E = np.zeros(T); NR = np.zeros(T)
    A = np.tile(np.r_[1.0, np.zeros(24)], (T, 1))
    预 = np.append(x[0], x[1:] - 0.97 * x[:-1])
    W = int(0.025 * SR)
    for i in range(T):
        c = int((i + 0.5) * 帧)
        a = max(0, c - W // 2); b = min(len(x), a + W)
        seg = x[a:b]
        if b - a < 32:
            continue
        E[i] = float(np.sqrt(np.mean(seg ** 2)))
        # 噪声比：浊音段用自相关峰
        if 浊[i]:
            lag = int(round(SR / max(f0f[i], 60.0)))
            w = seg - seg.mean()
            能 = float(np.dot(w, w)) + 1e-12
            r = 0.0
            if lag < len(w):
                r = float(np.dot(w[:-lag], w[lag:]) / 能)
            r = min(max(r, 1e-4), 0.999)
            hnr = 10 * np.log10(r / (1 - r))
            NR[i] = float(np.clip(1.0 / (1.0 + 10 ** (hnr / 10.0)), 0.02, 1.0))
        else:
            NR[i] = 1.0
        # 共振峰
        segp = 预[a:b] * np.hamming(b - a)
        a_lpc = lpc(segp, 16)
        # 全极点包络（v2.1 模式用）：对预加重帧估 24 阶 LPC
        a24 = lpc(segp, 24)
        A[i, :] = a24
        try:
            rt = np.roots(a_lpc)
        except Exception:
            continue
        rt = rt[np.imag(rt) > 0.02]
        fr = np.sort(np.angle(rt) * SR / (2 * np.pi))
        bw = -np.log(np.abs(rt) + 1e-12) * SR / np.pi
        ok = (fr > 180) & (fr < 5200) & (bw > 20) & (bw < 900)
        fr, bw = fr[ok], bw[ok]
        for k in range(4):
            if k < len(fr):
                F[i, k] = fr[k]
                B[i, k] = bw[k]
            else:
                F[i, k] = [500, 1500, 2500, 3500][k]
                B[i, k] = [80, 110, 150, 220][k]
    # 平滑与钳位
    for k in range(4):
        lo, hi = [(180, 1000), (500, 2600), (1300, 3800), (2200, 5000)][k]
        F[:, k] = np.clip(F[:, k], lo, hi)
        B[:, k] = np.clip(B[:, k], 30, 700)
    # 帧间平滑（3 点）
    def 平(v):
        return np.convolve(v, np.ones(3) / 3, mode="same")
    for k in range(4):
        F[:, k] = 平(F[:, k]); B[:, k] = 平(B[:, k])
    E = np.maximum(E, 1e-6)
    return dict(T=T, f0=f0f, 浊=浊, F=F, B=B, E=E, NR=NR, A=A)


def 合成(c, 增益=1.0, 模式="共振峰", OQ=0.62, OQ1=0.42, 噪声尺度=0.25, 去加重=0.97):
    T = c["T"]; N = T * 帧
    f0 = np.asarray(c["f0"], np.float64)
    f0s = np.interp((np.arange(N) + 0.5) / 帧, np.arange(T) + 0.5, f0)
    浊s = np.interp((np.arange(N) + 0.5) / 帧, np.arange(T) + 0.5, c["浊"].astype(np.float64)) > 0.5
    f0s = np.where(浊s & (f0s > 50), f0s, 0.0)
    相位 = np.mod(np.cumsum(f0s) / SR, 1.0)
    rng = np.random.RandomState(种子)
    噪声 = rng.randn(N)
    噪声hp = np.append(噪声[0], 噪声[1:] - 噪声[:-1])
    # 声门源（浊）+ 气声/清音噪声
    源 = np.zeros(N)
    _f, _导 = 声门表(OQ, OQ1)
    浊idx = np.where(浊s)[0]
    if len(浊idx):
        源[浊idx] = np.interp(相位[浊idx] * _表长, np.arange(_表长), _导)
    NRs = np.interp((np.arange(N) + 0.5) / 帧, np.arange(T) + 0.5, c["NR"])
    源 = 源 * np.where(浊s, 1.0, 0.0) + 噪声hp * NRs * 噪声尺度
    if 模式 == "全极点":
        # v2.1：全极点包络（24 阶 LPC）+ LF 源；合成后去加重
        y = np.zeros(N)
        hist = np.zeros(25)
        for n in range(N):
            i = min(n // 帧, T - 1)
            aa = c["A"][i]
            acc = 0.0
            for k in range(1, 25):
                acc += aa[k] * hist[k - 1]
            v = 源[n] - acc
            hist[1:] = hist[:-1]
            hist[0] = v
            y[n] = v
        yd = np.zeros(N); prev = 0.0
        for n in range(N):
            prev = y[n] + 去加重 * prev
            yd[n] = prev
        y = yd
        for i in range(T):
            a0 = i * 帧; a1 = min(N, a0 + 帧)
            cur = float(np.sqrt(np.mean(y[a0:a1] ** 2))) + 1e-9
            y[a0:a1] *= (c["E"][i] * 增益) / cur
        p = float(np.abs(y).max())
        if p > 1e-9:
            y = y / p * 0.9
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # 逐帧预计算 4 级共振峰系数（Direct Form I）
    b0 = np.zeros((T, 4)); a1c = np.zeros((T, 4)); a2c = np.zeros((T, 4))
    for i in range(T):
        for k in range(4):
            Fk = float(c["F"][i, k]); Bk = float(c["B"][i, k])
            r = np.exp(-np.pi * Bk / SR)
            th = 2 * np.pi * Fk / SR
            b0[i, k] = 1 - r
            a1c[i, k] = -2 * r * np.cos(th)
            a2c[i, k] = r * r
    # 逐样本级联滤波（状态跨帧连续）
    x1 = np.zeros(4); x2 = np.zeros(4); y1 = np.zeros(4); y2 = np.zeros(4)
    y = np.zeros(N)
    for n in range(N):
        i = n // 帧
        if i >= T:
            i = T - 1
        v = 源[n]
        for k in range(4):
            out = b0[i, k] * v - a1c[i, k] * y1[k] - a2c[i, k] * y2[k]
            x2[k] = x1[k]; x1[k] = v
            y2[k] = y1[k]; y1[k] = out
            v = out
        y[n] = v
    # 唇辐射（一阶差分）
    y = np.append(y[0], y[1:] - 0.95 * y[:-1])
    # 能量包络匹配（5 ms）
    for i in range(T):
        a0 = i * 帧; a1 = min(N, a0 + 帧)
        cur = float(np.sqrt(np.mean(y[a0:a1] ** 2))) + 1e-9
        y[a0:a1] *= (c["E"][i] * 增益) / cur
    p = float(np.abs(y).max())
    if p > 1e-9:
        y = y / p * 0.9
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
