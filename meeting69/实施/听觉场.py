# -*- coding: utf-8 -*-
"""meeting69/实施/听觉场.py —— S1 听场 Ψ（32 带 × 32 槽 = 1024；09 §2.2）。

输入：`耳蜗前端_S0_2.分析()` 的 `场未适`（T×64，压缩后、适应前；S0.2 已修包络）。
Ψ 构建（先写死，09 §2.2）：
  L = log(场未适 + 1e-3)
  每带用「截至当前拍的 2 s 历史」（最少 200 ms）求中位 m_b 与 IQR q_b
  z = clip((L − m_b)/max(q_b, 0.05), −3, 3)/3
  相邻 2 带合并 → 32 临界带；160 ms 窗按 5 ms 槽求均 → 32 槽
  band-major flatten：index = b·32 + slot（保拓扑）
"""
from __future__ import annotations

import numpy as np

频带数 = 32
时槽 = 32
槽ms = 5
窗ms = 160
归一窗ms = 2000
分组 = ((0, 6), (6, 11), (11, 16), (16, 21), (21, 26), (26, 32))   # 6×32=192 逐格意外


def 带心32(中心频率64):
    f = np.asarray(中心频率64, np.float64)
    return np.sqrt(f[0::2] * f[1::2])          # 相邻 2 带几何均值


def 构建(L, t, 带心=(), q下限=0.05, 返回未截断=False):
    """L: (T,64) 的 log 场未适；t: 拍时刻 ms；返回 Ψ float32[1024]（band-major）。

    q下限：09 v1 为 0.05；16（S1 v3）改为 1.0（去近静音带放大）。
    返回未截断=True 时返回 (Ψ, Z)：Z 为**未截断**的归一化场（单位为 log-IQR 倍数），
    供 18（S1 v4）的「误差通道不用截断后的 Ψ」使用。
    """
    T = len(L)
    t = int(max(0, min(t, T)))
    h0 = max(0, t - 归一窗ms)
    hist = L[h0:t]
    if len(hist) < 2:
        hist = L[: max(2, t)]
    m = np.median(hist, axis=0)
    q = np.percentile(hist, [25, 75], axis=0)
    q = np.maximum(q[1] - q[0], q下限)
    w0 = max(0, t - 窗ms)
    win = L[w0:t]
    n = len(win)
    if n == 0:
        z0 = np.zeros(频带数 * 时槽, np.float32)
        return (z0, z0.copy()) if 返回未截断 else z0
    zz = (win - m) / q                          # 未截断
    z = np.clip(zz, -3.0, 3.0) / 3.0
    if n < 窗ms:
        pad = np.zeros((窗ms - n, z.shape[1]), z.dtype)
        z = np.vstack([pad, z]); zz = np.vstack([pad.astype(zz.dtype), zz])
    z32 = (z[:, 0::2] + z[:, 1::2]) / 2.0      # (160, 32)
    Ψ = z32.reshape(时槽, 槽ms, 频带数).mean(axis=1)   # (32 槽, 32 带)
    Ψ = np.ascontiguousarray(Ψ.T.reshape(-1).astype(np.float32))
    if not 返回未截断:
        return Ψ
    zz32 = (zz[:, 0::2] + zz[:, 1::2]) / 2.0
    Z = zz32.reshape(时槽, 槽ms, 频带数).mean(axis=1)
    return Ψ, np.ascontiguousarray(Z.T.reshape(-1).astype(np.float32))


def 逐格意外(ε):
    """ε float32[1024] → float32[192]（6 带组 × 32 槽）。"""
    g = np.asarray(ε, np.float32).reshape(频带数, 时槽)
    out = np.zeros((len(分组), 时槽), np.float32)
    for i, (a, b) in enumerate(分组):
        out[i] = g[a:b].mean(axis=0)
    return np.ascontiguousarray(out.reshape(-1))


def 焦点窗统计(格, b0, s0, 高=4, 宽=8):
    return float(格[b0:b0 + 高, s0:s0 + 宽].mean())


def 积分窗和(格):
    """返回积分图，便于一次算完所有 725 个窗的 sum。"""
    g = np.asarray(格, np.float64)
    S = np.zeros((g.shape[0] + 1, g.shape[1] + 1), np.float64)
    S[1:, 1:] = np.cumsum(np.cumsum(g, axis=0), axis=1)
    return S


def 窗和(S, b0, s0, 高=4, 宽=8):
    return S[b0 + 高, s0 + 宽] - S[b0, s0 + 宽] - S[b0 + 高, s0] + S[b0, s0]


def 窗和矩阵(g, 高=4, 宽=8):
    """一次算完所有候选窗（29×25）的 sum。"""
    S = 积分窗和(g)
    return S[高:, 宽:] - S[:-高, 宽:] - S[高:, :-宽] + S[:-高, :-宽]


def A计权(f):
    """A 计权线性幅度（归一化 max=1；用于 21 v6 的听感加权误差通道）。"""
    f = np.asarray(f, np.float64)
    ra = (12194.0 ** 2 * f ** 4) / ((f ** 2 + 20.6 ** 2) * np.sqrt((f ** 2 + 107.7 ** 2) * (f ** 2 + 737.9 ** 2)) * (f ** 2 + 12194.0 ** 2))
    a = 20.0 * np.log10(ra + 1e-12) + 2.0
    w = 10.0 ** (a / 20.0)
    return w / w.max()
