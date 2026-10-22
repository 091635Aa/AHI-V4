# -*- coding: utf-8 -*-
"""S0.1 设计期诊断（非正式读数；正式读数见 判据_S0_1.py，须在预注册之后）。

目的：
  1) 复现 S0 的 J4 F0 误差，看误差集中在哪些帧（低 F0 / 边界 / 倍频 / 窗中心 vs 起点）；
  2) 在 稳态元音 上试三种适应实现（除式 5/100、除式 5/200、减式 v0），看 J6 同窗口比值；
  3) 试 YIN 型 F0 估计器（30/40/50 ms 窗；中心/起点对齐），粗看相对 MAE。
输出：meeting69/产物/S0.1/诊断_设计期.json
"""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M69 = HERE.parent
实施 = M69 / "实施"


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


前 = 载("耳蜗前端_S0", 实施 / "耳蜗前端_S0.py")
清 = json.load(open(M69 / "产物" / "S0" / "manifest.json", encoding="utf-8"))["真值"]
出 = M69 / "产物" / "S0.1"
出.mkdir(parents=True, exist_ok=True)
报 = {}

# ── 1) 复现 S0 J4：逐帧误差 ──
声调 = [g for g in 清 if g["类"] == "声调"]
帧 = {}
for g in 声调:
    r = 前.分析(g["路径"])
    f0v = np.array(r["F0"]["f0"]); t = np.array(r["F0"]["t"])
    gt = np.interp(t - 0.3, np.linspace(0, 0.5, len(g["f0轮廓"])), g["f0轮廓"])
    ok = f0v > 0
    err = np.where(ok, (f0v - gt) / gt * 100.0, np.nan)
    帧[g["标签"]] = dict(
        n帧=int(len(f0v)), 浊帧=int(ok.sum()),
        相对误差p50=round(float(np.nanpercentile(np.abs(err), 50)), 3),
        相对误差p90=round(float(np.nanpercentile(np.abs(err), 90)), 3),
        相对误差max=round(float(np.nanmax(np.abs(err))), 3),
        MAE_rel=round(float(np.nanmean(np.abs(err))), 3),
        浊帧估计范围=[round(float(f0v[ok].min()), 1), round(float(f0v[ok].max()), 1)] if ok.any() else None,
        GT范围=[round(float(gt.min()), 1), round(float(gt.max()), 1)],
    )
    坏 = np.argsort(-np.abs(np.nan_to_num(err)))[:5]
    帧[g["标签"]]["最大误差帧"] = [
        dict(t=round(float(t[i]), 3), 估=round(float(f0v[i]), 1), 真=round(float(gt[i]), 1))
        for i in 坏]
报["J4_现状逐帧"] = 帧

# ── 2) 三种适应实现的 J6 同窗口比（稳态元音 2 s，起于 0.2 s） ──
g = next(x for x in 清 if x["类"] == "适应")
x = 前.读wav(g["路径"])
样本, fs = 前.滤波组(x)
包 = 前.包络(样本)
n1 = len(x) // 前.包络步
包1 = 包[:, : n1 * 前.包络步].reshape(包.shape[0], n1, 前.包络步).mean(axis=2)
压缩 = np.power(包1, 0.4)


def 适应变体(压缩, 变体):
    T = 压缩.shape[1]
    h0 = np.maximum(np.median(压缩, axis=1) * 0.1, 1e-6)
    af = np.zeros(压缩.shape[0], np.float64)
    as_ = np.zeros(压缩.shape[0], np.float64)
    τs = 0.200 if 变体 == "除式5/200" else 0.100
    kf = 1.0 / (0.005 * 1000.0)
    ks = 1.0 / (τs * 1000.0)
    out = np.zeros(压缩.shape, np.float32)
    for t in range(T):
        c = 压缩[:, t]
        af += (c - af) * kf
        as_ += (c - as_) * ks
        if 变体.startswith("除式"):
            out[:, t] = c / (af + as_ + h0)
        else:  # 减式 v0 形：0.6 快 + 0.4 慢，clamp 0
            out[:, t] = np.maximum(c - (0.6 * af + 0.4 * as_), 0.0)
    return out.T.astype(np.float64)


适 = {}
for 变体 in ("除式5/100", "除式5/200", "减式"):
    f = 适应变体(压缩, 变体)
    早 = float(f[200:300].mean()); 晚 = float(f[1500:1800].mean())
    起 = float(f[200].mean())
    适[变体] = dict(比=round(晚 / (早 + 1e-12), 4), 早=round(早, 5), 晚=round(晚, 5),
                    起振首帧=round(起, 4), 晚除首帧=round(晚 / (起 + 1e-12), 4),
                    晚除压缩稳态=round(晚 / float(压缩[200:2200].mean() + 1e-12), 4))
报["J6_适应变体"] = 适

# ── 3) YIN 型 F0 估计器原型（诊断） ──
def 低通(x, fc):
    import scipy.signal as sps
    sos = sps.butter(4, fc / (前.SR / 2), btype="low", output="sos")
    return sps.sosfilt(sos, x)


def YIN(x, 帧长ms=40, 跳ms=2.5, fmin=60.0, fmax=500.0, 阈=0.15, lpf=1200.0, 中心对齐=True):
    y = 低通(x, lpf)
    W = int(帧长ms * 前.SR / 1000)
    跳 = int(跳ms * 前.SR / 1000)
    n = 1 + max(0, (len(y) - W) // 跳)
    if n <= 0:
        return np.zeros(0), np.zeros(0)
    帧s = np.lib.stride_tricks.sliding_window_view(y, W)[: n * 跳: 跳][:n].astype(np.float64)
    帧s = 帧s * np.hanning(W)
    τmin = max(2, int(前.SR / fmax))
    τmax = min(W - 2, int(前.SR / fmin))
    # d(τ) = E0(τ) + E1(τ) - 2 r(τ)，用 FFT 自相关一次算全 lags
    F = np.fft.rfft(帧s, n=2 * W, axis=1)
    R = np.fft.irfft(F * np.conj(F), n=2 * W, axis=1)[:, : W]
    E = np.concatenate([np.zeros((n, 1)), np.cumsum(帧s ** 2, axis=1)], axis=1)  # E[k]=Σ_{j<k}
    总 = E[:, W:W + 1]  # 总能量
    τs = np.arange(τmax + 1)
    e0 = E[:, W - τs]                      # Σ_{j=0}^{W-1-τ} a_j²
    e1 = 总 - E[:, τs]                     # Σ_{j=τ}^{W-1} a_j²  (E[:,τ] = Σ_{j<τ})
    d = np.maximum(e0 + e1 - 2 * R[:, :τmax + 1], 0.0)
    d[:, 0] = 0.0
    # CMNDF
    cs = np.cumsum(d, axis=1)
    累计 = np.ones_like(d)
    累计[:, 1:] = d[:, 1:] * τs[1:] / (cs[:, 1:] + 1e-12)
    f0 = np.zeros(n)
    for i in range(n):
        dd = 累计[i, :τmax + 1]
        cand = None
        for τ in range(τmin, τmax):
            if dd[τ] < 阈 and dd[τ] <= dd[τ - 1] and dd[τ] <= dd[τ + 1]:
                cand = τ
                break
        if cand is None:
            cand = int(np.argmin(dd[τmin:τmax + 1]) + τmin)
        ak = max(1, min(cand, τmax - 1))
        a1, b1, c1 = dd[ak - 1], dd[ak], dd[ak + 1]
        δ = float(np.clip(0.5 * (a1 - c1) / (a1 - 2 * b1 + c1 + 1e-12), -0.5, 0.5))
        滞后 = ak + δ
        f0[i] = 前.SR / 滞后 if 滞后 > 0 else 0.0
    if 中心对齐:
        t = (np.arange(n) * 跳 + W / 2.0) / 前.SR
    else:
        t = np.arange(n) * 跳 / 前.SR
    return t, f0


def 评估(f0v, t, g):
    gt = np.interp(t - 0.3, np.linspace(0, 0.5, len(g["f0轮廓"])), g["f0轮廓"])
    ok = f0v > 0
    if ok.sum() == 0:
        return None
    return dict(浊帧=int(ok.sum()), 总帧=int(len(f0v)),
                覆盖率=round(float(ok.mean()), 3),
                MAE_rel=round(float(np.mean(np.abs(f0v[ok] - gt[ok]) / gt[ok])), 4),
                p90=round(float(np.percentile(np.abs(f0v[ok] - gt[ok]) / gt[ok], 90)), 4))


Y = {}
for 窗 in (30, 40, 50):
    for 中心 in (True, False):
        key = f"窗{窗}ms_{'中心' if 中心 else '起点'}"
        for g in 声调:
            x = 前.读wav(g["路径"])
            t, f0v = YIN(x, 帧长ms=窗, 中心对齐=中心)
            Y.setdefault(key, {})[g["标签"]] = 评估(f0v, t, g)
报["YIN原型"] = Y

(出 / "诊断_设计期.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(报, ensure_ascii=False, indent=1))
