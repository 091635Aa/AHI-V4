# -*- coding: utf-8 -*-
"""meeting69/实施/谐波重合成_v1.py —— 候选「H」：把平滑 mel 重新用**谐波+噪声**激励合成。

动机：GL 的电音/嗡声来自「mel 已被模型平滑掉谐波结构 → GL 只能重建出噪声性激励」。本件用
自研 F0 轨迹（`耳蜗前端_S0_2.F0轨迹`，S0.1 已验证）**显式**给出谐波结构：
  浊音帧：Σ_k A_k(f0) cos(k·φ)，φ = 2π∫f0 — 相位锁定（声门样），振幅取自 mel 反演的线性幅度谱；
  清音帧：噪声（固定种子），谱形取自同一幅度谱（librosa.istft + 随机相位）。
零训练、确定性；仍非 Transformer、非外挂模型（C1/C10）。
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M69 = HERE.parent
M66 = M69.parent / "meeting66"
M66实施 = M66 / "实施"
SR = 22050


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


声 = 载("声码器_只读", M66实施 / "声码器与评测.py")
耳 = 载("耳蜗前端_S0_2", HERE / "耳蜗前端_S0_2.py")
精 = 载("音频线_v1_精修", HERE / "音频线_v1_精修.py")

N_FFT = 1024
窗长 = 551
跳长 = 220
MEL下限, MEL上限 = 声.MEL下限, 声.MEL上限
AMIN = 1e-6


def 幅度谱(mel):
    lib = 声.librosa模块()
    m = np.clip(np.nan_to_num(np.asarray(mel, np.float64), nan=MEL下限), MEL下限, MEL上限)
    功率 = np.maximum(np.exp(m) - AMIN, 0.0)
    M = lib.feature.inverse.mel_to_stft(功率.T, sr=int(SR), n_fft=N_FFT, power=2.0,
                                      fmin=声.FMIN, fmax=SR / 2.0)
    return np.maximum(np.asarray(M, np.float64), 0.0)          # (1+N/2, T)


def 谐波合成(mel, f0_源=None, 谐波上限Hz=8000.0, 噪声种子=20261022):
    """→ (波形, 元信息)。f0_源：用于估计 F0 的波形（默认用 GL 预览）。"""
    lib = 声.librosa模块()
    M = 幅度谱(mel)
    T = M.shape[1]
    N = (T - 1) * 跳长
    if f0_源 is None:
        f0_源 = 声.mel转波(mel)
    f0_源 = np.asarray(f0_源, np.float32)
    t, f0 = 耳.F0轨迹(f0_源)
    帧心 = (np.arange(T) * 跳长 + 跳长 / 2.0) / SR
    f0i = np.interp(帧心, t, f0)                                # 未浊段会被插值，下面用浊音掩码
    浊 = np.interp(帧心, t, (np.asarray(f0) > 0).astype(float)) >= 0.5
    f0i = np.where(浊, np.clip(f0i, 60.0, 500.0), 0.0)
    df = SR / N_FFT
    # 每谐波每帧的幅度（在 f0 帧上取值）
    kmax = int(np.floor(min(谐波上限Hz, SR / 2 - df) / max(np.nanmin(f0i[浊]) if 浊.any() else 200.0, 60.0)))
    kmax = int(min(max(kmax, 1), 60))
    # 逐样本 f0（帧心线性插值）
    样本t = np.arange(N) / SR
    f0s = np.interp(样本t, 帧心, f0i)
    浊s = np.interp(样本t, 帧心, 浊.astype(float))
    浊s = np.clip(浊s, 0.0, 1.0)
    # 相位：φ1 = 2π∫f0 dt
    φ1 = 2 * np.pi * np.cumsum(f0s) / SR
    y_v = np.zeros(N, np.float64)
    for k in range(1, kmax + 1):
        # 第 k 谐波幅度：每帧在 bin = k*f0i/df 处线性取 M
        A帧 = np.zeros(T, np.float64)
        for i in range(T):
            if not 浊[i]:
                continue
            b = int(round(k * f0i[i] / df))
            if 1 <= b < M.shape[0] - 1:
                w = k * f0i[i] / df - b
                A帧[i] = (1 - w) * M[b, i] + w * M[b + 1, i]
        if not np.any(A帧 > 0):
            continue
        As = np.interp(样本t, 帧心, A帧)
        y_v += As * np.cos(k * φ1)
    # 清音：随机相位 ISTFT（固定种子）
    rng = np.random.RandomState(int(噪声种子))
    相位 = np.exp(1j * rng.uniform(-np.pi, np.pi, size=M.shape))
    y_u = lib.istft(M * 相位, hop_length=跳长, win_length=窗长, n_fft=N_FFT,
                    window="hann", center=True, length=N)
    y_u = np.nan_to_num(np.asarray(y_u, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    # 浊/清交叉淡化（5 ms 斜坡）
    n = int(0.005 * SR)
    if n > 1:
        ker = np.ones(n) / n
        浊s = np.convolve(浊s, ker, mode="same")
    y = 浊s * y_v + (1 - 浊s) * y_u
    p = float(np.abs(y).max())
    if p > 1e-9:
        y = y / p * 0.9
    meta = dict(谐波数=int(kmax), 浊帧比=round(float(np.mean(浊)), 3),
                F0中位=round(float(np.median(f0i[浊])), 1) if 浊.any() else None)
    return y.astype(np.float32), meta


def 不纯度(x):
    """jitter（相邻 F0 相对变化）+ shimmer（相邻帧幅度相对变化）。"""
    t, f0 = 耳.F0轨迹(np.asarray(x, np.float32))
    js = []
    上 = None
    for ti, fi in zip(t, f0):
        if fi > 0:
            if 上 is not None:
                js.append(abs(float(fi) - 上) / 上)
            上 = float(fi)
        else:
            上 = None
    W = int(0.040 * SR)
    amps = []
    for ti, fi in zip(t, f0):
        c = int(round(float(ti) * SR)); a = c - W // 2
        if a < 0 or a + W > len(x):
            continue
        w = np.asarray(x[a:a + W], np.float64)
        amps.append(float(np.sqrt(np.mean(w * w))))
    sh = [abs(amps[i] - amps[i - 1]) / max(amps[i - 1], 1e-9) for i in range(1, len(amps)) if amps[i - 1] > 1e-6]
    return dict(jitter=round(float(np.mean(js)) if js else float("nan"), 5),
                shimmer=round(float(np.mean(sh)) if sh else float("nan"), 4))
