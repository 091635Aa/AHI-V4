# -*- coding: utf-8 -*-
"""meeting61/实施/引擎_v6.py —— 44.1kHz 源-滤波器声带引擎（v6：浊音声道改 Klatt 串联）。

为什么必须改（诊断_元音_v5.py + 实验_共振峰频谱.py 实测，不是听感猜测）：
  v4 起把浊音声道从「串联」改成「并联共振峰组」，原因是当时声源净谱接近平直、
  串联各峰高度接近 ⇒ 元音含混。但并联引入一个更严重的**模型级错误**：
  每个谐振器都是 A=1−B−C（DC 增益=1），5 路相加 ⇒ 低于 F1 的频段整体被抬高
  约 +5.5dB，形成低频平台。耳听等价量（输出(k·f0)=|H源|×|H声道|）实测：
      v5 并联：/a/ /e/ /i/ /u/ 的谱峰全部落在 240Hz，F1 突出度 −1.4/−0.5/−3.9/−1.0 dB
  ⇒ 所有元音共用一个低频「轰」，F1/F2 身份丢失 = 「听不出说什么」。

v6 的底层修改（只动浊音道；清辅音/噪声道与 v5 逐位同）：
  1) 浊音道改回 **Klatt 串联（F1→F5 级联）**：每个二阶节 DC 增益=1，串联后低于 F1
     增益=1（不再有平台），F1 处 |H|≈Q_F1 ⇒ 谱峰回到 F1。实测 /a/ 谱峰 840Hz、
     F1 突出度 +9.8dB（v5：240Hz / −1.4dB）。
  2) 串联天然给出 F1>F2>F3>F4>F5 的逐峰衰减（真人谱包络），不再需要手调逐峰增益。
  3) 为提高 F4/F5 的可听度同时**不引入低频平台**，加一条 **高通声门源的 F4/F5 并联支路**
     （HP 1.6kHz × 2 阶 → F4/F5 谐振 → 支路增益），直流处增益 0。
  4) 源标定按串联重新标定（同样 幅=1 时 RMS≈0.12）。

接口与 v5 相同：``合成(帧: dict, 音色: dict = None, 种子: int = 300) -> (波形, 元信息)``。
自检（真跑）新增耳听等价的「谱峰是否落在 F1」判据：``python 引擎_v6.py``。
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
import time
import wave

import numpy as np
from scipy.signal import lfilter, butter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import 引擎_v4 as E4  # noqa: E402  复用声门流表/谐振器/音色规整（v4 一字未改）

FS = E4.FS
帧率 = E4.帧率
每帧样本 = E4.每帧样本
产物临时 = E4.产物临时

# —— v6 标定（实测见 自检 与 判据_发音_v6.py）——
_源标定 = 0.304             # 声门源标定：按「带辅音音节内的浊音电平」对齐 v5（实测 v6/v5=+3.98dB）
_气声源权重 = 0.012
_气声直通权重 = 0.010
_噪标定 = 0.020             # 噪声帧标定（与 v5 同 ⇒ 辅音听感不回归）
_高频支路 = 0.55            # F3/F4/F5 高通并联支路增益（声门源 → 1.6k 高通 × F3/F4/F5 谐振）
_高频低切Hz = 1600.0
_高频支路增 = (0.30, 0.70, 0.40)   # F3/F4/F5 各自在并联支路里的相对幅度（Klatt 并联支路口径）
_声源低通Hz = 6000.0
_声源低通阶 = 2
_微颤Hz = 4.7
_微颤深 = 0.012
_并联下限 = 400.0
_默认共振峰增益 = (1.0, 0.55, 0.20, 0.09, 0.04)   # 仅保留给 声源=0 的旧路径兼容

_表栈 = E4._表栈
_表层数 = E4.表层数
_表长 = E4.表长
_谐振器系数 = E4.谐振器系数
_滤波块 = E4._滤波块
_帧数组 = E4._帧数组
_平滑 = E4._平滑


def 声门流表(n: int = 1024, 紧喉: float = 0.0) -> np.ndarray:
    """单周期声门流（Rosenberg），峰值归一、去直流（与 v5 同 ⇒ 源不变）。"""
    q = min(max(float(紧喉), 0.0), 1.0)
    Tp = 0.40 - 0.16 * q
    Tn = 0.16 + 0.04 * q
    φ = np.linspace(0.0, 1.0, int(n), endpoint=False)
    g = np.zeros_like(φ)
    m1 = φ < Tp
    m2 = (φ >= Tp) & (φ < Tp + Tn)
    g[m1] = 0.5 * (1.0 - np.cos(np.pi * φ[m1] / Tp))
    g[m2] = np.cos(np.pi * (φ[m2] - Tp) / (2.0 * Tn))
    gm = float(np.max(np.abs(g))) or 1.0
    g = g / gm
    g = g - float(np.mean(g))
    gmax = float(np.max(np.abs(g))) or 1.0
    return (g / gmax).astype(np.float64)


_流栈 = np.stack([声门流表(_表长, float(q)) for q in E4._表层])


def _辐射增益(x: np.ndarray) -> float:
    """一阶差分 y[n]=x[n]−x[n−1] 对该块信号的 RMS 增益（解析估计）。"""
    n = len(x)
    if n < 8:
        return 1.0
    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    g = 2.0 * np.sin(np.pi * np.maximum(f, 0.0) / FS)
    P = np.abs(X) ** 2
    总 = float(np.sum(P))
    if 总 <= 1e-30:
        return 1.0
    return max(math.sqrt(float(np.sum(P * g * g)) / 总), 1e-4)


def 合成(帧: dict, 音色: dict = None, 种子: int = 300):
    t0 = time.perf_counter()
    音 = E4._规整音色(音色)
    if not 音色 or "共振峰增益" not in 音色:
        音["共振峰增益"] = _默认共振峰增益
    N = len(帧.get("f0") or [])
    if N == 0:
        return np.zeros(0, dtype=np.float64), dict(
            采样率=FS, 帧率=帧率, 每帧样本=每帧样本, 样本数=0, 帧数=0,
            时长秒=0.0, 峰值=0.0, 峰值缩放=False, 音色指纹=音["_指纹"],
            高频支路dB=音["高频支路dB"], 合成耗时秒=0.0, 实时率=0.0)

    总样本 = N * 每帧样本
    帧轴 = np.arange(N, dtype=np.float64)
    样_idx = np.arange(总样本, dtype=np.float64) / 每帧样本

    def 插(a):
        return np.interp(样_idx, 帧轴, a)

    f0帧 = np.clip(_帧数组(帧, "f0", N, 0.0), 0.0, FS * 0.45)
    幅帧 = np.clip(_帧数组(帧, "幅", N, 0.5), 0.0, 1.5)
    气帧 = np.clip(_帧数组(帧, "气声", N, 0.0), 0.0, 1.0)
    送帧 = np.clip(_帧数组(帧, "送气", N, 0.0), 0.0, 1.0)
    紧帧 = np.clip(_帧数组(帧, "紧喉", N, 0.3), 0.0, 1.0)
    粗帧 = np.clip(_帧数组(帧, "粗糙", N, 0.0), 0.0, 1.0)
    源帧 = np.rint(_帧数组(帧, "声源", N, 1.0)).astype(np.int64)
    源帧 = np.where(源帧 > 0, 1, np.where(源帧 < 0, -1, 0))
    噪形帧 = np.rint(_帧数组(帧, "噪形", N, 0.0)).astype(np.int64)
    噪形帧 = np.where(噪形帧 > 0, 1, 0)

    缺省f = (800.0, 1200.0, 2500.0, 3500.0, 4500.0)
    F帧 = np.empty((5, N), dtype=np.float64)
    for j, 键 in enumerate(("f1", "f2", "f3", "f4", "f5")):
        F帧[j] = _帧数组(帧, 键, N, 缺省f[j])
    BW帧 = np.empty((5, N), dtype=np.float64)
    for j in range(5):
        BW帧[j] = E4.默认带宽[j]
    sF = np.asarray(音["共振峰缩放"], dtype=np.float64)[:, None]
    sB = np.asarray(音["带宽缩放"], dtype=np.float64)[:, None]
    F帧 = np.clip(F帧 * sF, 40.0, FS * 0.5 - 250.0)
    BW帧 = np.clip(BW帧 * sB, 20.0, FS * 0.5 - 250.0)
    if N > 1:
        for j in range(5):
            F帧[j] = _平滑(F帧[j])
    F帧 = np.clip(F帧, 40.0, FS * 0.5 - 250.0)

    噪带s = 帧.get("噪带")
    if 噪带s is None:
        噪带列表 = [None] * N
    else:
        噪带列表 = list(噪带s)
        if len(噪带列表) == 1:
            噪带列表 = 噪带列表 * N
        elif len(噪带列表) != N:
            raise ValueError("噪带长度 %d ≠ f0 长度 %d" % (len(噪带列表), N))

    rng = np.random.default_rng(int(种子))
    白气 = rng.standard_normal(总样本)
    白帧 = rng.standard_normal(总样本)
    _bln, _aln = butter(2, min(7500.0, FS * 0.45) / (FS * 0.5), btype="low")
    帧带 = lfilter(_bln, _aln, 白帧)
    _rn = float(np.std(帧带)) or 1.0
    帧带 = 帧带 / _rn
    _bl, _al = butter(2, min(6000.0, FS * 0.45) / (FS * 0.5), btype="low")
    气带 = lfilter(_bl, _al, 白气)
    _bh, _ah = butter(2, 180.0 / (FS * 0.5), btype="high")
    气带 = lfilter(_bh, _ah, 气带)
    _r = float(np.std(气带)) or 1.0
    气带 = 气带 / _r

    气声样 = 插(气帧) * 音["气声增益"]
    送气样 = 插(送帧) * 音["送气增益"]
    紧喉样 = 插(紧帧)
    粗糙样 = 插(粗帧) * 音["抖动增益"]
    幅样 = 插(幅帧)

    # —— 声门源（与 v5 逐位同）——
    抖动 = 1.0 + 粗糙样 * 0.025 * (
        np.sin(2.0 * math.pi * 13.7 * np.arange(总样本) / FS + 0.5)
        + 0.6 * np.sin(2.0 * math.pi * 31.3 * np.arange(总样本) / FS + 1.9))
    f0样 = 插(f0帧) * 抖动
    相位 = np.mod(np.cumsum(f0样 / FS), 1.0)
    相index = np.minimum((相位 * _表长).astype(np.int64), _表长 - 1)
    层index = np.clip(np.rint(紧喉样 * (_表层数 - 1)).astype(np.int64), 0, _表层数 - 1)
    源周期 = _流栈[层index, 相index]
    if 音["次谐波增益"] > 0.0 and float(np.max(粗糙样)) > 1e-6:
        相位_sub = np.mod(np.cumsum(f0样 * 0.5 / FS), 1.0)
        相index_sub = np.minimum((相位_sub * _表长).astype(np.int64), _表长 - 1)
        源周期 = 源周期 + (0.32 * 粗糙样 * 音["次谐波增益"]) * _流栈[层index, 相index_sub]
    源周期 = 源周期 * (1.0 + 0.15 * 粗糙样 * np.sin(2.0 * math.pi * 33.0
                                                      * np.arange(总样本) / FS + 0.7))
    源周期 = 源周期 * (1.0 + _微颤深 * np.sin(2.0 * math.pi * _微颤Hz
                                             * np.arange(总样本) / FS + 1.3))
    _b, _a = butter(int(_声源低通阶), min(_声源低通Hz, FS * 0.45) / (FS * 0.5), btype="low")
    源滤 = lfilter(_b, _a, 源周期)
    # F4/F5 高通并联支路的激励（直流处增益=0 ⇒ 不会造成低频平台）
    _bh2, _ah2 = butter(2, _高频低切Hz / (FS * 0.5), btype="high")
    源高 = lfilter(_bh2, _ah2, 源滤)
    噪混 = 气带 * (气声样 * _气声源权重 + 送气样 * _气声源权重)
    级联入 = (源滤 + 噪混) * _源标定 * 幅样     # 幅度包络作用在声门源上
    高支入 = 源高 * _源标定 * _高频支路 * 幅样
    g_气直通 = (气声样 * _气声直通权重 + 送气样 * _气声直通权重) * 音["噪声增益"]

    预 = np.zeros(总样本, dtype=np.float64)
    噪预 = np.zeros(总样本, dtype=np.float64)
    噪增益帧 = np.ones(N, dtype=np.float64)
    串zi = [np.zeros(2) for _ in range(5)]        # 串联 F1–F5
    高3zi = np.zeros(2)
    高4zi = np.zeros(2)
    高5zi = np.zeros(2)
    带zi1 = np.zeros(2)
    带zi2 = np.zeros(2)
    形zi = [np.zeros(2) for _ in range(3)]

    for fi in range(N):
        a, b = fi * 每帧样本, min((fi + 1) * 每帧样本, 总样本)
        if a >= b:
            break
        typ = int(源帧[fi])
        if typ == 1:
            # —— v6：Klatt 串联 F1→F5（DC 增益=1，逐节），状态跨帧连续 ——
            y = 级联入[a:b]
            for j in range(5):
                A, B, C = _谐振器系数(F帧[j, fi], BW帧[j, fi])
                y, 串zi[j] = _滤波块(A, B, C, y, 串zi[j])
            # F3/F4/F5 高通并联支路（补串联在高频的累积滚降；高通 ⇒ 不产生低频平台）
            x4 = 高支入[a:b]
            y3, 高3zi = _滤波块(*_谐振器系数(F帧[2, fi], BW帧[2, fi]), x4, 高3zi)
            y4, 高4zi = _滤波块(*_谐振器系数(F帧[3, fi], BW帧[3, fi]), x4, 高4zi)
            y5, 高5zi = _滤波块(*_谐振器系数(F帧[4, fi], BW帧[4, fi]), x4, 高5zi)
            预[a:b] = (y + _高频支路增[0] * y3 + _高频支路增[1] * y4
                       + _高频支路增[2] * y5 + g_气直通[a:b] * 气带[a:b])
        elif typ == 0:
            blk = 帧带[a:b]
            if fi == 0 or int(源帧[fi - 1]) != 0:
                带zi1 = np.zeros(2)
                带zi2 = np.zeros(2)
                形zi = [np.zeros(2) for _ in range(3)]
            if int(噪形帧[fi]) == 1:
                y = np.zeros_like(blk)
                for j, g in ((0, 1.00), (1, 0.62), (2, 0.38)):
                    A, B, C = _谐振器系数(max(F帧[j, fi], _并联下限), 260.0 + 90.0 * j)
                    yj, 形zi[j] = _滤波块(A, B, C, blk, 形zi[j])
                    y += g * yj
                x = y + 0.18 * blk
            else:
                噪带 = 噪带列表[fi]
                if isinstance(噪带, (tuple, list)) and len(噪带) >= 2:
                    Fc = min(max(float(噪带[0]), 40.0), FS * 0.5 - 250.0)
                    BWc = min(max(float(噪带[1]), 30.0), FS * 0.5 - 250.0)
                elif 噪带 is None:
                    Fc, BWc = 3000.0, 2000.0
                else:
                    Fc, BWc = min(max(float(噪带), 40.0), FS * 0.5 - 250.0), 1400.0
                A, B, C = _谐振器系数(Fc, BWc)
                y, 带zi1 = _滤波块(A, B, C, blk, 带zi1)
                y, 带zi2 = _滤波块(A, B, C, y, 带zi2)
                x = y + 0.22 * blk
            噪预[a:b] = x
            增益 = _辐射增益(x)
            目标 = float(幅帧[fi]) * _噪标定 * 音["噪声增益"] * (0.85 + 0.30 * 送帧[fi])
            rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
            噪增益帧[fi] = (目标 / (rms * 增益)) if (rms > 1e-9 and 目标 > 0.0) else 0.0
            z = np.zeros(b - a, dtype=np.float64)
            for j in range(5):
                A, B, C = _谐振器系数(F帧[j, fi], BW帧[j, fi])
                _, 串zi[j] = _滤波块(A, B, C, z, 串zi[j])
            _, 高3zi = _滤波块(*_谐振器系数(F帧[2, fi], BW帧[2, fi]), z, 高3zi)
            _, 高4zi = _滤波块(*_谐振器系数(F帧[3, fi], BW帧[3, fi]), z, 高4zi)
            _, 高5zi = _滤波块(*_谐振器系数(F帧[4, fi], BW帧[4, fi]), z, 高5zi)
        else:
            预[a:b] = 0.0
            z = np.zeros(b - a, dtype=np.float64)
            for j in range(5):
                A, B, C = _谐振器系数(F帧[j, fi], BW帧[j, fi])
                _, 串zi[j] = _滤波块(A, B, C, z, 串zi[j])
            _, 高3zi = _滤波块(*_谐振器系数(F帧[2, fi], BW帧[2, fi]), z, 高3zi)
            _, 高4zi = _滤波块(*_谐振器系数(F帧[3, fi], BW帧[3, fi]), z, 高4zi)
            _, 高5zi = _滤波块(*_谐振器系数(F帧[4, fi], BW帧[4, fi]), z, 高5zi)

    平滑噪增益 = 噪增益帧.copy()
    for fi in range(N):
        if int(源帧[fi]) != 0:
            continue
        lo, hi = max(0, fi - 1), min(N, fi + 2)
        邻 = [噪增益帧[k] for k in range(lo, hi) if int(源帧[k]) == 0]
        if 邻:
            平滑噪增益[fi] = float(sum(邻)) / len(邻)
    噪idx = np.where(源帧 == 0)[0]
    if len(噪idx):
        for fi in range(N):
            k = int(np.searchsorted(噪idx, fi))
            候 = []
            if k < len(噪idx):
                候.append(int(噪idx[k]))
            if k > 0:
                候.append(int(噪idx[k - 1]))
            最近 = min(候, key=lambda q: abs(q - fi))
            平滑噪增益[fi] = 平滑噪增益[最近]
    预 += 噪预 * 插(平滑噪增益)

    # 浊/清边界 2ms 交叉包络（与 v5 同）
    段包络 = np.ones(总样本, dtype=np.float64)
    渐变量 = min(88, 每帧样本 // 4)
    变点 = np.where(np.diff(源帧) != 0)[0] + 1
    for bf in 变点:
        bf = int(bf)
        if not (int(源帧[bf]) == 1 or int(源帧[bf - 1]) == 1):
            continue
        b = bf * 每帧样本
        a0 = max(0, b - 渐变量)
        if b > 0:
            段包络[a0:b] *= np.linspace(1.0, 0.0, b - a0)
        a1 = min(总样本, b + 渐变量)
        段包络[b:a1] = np.minimum(段包络[b:a1], np.linspace(0.0, 1.0, a1 - b))
    预 *= 段包络

    # 唇辐射（单次）
    辐 = np.empty_like(预)
    if 总样本:
        辐[0] = 预[0]
        if 总样本 > 1:
            辐[1:] = 预[1:] - 预[:-1]
    边 = min(int(0.008 * FS), 总样本 // 2)
    if 边 > 1:
        辐[:边] *= np.linspace(0.0, 1.0, 边, endpoint=False)
        辐[-边:] *= np.linspace(1.0, 0.0, 边, endpoint=False)
    out = 辐 * 音["增益"]

    if abs(音["谱倾斜"]) > 1e-9:
        t = 音["谱倾斜"]
        sh = np.empty_like(out)
        sh[0] = out[0]
        if len(out) > 1:
            sh[1:] = out[1:] - t * out[:-1]
        out = sh / (1.0 + 0.5 * abs(t))

    峰0 = float(np.max(np.abs(out))) if 总样本 else 0.0
    if not np.isfinite(峰0):
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        峰0 = float(np.max(np.abs(out)))
    缩放 = False
    if 峰0 > 0.95:
        out = out * (0.95 / 峰0)
        缩放 = True
    峰值 = float(np.max(np.abs(out))) if 总样本 else 0.0
    耗时 = time.perf_counter() - t0
    元信息 = dict(
        采样率=FS, 帧率=帧率, 每帧样本=每帧样本, 样本数=总样本, 帧数=N,
        时长秒=总样本 / FS, 峰值=峰值, 峰值缩放=缩放, 音色指纹=音["_指纹"],
        高频支路dB=音["高频支路dB"],
        共振峰均值=tuple(round(float(x), 1) for x in np.mean(F帧, axis=1)),
        声道="Klatt串联F1-F5 + F4/F5高通并联支路",
        合成耗时秒=round(耗时, 4), 实时率=round(耗时 / max(总样本 / FS, 1e-9), 4),
        f0样=f0样)
    return out, 元信息


# ---------------- 耳听等价自检 ----------------

def _谐波电平(x, f0n=120.0, fmax=6000.0):
    """输出(k·f0) 电平（FFT 谐波峰）→ 归一化 dB。"""
    n = len(x)
    X = np.abs(np.fft.rfft(x * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1.0 / FS)
    ls, db = [], []
    k = 1
    while k * f0n <= fmax:
        c = k * f0n
        m = (f >= c - 15) & (f <= c + 15)
        if np.any(m):
            ls.append(c)
            db.append(20.0 * math.log10(max(float(np.max(X[m])), 1e-12)))
        k += 1
    db = np.array(db)
    return np.array(ls), db - float(np.max(db))


def _带(x):
    P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    P[0] = 0.0
    T = float(P.sum()) or 1.0
    return {k: round(float(P[(f >= a) & (f < b)].sum() / T), 4)
            for k, (a, b) in {"0-500": (0, 500), "500-1k": (500, 1000),
                              "1-2k": (1000, 2000), "2-4k": (2000, 4000),
                              "4-8k": (4000, 8000)}.items()}


def _构(共振峰, 秒=0.30, f0=120.0):
    n = max(int(round(秒 * 帧率)), 2)
    return dict(f0=[float(f0)] * n, f1=[共振峰[0]] * n, f2=[共振峰[1]] * n,
                f3=[共振峰[2]] * n, f4=[共振峰[3]] * n, f5=[共振峰[4]] * n,
                幅=[0.85] * n, 气声=[0.02] * n, 声源=[1] * n, 噪带=[None] * n,
                送气=[0.0] * n, 紧喉=[0.25] * n, 粗糙=[0.03] * n, 噪形=[0] * n)


def _写wav(路径, x):
    os.makedirs(os.path.dirname(路径), exist_ok=True)
    y = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    with wave.open(路径, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(np.rint(y * 32767.0).astype("<i2").tobytes())


元音样 = {"a": (825.0, 1305.0, 2714.0, 3654.0, 4698.0),
         "i": (330.0, 2349.0, 3132.0, 3967.0, 5011.0),
         "u": (352.0, 752.0, 2610.0, 3550.0, 4489.0)}


def 自检() -> dict:
    读数 = {"版本": "v6-44k-串联", "采样率": FS, "元音": {}, "判据": {}}
    波形 = {}
    print("=" * 92)
    print("引擎_v6 自检（耳听等价：输出谱峰是否落在 F1）")
    print("=" * 92)
    for 名, Fs in 元音样.items():
        x, 元 = 合成(_构(Fs), 音色=None, 种子=300)
        ls, db = _谐波电平(x)
        # F1 附近的最高谐波
        i1 = (ls >= Fs[0] * 0.75) & (ls <= Fs[0] * 1.35)
        峰 = float(ls[int(np.argmax(db))])
        低 = float(np.mean(db[:3]))
        F1顶 = float(np.max(db[i1])) if np.any(i1) else -99.0
        读数["元音"][名] = dict(RMS=round(float(np.sqrt(np.mean(x ** 2))), 4),
                             峰值=round(float(np.max(np.abs(x))), 3),
                             最高谐波Hz=峰, 低频平台dB=round(低, 1),
                             F1突出度dB=round(F1顶 - 低, 1), 带=_带(x),
                             RMS_dBFS=round(20 * math.log10(max(
                                 float(np.sqrt(np.mean(x ** 2))), 1e-12)), 2))
        print("[%s] 谱峰 %5.0fHz | 低频台 %6.1f | F1突出 %6.1f | RMS %.4f | 带 %s"
              % (名, 峰, 低, F1顶 - 低, 读数["元音"][名]["RMS"],
                 json.dumps(读数["元音"][名]["带"], ensure_ascii=False)))
        波形[名] = x
    x1, _ = 合成(_构(元音样["a"]), 种子=300)
    x2, _ = 合成(_构(元音样["a"]), 种子=300)
    x3, _ = 合成(_构(元音样["a"]), 种子=301)
    同 = bool(np.array_equal(x1, x2))
    异 = bool(not np.array_equal(x1, x3))
    n帧 = 10 * 帧率
    t = np.arange(n帧, dtype=np.float64) / 帧率
    帧长 = dict(f0=[100.0 + 35.0 * math.sin(2 * math.pi * 2.3 * tt) for tt in t],
                f1=[700.0 + 120.0 * math.sin(2 * math.pi * 1.7 * tt) for tt in t],
                f2=[1200.0 + 150.0 * math.sin(2 * math.pi * 1.1 * tt) for tt in t],
                f3=[2500.0] * n帧, f4=[3500.0] * n帧, f5=[4500.0] * n帧,
                幅=[0.8] * n帧, 气声=[0.02] * n帧, 声源=[1] * n帧,
                噪带=[None] * n帧, 送气=[0.0] * n帧, 紧喉=[0.3] * n帧,
                粗糙=[0.03] * n帧, 噪形=[0] * n帧)
    ta = time.perf_counter()
    合成(帧长, 种子=300)
    t10 = time.perf_counter() - ta
    读数["确定性"] = dict(两趟同=同, 异种子异=异)
    读数["性能"] = dict(十秒耗时秒=round(t10, 4), 实时率=round(t10 / 10.0, 4))
    # a+静+i+静+u wav
    拼帧 = {k: [] for k in ("f0", "f1", "f2", "f3", "f4", "f5", "幅", "气声",
                           "声源", "噪带", "送气", "紧喉", "粗糙", "噪形")}
    for k, 名 in enumerate(("a", "i", "u")):
        f = _构(元音样[名])
        for key in 拼帧:
            拼帧[key].extend(f[key])
        if k < 2:
            静 = dict(f0=120.0, f1=500.0, f2=1500.0, f3=2500.0, f4=3500.0, f5=4500.0,
                      幅=0.0, 气声=0.0, 声源=-1, 噪带=None, 送气=0.0,
                      紧喉=0.3, 粗糙=0.0, 噪形=0)
            for key in 拼帧:
                拼帧[key].extend([静[key]] * 10)
    xw, _ = 合成(拼帧, 种子=300)
    wav路径 = os.path.join(产物临时, "引擎自检_v6.wav")
    _写wav(wav路径, xw)
    突s = [读数["元音"][k]["F1突出度dB"] for k in 读数["元音"]]
    # F1 < 500Hz 的元音（/i/ /u/）在 F0=120 时低阶谐波本就落在 F1 裙边，低频谱台
    # 天然最强 ⇒ 只要求谱峰落在 F1 带内；F1 ≥ 500Hz 才要求 F1 突出 ≥ 3dB。
    落F1 = []
    for k, v in 读数["元音"].items():
        f1 = 元音样[k][0]
        if f1 >= 500.0:
            落F1.append(bool(v["最高谐波Hz"] <= f1 * 1.30 and v["F1突出度dB"] >= 3.0))
        else:
            落F1.append(bool(f1 * 0.65 <= v["最高谐波Hz"] <= f1 * 1.55))
    读数["判据"] = dict(
        谱峰落F1=bool(all(落F1)),
        F1突出度_gt_3dB=bool(all(d >= 3.0 for k, d in zip(读数["元音"], 突s)
                                if 元音样[k][0] >= 500.0)),
        无NaN=bool(all(np.all(np.isfinite(合成(_构(元音样[k]))[0])) for k in 元音样)),
        两趟同=同, 性能ok=bool(t10 < 2.0))
    读数["wav"] = wav路径
    print("-" * 92)
    print("确定性 两趟同=%s 异种子异=%s | 性能 10s=%.3fs 实时率 %.4f"
          % (同, 异, t10, t10 / 10.0))
    print("判据：" + " ".join("%s=%s" % (k, v) for k, v in 读数["判据"].items()))
    print("总体通过：%s" % all(读数["判据"].values()))
    os.makedirs(os.path.join(产物临时, "..", "判据"), exist_ok=True)
    with open(os.path.join(产物临时, "引擎自检_v6.json"), "w", encoding="utf-8") as f:
        json.dump(读数, f, ensure_ascii=False, indent=2)
    return 读数


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    import json
    自检()
