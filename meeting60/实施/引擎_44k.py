# -*- coding: utf-8 -*-
"""meeting60/实施/引擎_44k.py —— 44.1kHz 高保真源-滤波器声带引擎（v3）。

背景：旧版（meeting59/v2）16kHz 三共振峰合成器输出「严重失真 + 非常模糊」
（95% 滚降仅 1.0–2.6kHz、4–8k 能量 0.6–2.5%，靠 ±8dB EQ + tanh 凑近参考）。

本引擎是纯 numpy/scipy 自研 DSP（C1/C10：无 TTS / RVC / 神经声码器 / 外部模型）：
    声门源（Rosenberg/KLGLOTT 1024 点脉冲表，紧喉→开商，粗糙→抖动+次谐波，
            气声→高通白噪混入）
    → 声道 F1–F5 五级级联 biquad 谐振器（逐帧分块 lfilter，状态跨帧）
      + F4/F5 并联高频支路（默认 4.5dB，保 4–8k 能量）
    → 唇辐射（一阶差分，+6dB/oct）
    → 逐样本线性幅度包络 → 峰值保护（只整体缩放，**无 tanh/硬削波**）

API
---
``合成(帧: dict, 音色: dict = None, 种子: int = 300) -> (波形 np.float64, 元信息 dict)``

帧（100Hz，每帧 441 样本；长度 = len(f0)*441）：
    f0      基频 Hz
    f1..f5  F1–F5 Hz（f4/f5 缺省 3500/4500，f3 缺省 2500）
    幅      幅度 0..1
    气声    气声度 0..1
    声源    1=浊（声门源）/ 0=噪（噪带成形噪声）/ -1=静默
    噪带    (Fc, BW) 或 None（声源=0 时缺省 (3000,2000)）
    送气    送气量 0..1
    紧喉    紧喉度 0..1（→ Tp=0.40−0.16*紧喉, Tn=0.16+0.04*紧喉）
    粗糙    粗糙度 0..1（→ ±抖动 + 次谐波）

音色（可选；缺省即中性音色）：
    共振峰缩放  float 或 3/5 元组（3 元组按 F1/F2/F3 同比例延到 F4/F5）
    带宽缩放    float 或 3/5 元组
    高频支路dB  F4/F5 并联支路增益（默认 4.5，建议 3–6）
    气声增益 / 送气增益 / 噪声增益 / 抖动增益 / 次谐波增益 / 谱倾斜 / 增益
    另外支持 共振峰缩放/带宽缩放 的 5 元组逐峰缩放。

自检（真跑）：``python 引擎_44k.py`` → 产物/临时/引擎自检.wav + 引擎自检读数.json
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import wave

import numpy as np
from scipy.signal import lfilter, butter

FS = 44100                    # 采样率
帧率 = 100                    # 控制帧率 100Hz
每帧样本 = FS // 帧率         # 441
_本目录 = os.path.dirname(os.path.abspath(__file__))
产物临时 = os.path.normpath(os.path.join(_本目录, "..", "产物", "临时"))

# 共振峰带宽（Hz；Klatt 量级：70/90/130/200/280）
默认带宽 = (70.0, 90.0, 130.0, 200.0, 280.0)
# 声门脉冲表
表长 = 1024
表层数 = 17                   # 紧喉 0..1 量化为 17 层
_表层 = np.linspace(0.0, 1.0, 表层数)

# —— 引擎内部标定常数（自检实测标定，见 引擎自检读数.json）——
_源标定 = 0.22                # 声门源整体标定：典型元音 幅=1 峰值≈0.8（保 幅 动态不被峰值归一吃掉）
_并联源权重 = 0.08            # 进 F4/F5 并联支路的声门源比例（再叠 高频支路dB）
_气声源权重 = 0.012           # 气声/送气噪声在级联输入中的混合量（v3：带限噪声后重新标定，气声 0.2 ≈ -20dB）
_气声直通权重 = 0.004         # 气声/送气噪声高频直通量（v3：原 0.55 过亮，8–16k 曾达 23%）
_噪声基幅 = 0.15              # 噪声帧（声源=0）相对浊音帧的基幅标定（v3：擦音比浊音低约 10dB）
_噪声高频 = 0.10              # 噪声帧高频提升量（v3：原 0.55 过亮）
_噪声超高频 = 0.55            # 噪声帧 8–16kHz 气声/齿音成分（只加在清辅音帧，补参考的 8–16k）
# v3 修复：声门源经「声源低通」再进声道（标准声源谱 -12dB/oct 附近；否则源+唇辐射净 0dB/oct 高频爆炸）
_声源低通Hz = 3800.0
_声源低通阶 = 2
# v3 修复：浊音段输出一阶低通（≈−6dB/oct，截 2.6k）——压低 4–8k 的嘶嘶，又保留清辅音亮度
_浊音低通Hz = 4200.0


def _夹(v: float, lo: float, hi: float) -> float:
    return float(min(max(float(v), lo), hi))


def _五元组(v) -> tuple:
    """音色缩放量 → 5 元组（float → 全同；3 元组 → F4/F5 沿用 F3）。"""
    if isinstance(v, (int, float)):
        x = float(v)
        return (x, x, x, x, x)
    try:
        a = [float(t) for t in v]
    except TypeError:
        return (1.0, 1.0, 1.0, 1.0, 1.0)
    if len(a) >= 5:
        return tuple(a[:5])
    if len(a) == 3:
        return (a[0], a[1], a[2], a[2], a[2])
    if len(a) == 1:
        return (a[0],) * 5
    while len(a) < 3:
        a.append(1.0)
    return (a[0], a[1], a[2], a[2], a[2])


def _规整音色(音色: dict | None) -> dict:
    音 = dict(音色 or {})
    出 = dict(
        共振峰缩放=_五元组(音.get("共振峰缩放", 1.0)),
        带宽缩放=_五元组(音.get("带宽缩放", 1.0)),
        高频支路dB=_夹(音.get("高频支路dB", -2.0), -12.0, 12.0),
        气声增益=_夹(音.get("气声增益", 1.0), 0.0, 4.0),
        送气增益=_夹(音.get("送气增益", 1.0), 0.0, 4.0),
        噪声增益=_夹(音.get("噪声增益", 1.0), 0.0, 4.0),
        抖动增益=_夹(音.get("抖动增益", 1.0), 0.0, 4.0),
        次谐波增益=_夹(音.get("次谐波增益", 1.0), 0.0, 4.0),
        谱倾斜=_夹(音.get("谱倾斜", 0.0), -0.95, 0.95),
        增益=_夹(音.get("增益", 1.0), 0.0, 4.0),
    )
    出["_指纹"] = _音色指纹(音)
    return 出


def _音色指纹(音色: dict | None) -> str:
    if not 音色:
        return ""
    s = json.dumps(音色, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# ---------------- 声门源：Rosenberg/KLGLOTT 脉冲表 ----------------

def 声门脉冲表(n: int = 表长, 紧喉: float = 0.0) -> np.ndarray:
    """单周期 Rosenberg 脉冲查表（相位归一 0..1）。

    紧喉 → 开商：Tp = 0.40 − 0.16*紧喉，Tn = 0.16 + 0.04*紧喉。
    """
    q = _夹(紧喉, 0.0, 1.0)
    Tp = 0.40 - 0.16 * q
    Tn = 0.16 + 0.04 * q
    φ = np.linspace(0.0, 1.0, int(n), endpoint=False)
    g = np.zeros_like(φ)
    m1 = φ < Tp
    m2 = (φ >= Tp) & (φ < Tp + Tn)
    g[m1] = 0.5 * (1.0 - np.cos(np.pi * φ[m1] / Tp))
    g[m2] = np.cos(np.pi * (φ[m2] - Tp) / (2.0 * Tn))
    # v3 修复：改用「声门流导数」当声源（原来用声门流形状 + 唇辐射差分 ⇒ 净 +6dB/oct 高频爆炸）
    gd = np.diff(np.concatenate([g, g[:1]]))
    m = float(np.max(np.abs(gd))) if gd.size else 0.0
    return (gd / m if m > 0 else gd).astype(np.float64)


def _脉冲栈() -> np.ndarray:
    """17 层（紧喉 0..1）脉冲表栈，形状 (17, 1024)。"""
    return np.stack([声门脉冲表(表长, float(q)) for q in _表层])


_表栈 = _脉冲栈()


# ---------------- biquad 谐振器 ----------------

def 谐振器系数(F: float, BW: float, fs: int = FS) -> tuple:
    """二阶谐振器系数（Klatt 形式，DC 增益=1）：y=A*x+B*y1+C*y2。"""
    F = _夹(F, 20.0, fs * 0.5 - 200.0)
    BW = _夹(BW, 15.0, fs * 0.5 - 200.0)
    T = 1.0 / fs
    C = -math.exp(-2.0 * math.pi * BW * T)
    B = 2.0 * math.exp(-math.pi * BW * T) * math.cos(2.0 * math.pi * F * T)
    A = 1.0 - B - C
    return float(A), float(B), float(C)


def _滤波块(A: float, B: float, C: float, x: np.ndarray, zi: np.ndarray):
    """块滤波二阶 IIR（scipy 有状态；状态跨块保持）。"""
    y, zf = lfilter([A], [1.0, -B, -C], x, zi=zi)
    return y, zf


def _一阶低通(x: np.ndarray, 系数: float) -> np.ndarray:
    return lfilter([1.0 - 系数], [1.0, -系数], x)


# ---------------- 主合成 ----------------

def _帧数组(帧: dict, 键: str, N: int, 缺省) -> np.ndarray:
    v = 帧.get(键)
    if v is None:
        a = np.full(N, float(缺省), dtype=np.float64)
    else:
        a = np.asarray(v, dtype=np.float64).reshape(-1)
        if a.size == 1:
            a = np.full(N, float(a[0]), dtype=np.float64)
        elif a.size != N:
            raise ValueError("帧键 %r 长度 %d ≠ f0 长度 %d" % (键, a.size, N))
    a = a.copy()
    if not np.all(np.isfinite(a)):
        a = np.nan_to_num(a, nan=float(缺省), posinf=float(缺省),
                          neginf=float(缺省))
    return a


def _平滑(a: np.ndarray, α: float = 0.55) -> np.ndarray:
    """帧级一阶平滑：消除逐帧系数台阶（与 meeting32 同思路）。"""
    b = a.copy()
    for i in range(1, len(b)):
        b[i] = b[i - 1] + α * (b[i] - b[i - 1])
    return b


def 合成(帧: dict, 音色: dict = None, 种子: int = 300):
    """44.1kHz 源-滤波器合成。

    返回 ``(波形 np.float64, 元信息 dict)``；波形长度 = len(帧['f0'])*441，
    峰值 ≤ 0.95（>0.95 时整体缩放），无 NaN，纯确定性（np.random.default_rng(种子)）。
    """
    t0 = time.perf_counter()
    音 = _规整音色(音色)
    N = len(帧.get("f0") or [])
    if N == 0:
        return np.zeros(0, dtype=np.float64), dict(
            采样率=FS, 帧率=帧率, 每帧样本=每帧样本, 样本数=0, 帧数=0,
            时长秒=0.0, 峰值=0.0, 峰值缩放=False, 音色指纹=音["_指纹"],
            高频支路dB=音["高频支路dB"], 合成耗时秒=0.0, 实时率=0.0)

    总样本 = N * 每帧样本
    帧轴 = np.arange(N, dtype=np.float64)
    样_idx = np.arange(总样本, dtype=np.float64) / 每帧样本

    def 插帧级(a: np.ndarray) -> np.ndarray:
        return np.interp(样_idx, 帧轴, a)

    # —— 帧参数（读入副本；不修改调用方 dict）——
    f0帧 = _帧数组(帧, "f0", N, 0.0)
    f0帧 = np.clip(f0帧, 0.0, FS * 0.45)
    幅帧 = np.clip(_帧数组(帧, "幅", N, 0.5), 0.0, 1.5)
    气帧 = np.clip(_帧数组(帧, "气声", N, 0.0), 0.0, 1.0)
    送帧 = np.clip(_帧数组(帧, "送气", N, 0.0), 0.0, 1.0)
    紧帧 = np.clip(_帧数组(帧, "紧喉", N, 0.3), 0.0, 1.0)
    粗帧 = np.clip(_帧数组(帧, "粗糙", N, 0.0), 0.0, 1.0)
    源帧 = np.rint(_帧数组(帧, "声源", N, 1.0)).astype(np.int64)
    源帧 = np.where(源帧 > 0, 1, np.where(源帧 < 0, -1, 0))
    缺省f = (800.0, 1200.0, 2500.0, 3500.0, 4500.0)
    F帧 = np.empty((5, N), dtype=np.float64)
    for j, 键 in enumerate(("f1", "f2", "f3", "f4", "f5")):
        F帧[j] = _帧数组(帧, 键, N, 缺省f[j])
    BW帧 = np.empty((5, N), dtype=np.float64)
    for j in range(5):
        BW帧[j] = 默认带宽[j]
    sF = np.asarray(音["共振峰缩放"], dtype=np.float64)[:, None]
    sB = np.asarray(音["带宽缩放"], dtype=np.float64)[:, None]
    F帧 = np.clip(F帧 * sF, 40.0, FS * 0.5 - 250.0)
    BW帧 = np.clip(BW帧 * sB, 20.0, FS * 0.5 - 250.0)
    if N > 1:
        for j in range(5):
            F帧[j] = _平滑(F帧[j])
    F帧 = np.clip(F帧, 40.0, FS * 0.5 - 250.0)

    # 噪带：逐帧 (Fc, BW) 或 None
    噪带s = 帧.get("噪带")
    if 噪带s is None:
        噪带列表 = [None] * N
    else:
        噪带列表 = list(噪带s)
        if len(噪带列表) == 1:
            噪带列表 = 噪带列表 * N
        elif len(噪带列表) != N:
            raise ValueError("噪带长度 %d ≠ f0 长度 %d" % (len(噪带列表), N))

    # —— 噪声源（一次抽取；顺序固定 ⇒ 确定性）——
    rng = np.random.default_rng(int(种子))
    白气 = rng.standard_normal(总样本)          # 气声/送气噪声
    白帧 = rng.standard_normal(总样本)          # 噪声帧成形噪声
    # v3：气声改用带限噪声（白噪 → 6k 低通 → 200Hz 高通 → 单位 RMS），
    # 原来用一阶差分（+6dB/oct）导致 8–16k 占比爆炸、听感嘶嘶
    _bl, _al = butter(2, min(6000.0, FS * 0.45) / (FS * 0.5), btype="low")
    气带 = lfilter(_bl, _al, 白气)
    _bh, _ah = butter(2, 180.0 / (FS * 0.5), btype="high")
    气带 = lfilter(_bh, _ah, 气带)
    _r = float(np.std(气带)) or 1.0
    气带 = 气带 / _r
    帧_hp = np.diff(白帧, prepend=0.0)
    # v3：8–16k 超高频噪声（只进清辅音帧；补参考 8–16k 的齿音/气声质感）
    _bh8, _ah8 = butter(2, 8000.0 / (FS * 0.5), btype="high")
    超高频 = lfilter(_bh8, _ah8, 白帧)
    _r8 = float(np.std(超高频)) or 1.0
    超高频 = 超高频 / _r8

    气声样 = 插帧级(气帧) * 音["气声增益"]
    送气样 = 插帧级(送帧) * 音["送气增益"]
    紧喉样 = 插帧级(紧帧)
    粗糙样 = 插帧级(粗帧) * 音["抖动增益"]

    # —— 声门源：相位 cumsum(f0/FS) → 1024 点脉冲表 ——
    抖动 = 1.0 + 粗糙样 * 0.025 * (
        np.sin(2.0 * math.pi * 13.7 * np.arange(总样本) / FS + 0.5)
        + 0.6 * np.sin(2.0 * math.pi * 31.3 * np.arange(总样本) / FS + 1.9))
    相位 = np.mod(np.cumsum(f0帧_to_样(f0帧, 插帧级, 总样本, 抖动) / FS), 1.0)
    相index = np.minimum((相位 * 表长).astype(np.int64), 表长 - 1)
    层index = np.clip(np.rint(紧喉样 * (表层数 - 1)).astype(np.int64), 0, 表层数 - 1)
    源周期 = _表栈[层index, 相index]
    # 次谐波（粗糙度）：半频脉冲混入
    if 音["次谐波增益"] > 0.0 and float(np.max(粗糙样)) > 1e-6:
        相位_sub = np.mod(np.cumsum(f0帧_to_样(f0帧, 插帧级, 总样本, 抖动) * 0.5 / FS), 1.0)
        相index_sub = np.minimum((相位_sub * 表长).astype(np.int64), 表长 - 1)
        源周期 = 源周期 + (0.32 * 粗糙样 * 音["次谐波增益"]) * _表栈[层index, 相index_sub]
    # 粗糙度 AM（黏膜波不规则）
    源周期 = 源周期 * (1.0 + 0.15 * 粗糙样 * np.sin(2.0 * math.pi * 33.0
                                                          * np.arange(总样本) / FS + 0.7))
    # v3：声源低通（声门源谱倾斜；否则「声门流导数 + 唇辐射」净 0dB/oct，4–8k 占比爆炸）
    _b, _a = butter(int(_声源低通阶), min(_声源低通Hz, FS * 0.45) / (FS * 0.5), btype="low")
    源周期滤 = lfilter(_b, _a, 源周期)
    # 气声/送气：高通白噪混入级联输入（经声道滤波的呼吸噪声）
    噪混 = 气带 * (气声样 * _气声源权重 + 送气样 * _气声源权重)
    级联入 = (源周期滤 + 噪混) * _源标定

    # —— 声道：F1–F5 级联 + F4/F5 并联高频支路 ——
    g_hi = 10.0 ** (音["高频支路dB"] / 20.0)
    g_气直通 = (气声样 * _气声直通权重 + 送气样 * _气声直通权重) * 音["噪声增益"]
    并联入 = 源周期滤 * _并联源权重 + 气带 * (
        气声样 * 0.03 + 送气样 * 0.03) * 音["噪声增益"]

    预 = np.zeros(总样本, dtype=np.float64)
    级联zi = [np.zeros(2) for _ in range(5)]
    hi4zi = np.zeros(2)
    hi5zi = np.zeros(2)
    噪zi1 = np.zeros(2)
    噪zi2 = np.zeros(2)
    段包络 = np.ones(总样本, dtype=np.float64)
    变点 = np.where(np.diff(源帧) != 0)[0] + 1
    渐变量 = min(88, 每帧样本 // 4)             # ≈2ms
    for b in 变点:
        b = int(b)
        a0 = max(0, b - 渐变量)
        if b > 0:
            段包络[a0:b] *= np.linspace(1.0, 0.0, b - a0)
        a1 = min(总样本, b + 渐变量)
        段包络[b:a1] = np.minimum(段包络[b:a1], np.linspace(0.0, 1.0, a1 - b))
    段包络[:min(渐变量, 总样本)] *= np.linspace(0.0, 1.0, min(渐变量, 总样本))
    段包络[-渐变量:] *= np.linspace(1.0, 0.0, min(渐变量, 总样本))

    # 噪声帧幅度系数（幅 × 送气）
    帧增益 = 幅帧.copy()
    噪掩 = 源帧 == 0
    帧增益[噪掩] = 帧增益[噪掩] * (0.75 + 0.55 * 送帧[噪掩])
    静掩 = 源帧 == -1
    帧增益[静掩] = 0.0
    增益样 = 插帧级(帧增益) * 音["增益"]

    for fi in range(N):
        a, b = fi * 每帧样本, min((fi + 1) * 每帧样本, 总样本)
        if a >= b:
            break
        typ = int(源帧[fi])
        if fi == 0 or typ != int(源帧[fi - 1]):
            级联zi = [np.zeros(2) for _ in range(5)]
            hi4zi = np.zeros(2)
            hi5zi = np.zeros(2)
            噪zi1 = np.zeros(2)
            噪zi2 = np.zeros(2)
        if typ == 1:
            x = 级联入[a:b]
            y = x
            for j in range(5):
                A, B, C = 谐振器系数(F帧[j, fi], BW帧[j, fi])
                y, 级联zi[j] = _滤波块(A, B, C, y, 级联zi[j])
            xh = 并联入[a:b]
            A4, B4, C4 = 谐振器系数(F帧[3, fi], BW帧[3, fi])
            y4, hi4zi = _滤波块(A4, B4, C4, xh, hi4zi)
            A5, B5, C5 = 谐振器系数(F帧[4, fi], BW帧[4, fi])
            y5, hi5zi = _滤波块(A5, B5, C5, xh, hi5zi)
            预[a:b] = y + g_hi * (y4 + y5) + g_气直通[a:b] * 气带[a:b]
        elif typ == 0:
            噪带 = 噪带列表[fi]
            if isinstance(噪带, (tuple, list)) and len(噪带) >= 2:
                Fc = _夹(float(噪带[0]), 40.0, FS * 0.5 - 250.0)
                BWc = _夹(float(噪带[1]), 30.0, FS * 0.5 - 250.0)
            elif 噪带 is None:
                Fc, BWc = 3000.0, 2000.0
            else:
                Fc, BWc = _夹(float(噪带), 40.0, FS * 0.5 - 250.0), 1200.0
            x = 帧_hp[a:b]
            A, B, C = 谐振器系数(Fc, BWc)
            x, 噪zi1 = _滤波块(A, B, C, x, 噪zi1)
            x, 噪zi2 = _滤波块(A, B, C, x, 噪zi2)
            # 二级谐振后再补高频直通（擦音/爆破的 4k+ 能量）
            x = x + _噪声高频 * 帧_hp[a:b] * (0.5 + 0.5 * 送气样[a:b])
            # 8–16k 超高频（齿音/气声质感；只作用于噪声帧，浊音段不加）
            x = x + _噪声超高频 * 超高频[a:b] * (0.35 + 0.65 * 送气样[a:b])
            预[a:b] = x * _噪声基幅 * 音["噪声增益"]
        else:
            预[a:b] = 0.0
    预 *= 段包络

    # —— 唇辐射：一阶差分（+6dB/oct），在谐振器之后 ——
    辐 = np.empty_like(预)
    if 总样本:
        辐[0] = 预[0]
        if 总样本 > 1:
            辐[1:] = 预[1:] - 预[:-1]
    # —— 浊音段高频自然衰减（v3 修复：只作用于浊音，清辅音/擦音保留亮度）——
    #     一阶低通 ≈ −6dB/oct @3k；v2 是过闷，这里要「自然但不嘶」
    if 总样本:
        α = math.exp(-2.0 * math.pi * float(_浊音低通Hz) / FS)
        zi1 = np.zeros(1)
        for fi in range(N):
            a, b = fi * 每帧样本, min((fi + 1) * 每帧样本, 总样本)
            if a >= b:
                break
            if int(源帧[fi]) == 1:
                辐[a:b], zi1 = lfilter([1.0 - α], [1.0, -α], 辐[a:b], zi=zi1)
            else:
                zi1 = np.zeros(1)
    # —— 逐样本线性幅度包络 ——
    out = 辐 * 增益样
    # 谱倾斜（可选；默认 0，避免 v2 那种 ±8dB 静态 EQ）
    if abs(音["谱倾斜"]) > 1e-9:
        t = 音["谱倾斜"]
        sh = np.empty_like(out)
        sh[0] = out[0]
        if len(out) > 1:
            sh[1:] = out[1:] - t * out[:-1]
        out = sh / (1.0 + 0.5 * abs(t))
    # —— 数值健康 + 峰值保护（禁 tanh/硬削波：只整体缩放）——
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
        共振峰均值=tuple(round(float(x), 1) for x in 帧均值(F帧)),
        合成耗时秒=round(耗时, 4), 实时率=round(耗时 / max(总样本 / FS, 1e-9), 4),
        f0样=插帧级(f0帧),
    )
    return out, 元信息


def f0帧_to_样(f0帧: np.ndarray, 插帧级, 总样本: int, 抖动: np.ndarray) -> np.ndarray:
    """f0 帧 → 逐样本（含抖动因子）。"""
    return 插帧级(f0帧) * 抖动


def 帧均值(F帧: np.ndarray) -> np.ndarray:
    return np.mean(F帧, axis=1)


# ---------------- 自检 ----------------

def _谱指标(x: np.ndarray, fs: int = FS, 分析区间=(0.30, 0.75)) -> dict:
    """功率谱能量占比 / 95% 滚降 / 谱重心（分析中段，避开段包络）。"""
    n = len(x)
    a = int(n * 分析区间[0])
    b = int(n * 分析区间[1])
    seg = np.asarray(x[a:b], dtype=np.float64)
    seg = seg - float(np.mean(seg)) if len(seg) else seg
    m = len(seg)
    if m < 256:
        seg, m = np.asarray(x, dtype=np.float64), n
    win = np.hanning(m)
    X = np.fft.rfft(seg * win)
    P = np.abs(X) ** 2
    f = np.fft.rfftfreq(m, 1.0 / fs)
    总 = float(np.sum(P)) or 1.0
    带 = lambda lo, hi: float(np.sum(P[(f >= lo) & (f < hi)])) / 总
    cum = np.cumsum(P) / 总
    idx95 = int(np.searchsorted(cum, 0.95))
    idx95 = min(idx95, len(f) - 1)
    return dict(
        谱重心Hz=float(np.sum(f * P) / 总),
        滚降95Hz=float(f[idx95]),
        能量_1_4k=带(1000.0, 4000.0),
        能量_4_8k=带(4000.0, 8000.0),
        能量_8_16k=带(8000.0, 16000.0),
        总能量=总,
    )


def _峰位(x: np.ndarray, 目标s, fs: int = FS, 分析区间=(0.30, 0.75)) -> list:
    """FFT 谱峰位：在目标 ±15% 窗内取功率谱最大点（报告真误差）。"""
    n = len(x)
    a = int(n * 分析区间[0])
    b = int(n * 分析区间[1])
    seg = np.asarray(x[a:b], dtype=np.float64)
    seg = seg - float(np.mean(seg))
    m = len(seg)
    win = np.hanning(m)
    X = np.fft.rfft(seg * win)
    P = np.abs(X) ** 2
    f = np.fft.rfftfreq(m, 1.0 / fs)
    out = []
    for 目标 in 目标s:
        lo = 目标 * 0.85
        hi = 目标 * 1.15
        mask = (f >= lo) & (f <= hi)
        if not np.any(mask):
            out.append(dict(目标=float(目标), 实测=0.0, 误差=1.0))
            continue
        ff = f[mask]
        pp = P[mask]
        k = int(np.argmax(pp))
        # 抛物线插值（半 bin 精度）
        if 0 < k < len(pp) - 1:
            y0, y1, y2 = pp[k - 1], pp[k], pp[k + 1]
            den = y0 - 2.0 * y1 + y2
            δ = 0.5 * (y0 - y2) / den if abs(den) > 1e-30 else 0.0
            δ = max(-0.5, min(0.5, δ))
        else:
            δ = 0.0
        fm = float(ff[k] + δ * (ff[1] - ff[0]) if len(ff) > 1 else ff[k])
        out.append(dict(目标=float(目标), 实测=round(fm, 1),
                        误差=abs(fm - 目标) / 目标))
    return out


def _构元音帧(共振峰, 秒: float = 0.40, f0: float = 100.0, 气声: float = 0.12,
              送气: float = 0.04) -> dict:
    n = max(int(round(秒 * 帧率)), 2)
    return dict(
        f0=[float(f0)] * n,
        f1=[float(共振峰[0])] * n,
        f2=[float(共振峰[1])] * n,
        f3=[float(共振峰[2])] * n,
        f4=[float(共振峰[3])] * n,
        f5=[float(共振峰[4])] * n,
        幅=[0.85] * n,
        气声=[float(气声)] * n,
        声源=[1] * n,
        噪带=[None] * n,
        送气=[float(送气)] * n,
        紧喉=[0.25] * n,
        粗糙=[0.04] * n,
    )


def _写wav(路径: str, x: np.ndarray, fs: int = FS) -> None:
    os.makedirs(os.path.dirname(路径), exist_ok=True)
    y = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    pcm = np.rint(y * 32767.0).astype("<i2")
    with wave.open(路径, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(pcm.tobytes())


def 自检() -> dict:
    """真跑自检：a/i/u 五个峰位、4–8k/8–16k 能量、确定性、性能，落盘 wav+读数。"""
    元音 = {"a": (800.0, 1200.0, 2500.0, 3500.0, 4500.0),
            "i": (300.0, 2200.0, 3000.0, 3700.0, 4800.0),
            "u": (320.0, 700.0, 2500.0, 3400.0, 4400.0)}
    读数 = {"版本": "v3-44k", "采样率": FS, "帧率": 帧率,
            "每帧样本": 每帧样本, "判据": {}, "元音": {}}
    波形表 = {}
    通过 = True
    for 名, 峰s in 元音.items():
        帧 = _构元音帧(峰s)
        x, 元 = 合成(帧, 音色=None, 种子=300)
        指标 = _谱指标(x)
        峰 = _峰位(x, 峰s)
        无nan = bool(np.all(np.isfinite(x)))
        峰误 = max(p["误差"] for p in 峰)
        rec = dict(时长秒=round(len(x) / FS, 3), 峰值=round(float(np.max(np.abs(x))), 4),
                   无NaN=无nan, **指标,
                   峰位=峰, 最大峰位误差=round(峰误, 4),
                   通过=(无nan and 峰误 < 0.08
                         and 指标["能量_4_8k"] > 0.03
                         and 指标["能量_8_16k"] > 0.002
                         and float(np.max(np.abs(x))) <= 1.0))
        读数["元音"][名] = rec
        波形表[名] = x
        通过 = 通过 and rec["通过"]

    # 两趟逐位同（同一帧 + 同一种子）
    帧a = _构元音帧(元音["a"])
    x1, _ = 合成(帧a, 种子=300)
    x2, _ = 合成(帧a, 种子=300)
    x3, _ = 合成(帧a, 种子=301)
    sha1 = hashlib.sha256(np.ascontiguousarray(x1).tobytes()).hexdigest()
    sha2 = hashlib.sha256(np.ascontiguousarray(x2).tobytes()).hexdigest()
    sha3 = hashlib.sha256(np.ascontiguousarray(x3).tobytes()).hexdigest()
    相同 = bool(sha1 == sha2)
    异种子异 = bool(sha1 != sha3)

    # 性能：10s 连续浊音（1000 帧）
    n帧 = 10 * 帧率
    t = np.arange(n帧, dtype=np.float64) / 帧率
    帧长 = dict(
        f0=[100.0 + 35.0 * math.sin(2 * math.pi * 2.3 * tt) for tt in t],
        f1=[700.0 + 120.0 * math.sin(2 * math.pi * 1.7 * tt) for tt in t],
        f2=[1200.0 + 150.0 * math.sin(2 * math.pi * 1.1 * tt) for tt in t],
        f3=[2500.0] * n帧, f4=[3500.0] * n帧, f5=[4500.0] * n帧,
        幅=[0.8] * n帧, 气声=[0.10] * n帧, 声源=[1] * n帧,
        噪带=[None] * n帧, 送气=[0.03] * n帧, 紧喉=[0.3] * n帧,
        粗糙=[0.03] * n帧)
    t1 = time.perf_counter()
    xp, 元p = 合成(帧长, 种子=300)
    t10 = time.perf_counter() - t1
    实时率 = t10 / 10.0

    # 拼接 wav：a + 100ms 静 + i + 100ms 静 + u
    静帧 = 10
    拼帧 = dict(f0=[], f1=[], f2=[], f3=[], f4=[], f5=[], 幅=[], 气声=[],
                声源=[], 噪带=[], 送气=[], 紧喉=[], 粗糙=[])
    for k, 名 in enumerate(("a", "i", "u")):
        帧 = _构元音帧(元音[名])
        for key in 拼帧:
            拼帧[key].extend(帧[key])
        if k < 2:
            静值 = {"f0": 100.0, "f1": 500.0, "f2": 1500.0, "f3": 2500.0,
                    "f4": 3500.0, "f5": 4500.0, "幅": 0.0, "气声": 0.0,
                    "声源": -1, "噪带": None, "送气": 0.0,
                    "紧喉": 0.3, "粗糙": 0.0}
            for key in 拼帧:
                拼帧[key].extend([静值[key]] * 静帧)
    xw, _ = 合成(拼帧, 种子=300)
    wav路径 = os.path.join(产物临时, "引擎自检.wav")
    _写wav(wav路径, xw)

    读数["确定性"] = dict(两趟sha256同=相同, sha256=sha1[:32],
                          异种子不同=异种子异)
    读数["性能"] = dict(十秒耗时秒=round(t10, 4), 实时率=round(实时率, 4),
                        判据_lt_1p5s=bool(t10 < 1.5))
    读数["判据"] = dict(
        峰位误差_lt_8pct=bool(all(读数["元音"][k]["最大峰位误差"] < 0.08
                                 for k in 元音)),
        能量4_8k_gt_3pct=bool(all(读数["元音"][k]["能量_4_8k"] > 0.03
                                  for k in 元音)),
        能量8_16k_gt_0p2pct=bool(all(读数["元音"][k]["能量_8_16k"] > 0.002
                                     for k in 元音)),
        无NaN=bool(all(读数["元音"][k]["无NaN"] for k in 元音)),
        峰值_le_1=bool(all(读数["元音"][k]["峰值"] <= 1.0 for k in 元音)),
        两趟同=相同, 性能=bool(t10 < 1.5))
    读数["wav"] = wav路径
    读数["总体通过"] = bool(通过 and 相同 and 异种子异 and t10 < 1.5
                            and all(读数["判据"].values()))
    json路径 = os.path.join(产物临时, "引擎自检读数.json")
    os.makedirs(产物临时, exist_ok=True)
    with open(json路径, "w", encoding="utf-8") as f:
        json.dump(读数, f, ensure_ascii=False, indent=2)
    读数["读数json"] = json路径
    return 读数


def _打印自检(读数: dict) -> None:
    print("=" * 78)
    print("引擎_44k 自检  |  44.1kHz 源-滤波器  |  a/i/u 元音")
    print("=" * 78)
    for 名, r in 读数["元音"].items():
        峰s = " ".join("%.0f→%.1f" % (p["目标"], p["实测"]) for p in r["峰位"])
        print("[%s] 峰位 %s | 最大误差 %.2f%%" % (名, 峰s, r["最大峰位误差"] * 100))
        print("     4–8k %.2f%% | 8–16k %.2f%% | 95%%滚降 %.0fHz | 重心 %.0fHz"
              % (r["能量_4_8k"] * 100, r["能量_8_16k"] * 100,
                 r["滚降95Hz"], r["谱重心Hz"]))
    d = 读数["确定性"]
    print("确定性：两趟 sha256 同=%s（%s…） 异种子不同=%s"
          % (d["两趟sha256同"], d["sha256"], d["异种子不同"]))
    p = 读数["性能"]
    print("性能：10s 合成 %.3fs | 实时率 %.4f | <1.5s=%s"
          % (p["十秒耗时秒"], p["实时率"], p["判据_lt_1p5s"]))
    print("判据：" + " ".join("%s=%s" % (k, v) for k, v in 读数["判据"].items()))
    print("总体通过：%s" % 读数["总体通过"])
    print("wav：%s" % 读数["wav"])


if __name__ == "__main__":
    读数 = 自检()
    _打印自检(读数)
