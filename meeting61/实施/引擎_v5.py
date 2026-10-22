# -*- coding: utf-8 -*-
"""meeting61/实施/引擎_v5.py —— 44.1kHz 源-滤波器声带引擎（v5）。

相对 v4（引擎_v4.py）的**底层**修改（不是 EQ/音量补丁）：
  1) 噪声不再双重高通：v4 的噪声源先用 `diff(白噪)`（+6dB/oct），输出端又做唇辐射
     `diff`（再 +6dB/oct），净 +12dB/oct ⇒ 擦音/送气极薄、谱重心虚高（EmoCompanion 1847Hz
     vs 参考 ~974Hz）。v5 噪声源直接用白噪，只在输出端做一次唇辐射。
  2) 送气噪声按**后续元音共振峰**成形（噪形=1）：塞音的 VOT/喉擦 h 用 F1–F3 并联
     谐振器塑形，而不是固定窄带 ⇒ 爆破部位与 CV 过渡有可听线索（这是可懂度的关键）。
  3) 滤波状态跨段连续：v4 在每次「声源类型」变化时清空全部谐振器状态，辅音→元音
     的共振峰过渡被截断；v5 只在整句开头清零，静默段用零输入让状态自然衰减。
  4) 取消逐段 2ms 淡入淡出（v4 在每个类型边界都挖一个 2ms 的包络凹口，制造断音）；
     v5 只在整句首尾做 8ms 淡入淡出。
  5) 幅度包络作用在声门源上（逐样本插值），噪声帧按块做 RMS 标定 ⇒ 清/浊电平可控。
  6) 浊音段不再整体 4.2kHz 低通（v4 为压嘶声而加，连带把 F4/F5 与自然高频压掉）。

接口与 v4 相同：``合成(帧: dict, 音色: dict = None, 种子: int = 300) -> (波形, 元信息)``。
帧键在 v4 基础上多一个可选 ``噪形``（0=频带噪声，1=共振峰成形噪声）。
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from scipy.signal import lfilter, butter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import 引擎_v4 as E4  # noqa: E402  复用声门脉冲表/谐振器/音色规整

FS = E4.FS
帧率 = E4.帧率
每帧样本 = E4.每帧样本
产物临时 = E4.产物临时

# —— v5 标定（实测见 自检 与 判据_发音_v5.py）——
_源标定 = 0.55              # 声门源标定：幅=1 的稳态元音 RMS≈0.12（v4 为 0.22 且峰值归一）
_气声源权重 = 0.012
_气声直通权重 = 0.010
_噪标定 = 0.020             # 噪声帧：幅=1 时 RMS≈0.021 ⇒ 强擦音相对元音约 −12dB（可听且不吵）
_默认共振峰增益 = (1.0, 0.55, 0.20, 0.09, 0.04)   # v5：压低 F3–F5（v4 的 0.30/0.16/0.08 使 2–4k 过亮）
_声源低通Hz = 6000.0
_声源低通阶 = 2
_微颤Hz = 4.7
_微颤深 = 0.012
_并联下限 = 400.0           # 共振峰成形噪声只用 F1–F3，且不低于此值

_表栈 = E4._表栈
_表层数 = E4.表层数
_表长 = E4.表长


def 声门流表(n: int = 1024, 紧喉: float = 0.0) -> np.ndarray:
    """单周期声门**流**（Rosenberg），峰值归一。

    v4 用声门流的导数当声源，叠加唇辐射后净频谱近似平直（2–4k 过亮）；
    真人声门流谱约 −12dB/oct，唇辐射 +6dB/oct ⇒ 净 −6dB/oct。
    """
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
    # 去直流：声门流在一个周期内有非零均值；保留它会形成准直流平台，
    # 在「浊音→清辅音」边界被唇辐射差分打成单采样咔哒（实测 −0.86）。
    g = g - float(np.mean(g))
    gmax = float(np.max(np.abs(g))) or 1.0
    return (g / gmax).astype(np.float64)


_流栈 = np.stack([声门流表(_表长, float(q)) for q in E4._表层])
_谐振器系数 = E4.谐振器系数
_滤波块 = E4._滤波块
_帧数组 = E4._帧数组
_平滑 = E4._平滑


def _辐射增益(x: np.ndarray) -> float:
    """一阶差分 y[n]=x[n]-x[n-1] 对该块信号的 RMS 增益（解析估计）。"""
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
    # v5：辅音噪声先用 7.5kHz 低通（否则唇辐射 +6dB/oct 把白噪推成蓝噪声，
    # 擦音谱重心虚高到 8–14kHz，参考只有 0.6–1kHz）
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

    # —— 声门源 ——
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
    噪混 = 气带 * (气声样 * _气声源权重 + 送气样 * _气声源权重)
    级联入 = (源滤 + 噪混) * _源标定 * 幅样     # 幅度包络作用在声门源上
    g_气直通 = (气声样 * _气声直通权重 + 送气样 * _气声直通权重) * 音["噪声增益"]

    预 = np.zeros(总样本, dtype=np.float64)
    噪预 = np.zeros(总样本, dtype=np.float64)
    噪增益帧 = np.ones(N, dtype=np.float64)
    级联zi = [np.zeros(2) for _ in range(5)]
    带zi1 = np.zeros(2)
    带zi2 = np.zeros(2)
    形zi = [np.zeros(2) for _ in range(3)]
    增 = 音["共振峰增益"]

    for fi in range(N):
        a, b = fi * 每帧样本, min((fi + 1) * 每帧样本, 总样本)
        if a >= b:
            break
        typ = int(源帧[fi])
        if typ == 1:
            x = 级联入[a:b]
            ysum = np.zeros_like(x)
            for j in range(5):
                A, B, C = _谐振器系数(F帧[j, fi], BW帧[j, fi])
                yj, 级联zi[j] = _滤波块(A, B, C, x, 级联zi[j])
                ysum += 增[j] * yj
            预[a:b] = ysum + g_气直通[a:b] * 气带[a:b]
        elif typ == 0:
            blk = 帧带[a:b]
            # 每段噪声起点重置噪声滤波器状态；段内保持连续，避免逐帧重启的瞬态
            if fi == 0 or int(源帧[fi - 1]) != 0:
                带zi1 = np.zeros(2)
                带zi2 = np.zeros(2)
                形zi = [np.zeros(2) for _ in range(3)]
            if int(噪形帧[fi]) == 1:
                # 送气/喉擦：按后续元音 F1–F3 并联成形（状态跨帧连续）
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
                x = y + 0.22 * blk          # 宽带底噪，避免纯窄带“哨音”
            # 先存未标定的噪声；逐帧增益在循环后平滑，再统一乘上（消除帧边界台阶）
            噪预[a:b] = x
            增益 = _辐射增益(x)
            目标 = float(幅帧[fi]) * _噪标定 * 音["噪声增益"] * (0.85 + 0.30 * 送帧[fi])
            rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
            噪增益帧[fi] = (目标 / (rms * 增益)) if (rms > 1e-9 and 目标 > 0.0) else 0.0
            # 清辅音段让浊音声道滤波器状态自然衰减，避免进入元音时旧状态冲出瞬态
            z = np.zeros(b - a, dtype=np.float64)
            for j in range(5):
                A, B, C = _谐振器系数(F帧[j, fi], BW帧[j, fi])
                _, 级联zi[j] = _滤波块(A, B, C, z, 级联zi[j])
        else:
            预[a:b] = 0.0
            # 非浊音段用零输入让声道状态自然衰减（v4 是直接清空 ⇒ 断音）
            z = np.zeros(b - a, dtype=np.float64)
            for j in range(5):
                A, B, C = _谐振器系数(F帧[j, fi], BW帧[j, fi])
                _, 级联zi[j] = _滤波块(A, B, C, z, 级联zi[j])

    # 噪声逐帧增益 3 帧滑动平滑（避免 10ms 帧边界上的台阶/咔哒）
    平滑噪增益 = 噪增益帧.copy()
    for fi in range(N):
        if int(源帧[fi]) != 0:
            continue
        lo, hi = max(0, fi - 1), min(N, fi + 2)
        邻 = [噪增益帧[k] for k in range(lo, hi) if int(源帧[k]) == 0]
        if 邻:
            平滑噪增益[fi] = float(sum(邻)) / len(邻)
    # 非噪声帧的增益取「最近噪声帧」的增益（最近邻填充）。
    # 若默认 1，插值会在噪声段首/尾把增益拉回 1，导致最后一帧放大数百倍（实测 0.95 峰值）。
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
    # —— 浊/清边界 2ms 交叉包络 ——
    # 声道谐振器在段末不必为 0；若直接切到静默/噪声，唇辐射差分会把「台阶」
    # 变成单采样咔哒（实测 maxdiff≈1.07×峰值）。这里只在含浊音的边界做 2ms 淡入淡出。
    段包络 = np.ones(总样本, dtype=np.float64)
    渐变量 = min(88, 每帧样本 // 4)               # ≈2ms
    变点 = np.where(np.diff(源帧) != 0)[0] + 1
    for bf in 变点:
        bf = int(bf)
        if not (int(源帧[bf]) == 1 or int(源帧[bf - 1]) == 1):
            continue
        b = bf * 每帧样本                      # 帧索引 → 样本索引（v5 修正：原先把帧号当样本号）
        a0 = max(0, b - 渐变量)
        if b > 0:
            段包络[a0:b] *= np.linspace(1.0, 0.0, b - a0)
        a1 = min(总样本, b + 渐变量)
        段包络[b:a1] = np.minimum(段包络[b:a1], np.linspace(0.0, 1.0, a1 - b))
    预 *= 段包络

    # —— 唇辐射：只做一次（v5 修复 v4 的双重高通）——
    辐 = np.empty_like(预)
    if 总样本:
        辐[0] = 预[0]
        if 总样本 > 1:
            辐[1:] = 预[1:] - 预[:-1]
    # 整句首尾淡入淡出（8ms），段边界不再挖包络
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
        合成耗时秒=round(耗时, 4), 实时率=round(耗时 / max(总样本 / FS, 1e-9), 4),
        f0样=f0样)
    return out, 元信息


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    n = 40
    帧 = dict(f0=[180.0] * n, f1=[750.0] * n, f2=[1250.0] * n, f3=[2600.0] * n,
             f4=[3500.0] * n, f5=[4500.0] * n, 幅=[0.8] * n, 气声=[0.02] * n,
             声源=[1] * n, 噪带=[None] * n, 送气=[0.0] * n, 紧喉=[0.3] * n,
             粗糙=[0.03] * n, 噪形=[0] * n)
    x, 元 = 合成(帧)
    print("元音 a：时长 %.2fs 峰值 %.3f RMS %.4f" %
          (元["时长秒"], 元["峰值"], float(np.sqrt(np.mean(x * x)))))
    帧s = dict(f0=[180.0] * 20, f1=[300.0] * 20, f2=[1800.0] * 20, f3=[2600.0] * 20,
              f4=[3500.0] * 20, f5=[4500.0] * 20, 幅=[0.8] * 20, 气声=[0.0] * 20,
              声源=[0] * 20, 噪带=[(6800.0, 2200.0)] * 20, 送气=[1.0] * 20,
              紧喉=[0.3] * 20, 粗糙=[0.0] * 20, 噪形=[0] * 20)
    xs, _ = 合成(帧s)
    print("擦音 s：峰值 %.3f RMS %.4f（相对元音 %+.1f dB）" %
          (float(np.max(np.abs(xs))), float(np.sqrt(np.mean(xs * xs))),
           20 * np.log10(max(np.sqrt(np.mean(xs * xs)), 1e-9)
                         / max(np.sqrt(np.mean(x * x)), 1e-9))))
