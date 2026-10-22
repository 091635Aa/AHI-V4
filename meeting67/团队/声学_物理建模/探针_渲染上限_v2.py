# -*- coding: utf-8 -*-
"""meeting67/团队/声学_物理建模/探针_渲染上限_v2.py —— G0-v2（v0.1）复测探针。

预注册：`meeting67/07_预注册_G0v2.md`（2026-10-21T12:08，先于任何读数）。
**只修 06 R-7 划定的三类接线/工程缺陷**：
  1. 归一化：全局峰值 → **逐帧 RMS + 分位裁剪 + 失稳帧限幅**（禁止整段压死/空串）
  2. 气声/送气噪声**真正接线**：按周期度连续混合、按 SNR 定增益（不再固定 0.25）
  3. 清浊二值 → **周期度连续 + 5 ms 更新**（10 ms 帧只作 LPC 分析窗）
不允许（留 G0-v3）：源谱倾斜/开商可调、亚帧瞬态、LPC 阶/帧长、44.1k、KL 映射重做。

臂（07 §2）：P2′_主 / P1_脉冲 / A1_断气声 / A2_旧归一化 / A3_二值清浊（+P3′ 仅当 KL 修好；本版**未修好**⇒登记不跑）
片段：短 8 条×4.0 s（同 03 表）+ 长 4 条×10.0 s（确定性，见 长片段表）⇒ 合并 n=12
预算：≤70 次 ASR、≤25 min、单进程

用法：
  & "<workspace>/\03_AHI\\.venv\\Scripts\\python.exe" "…\\探针_渲染上限_v2.py" 自检   # 不载 ASR
  & "…\\探针_渲染上限_v2.py" 探针                                          # 载 ASR 跑全臂
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AHI4 = HERE.parents[2]
M66 = AHI4 / "meeting66"

SR = 22050
HOP10 = 220                 # LPC 分析帧 / 10ms
HOP5 = 110                  # 控制/合成更新 / 5ms
WIN = 551                   # 25ms 分析窗
阶 = 18
k上限 = 0.97
分位 = 99.0                 # 分位裁剪的百分位
失稳倍 = 20.0               # 帧 RMS > 失稳倍×中位 ⇒ 判失稳并限幅
增益限 = (1.0 / 8.0, 8.0)
短片段表 = ((0, 12.0), (6, 20.5), (12, 8.0), (22, 30.0),
            (30, 15.0), (43, 25.5), (56, 5.0), (72, 35.0))      # 同 03 表（行号, 起点秒）
长片段表 = ((8, 10.0), (33, 10.0), (61, 10.0), (90, 10.0))       # 4 条 × 10.0 s（确定性）
段长_短 = 4.0
段长_长 = 10.0
数据根 = Path(r"<path>\SpeakerA7代模型训练数据")
清单路 = 数据根 / "asr_opt" / "SpeakerA7代模型训练数据.list"
出目录 = HERE / "产物"
默认输出 = 出目录 / "G0v2_探针.json"

_缓存: dict = {}


def _载(名, 路):
    spec = importlib.util.spec_from_file_location(名, str(路))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def V1():
    """只读复用 v1 的自相关/列文森/量具/声码器（保证 EP 口径逐参数同）。"""
    if "v1" not in _缓存:
        _缓存["v1"] = _载("探针_渲染上限_v1_只读", HERE / "探针_渲染上限.py")
    return _缓存["v1"]


# ══════════════════════════ 数据 ══════════════════════════
def 清单() -> list:
    if "清单" not in _缓存:
        行s = 清单路.read_text(encoding="utf-8", errors="replace").splitlines()
        出 = []
        for 行 in 行s:
            if not 行.strip():
                continue
            段 = 行.split("|")
            出.append(dict(文件=段[0].split("\\")[-1].strip(), 文本=(段[3] if len(段) > 3 else "")))
        _缓存["清单"] = 出
    return _缓存["清单"]


def 读片段(行号: int, 起点秒: float, 段长秒: float) -> np.ndarray:
    import soundfile as sf  # noqa: PLC0415
    from math import gcd
    x, sr = sf.read(str(数据根 / 清单()[int(行号)]["文件"]), dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = gcd(int(sr), SR)
        from scipy.signal import resample_poly  # noqa: PLC0415
        x = resample_poly(x, SR // g, int(sr) // g)
    a = int(round(起点秒 * SR))
    n = int(round(段长秒 * SR))
    x = x[a:a + n]
    if x.size < n:
        x = np.pad(x, (0, n - x.size))
    return np.ascontiguousarray(x, dtype=np.float64)


# ══════════════════════════ oracle 控制（10ms LPC + 5ms 周期度/F0/幅度） ══════════════════════════
def 提取控制_v2(x: np.ndarray) -> dict:
    lib = V1().librosa模块()
    p = V1()
    f0, 浊, 概率 = lib.pyin(np.asarray(x, dtype=np.float32), fmin=60.0, fmax=500.0, sr=SR,
                            frame_length=1024, hop_length=HOP10, center=True,
                            fill_na=0.0, switch_prob=0.5, no_trough_prob=0.1)
    T10 = 1 + len(x) // HOP10
    f0 = np.asarray(f0, np.float64)[:T10]
    浊 = np.asarray(浊, bool)[:T10]
    概率 = np.asarray(概率, np.float64)[:T10]
    if f0.size < T10:
        f0 = np.pad(f0, (0, T10 - f0.size), mode="edge")
        浊 = np.pad(浊, (0, T10 - 浊.size), mode="edge")
        概率 = np.pad(概率, (0, T10 - 概率.size), mode="edge")
    # 周期度（连续）：浊帧取 voiced_prob，清帧取 (1-voiced_prob)*0.5 以下（连续过渡，不二值）
    周期度10 = np.where(浊, np.clip(概率, 0.0, 1.0), np.clip(概率 * 0.5, 0.0, 0.5))
    # LPC（10ms 分析窗，阶 18）——口径不变
    xp = np.pad(x, (WIN // 2, WIN), mode="constant")
    窗 = np.hamming(WIN)
    A = np.zeros((T10, 阶 + 1))
    A[:, 0] = 1.0
    for t in range(T10):
        s = t * HOP10
        帧 = xp[s: s + WIN]
        if 帧.size < WIN:
            帧 = np.pad(帧, (0, WIN - 帧.size), mode="constant")
        帧 = 帧 * 窗
        r = p.自相关(帧, 阶)
        if r[0] > 1e-12:
            a, _k = p.列文森(r, 阶)
            if np.all(np.isfinite(a)):
                A[t] = a
    # 5ms 网格（控制/合成更新）：周期度/F0/幅度线性插值
    T5 = 1 + (len(x) - 1) // HOP5
    t10 = np.arange(T10) * HOP10
    t5 = np.arange(T5) * HOP5
    周期度 = np.interp(t5, t10, 周期度10)
    f0_5 = np.interp(t5, t10, f0)
    浊_5 = 周期度 > 0.30
    幅度 = np.zeros(T5)
    for t in range(T5):
        a, b = t * HOP5, min((t + 1) * HOP5, len(x))
        幅度[t] = float(np.sqrt(np.mean(x[a:b] ** 2)) + 1e-12) if b > a else 1e-12
    return dict(f0=f0_5, 周期度=周期度, 浊=浊_5, 幅度=幅度, A10=A, T5=T5, T10=T10,
                浊10=浊, 概率10=概率, f010=f0)


def 幅度10(控制: dict) -> np.ndarray:
    """5ms 幅度 → 10ms 幅度（取每 10ms 两格的平均），供 10ms 粒度臂使用。"""
    a = 控制["幅度"]
    T = 控制["T10"]
    出 = np.zeros(T)
    for t in range(T):
        i = 2 * t
        出[t] = float(np.mean(a[i:i + 2])) if i < len(a) else float(a[-1])
    return 出


# ══════════════════════════ 激发（连续周期度混合 + SNR 定噪声） ══════════════════════════
def 声门脉冲串(控制: dict, n: int, 幅度网格: np.ndarray, 周期网格: np.ndarray) -> np.ndarray:
    """Rosenberg 声门流导数脉冲，幅度按 p^0.5 加权（p=周期度）。"""
    源 = np.zeros(n)
    相位 = 0.0
    网格 = HOP5
    for i in range(n):
        t = i // 网格
        if t >= len(幅度网格):
            t = len(幅度网格) - 1
        f0 = float(控制["f0"][min(t * (网格 // HOP5) if 网格 == HOP5 else t, len(控制["f0"]) - 1)])
        p = float(np.clip(周期网格[t], 0.0, 1.0))
        if p <= 0.02 or f0 <= 1.0 or not np.isfinite(f0):
            continue
        相位 += f0 / SR
        if 相位 >= 1.0:
            相位 -= 1.0
            L = max(int(round(0.6 * SR / f0)), 8)
            L开 = max(int(round(0.6 * L)), 3)
            L闭 = max(L - L开, 2)
            形 = np.zeros(L)
            k = np.arange(L开)
            形[:L开] = 0.5 * (np.pi / L开) * np.sin(np.pi * k / L开)
            k2 = np.arange(L闭)
            形[L开:] = -(np.pi / (2.0 * L闭)) * np.sin(np.pi * k2 / (2.0 * L闭))
            峰 = float(np.abs(形).max())
            if 峰 > 1e-9:
                形 /= 峰
            形 *= float(幅度网格[t]) * np.sqrt(p)
            末 = min(i + L, n)
            源[i:末] += 形[: 末 - i]
    return 源


def 脉冲串(控制: dict, n: int, 幅度网格: np.ndarray, 周期网格: np.ndarray) -> np.ndarray:
    """单位冲激（P1 臂），幅度按 p^0.5。"""
    源 = np.zeros(n)
    相位 = 0.0
    for i in range(n):
        t = i // HOP5
        if t >= len(幅度网格):
            break
        p = float(np.clip(周期网格[t], 0.0, 1.0))
        f0 = float(控制["f0"][min(t, len(控制["f0"]) - 1)])
        if p <= 0.02 or f0 <= 1.0:
            continue
        相位 += f0 / SR
        if 相位 >= 1.0:
            相位 -= 1.0
            源[i] += float(幅度网格[t]) * np.sqrt(p)
    return 源


def 噪声激发(控制: dict, n: int, 幅度网格: np.ndarray, 周期网格: np.ndarray,
             种子: int, 断气声: bool = False) -> np.ndarray:
    """气声/送气噪声：按 SNR 定增益（SNR = 10log10(p/(1-p))，截 [-6,30] dB）。"""
    if 断气声:
        return np.zeros(n)
    rng = np.random.default_rng(int(种子) & 0x7FFFFFFF)
    噪 = rng.standard_normal(n)
    出 = np.zeros(n)
    for t in range(min(len(幅度网格), len(周期网格))):
        a, b = t * HOP5, min((t + 1) * HOP5, n)
        if b <= a:
            break
        p = float(np.clip(周期网格[t], 1e-4, 1.0 - 1e-4))
        snr = float(np.clip(10.0 * np.log10((p + 1e-4) / (1.0 - p + 1e-4)), -6.0, 30.0))
        幅度 = float(幅度网格[t]) * np.sqrt(max(1.0 - p, 0.0)) * 10.0 ** (-snr / 20.0)
        出[a:b] = 噪[a:b] * (幅度 * 3.0)
    return 出


# ══════════════════════════ 声道（LPC 全极点，10ms 系数 / 5ms 拼接） ══════════════════════════
def 合成_全极点(控制: dict, 激: np.ndarray, 网格: int = HOP5) -> tuple[np.ndarray, int]:
    """逐帧 LPC 合成；`网格`=HOP5 时每 5ms 更新（A 取所在 10ms 帧），状态继承。
    返回 (波形, 失稳帧数)。"""
    ss = V1().scipy_signal()
    y = np.zeros_like(激)
    zi = np.zeros(阶)
    上 = np.zeros(阶 + 1)
    上[0] = 1.0
    T = 1 + (len(激) - 1) // 网格
    原始RMS = np.zeros(T)
    for t in range(T):
        a, b = t * 网格, min((t + 1) * 网格, len(激))
        if b <= a:
            break
        idx = min((t * 网格) // HOP10, 控制["T10"] - 1)
        A = 控制["A10"][idx]
        if not np.all(np.isfinite(A)) or abs(A[0]) < 1e-12:
            A = 上
        上 = A
        y[a:b], zi = ss.lfilter([1.0], A, 激[a:b], zi=zi)
        原始RMS[t] = float(np.sqrt(np.mean(y[a:b] ** 2)) + 1e-15)
    # —— 失稳帧限幅：帧 RMS > 失稳倍×中位 ⇒ 该帧清零并重置滤波状态 ——
    失稳 = 0
    基准 = float(np.percentile(原始RMS, 95)) + 1e-15      # 用 p95 而非中位（中位会被停顿帧压低）
    # 需要重跑一遍才能"清零+重置"⇒ 用两遍法：先定位，再合成
    坏 = np.where(原始RMS > 失稳倍 * 基准)[0]
    if 坏.size:
        失稳 = int(坏.size)
        y = np.zeros_like(激)
        zi = np.zeros(阶)
        上 = np.zeros(阶 + 1)
        上[0] = 1.0
        for t in range(T):
            a, b = t * 网格, min((t + 1) * 网格, len(激))
            if b <= a:
                break
            if t in set(坏.tolist()):
                zi = np.zeros(阶)          # 重置状态
                continue                    # 该帧留 0（限幅）
            idx = min((t * 网格) // HOP10, 控制["T10"] - 1)
            A = 控制["A10"][idx]
            if np.all(np.isfinite(A)) and abs(A[0]) > 1e-12:
                上 = A
            y[a:b], zi = ss.lfilter([1.0], 上, 激[a:b], zi=zi)
    return y, 失稳


# ══════════════════════════ 归一化（逐帧 RMS + 分位裁剪） ══════════════════════════
def 逐帧对幅(y: np.ndarray, 目标: np.ndarray, 网格: int = HOP5) -> np.ndarray:
    出 = y.copy()
    T = min(len(目标), 1 + (len(y) - 1) // 网格)
    g = np.ones(T)
    for t in range(T):
        a, b = t * 网格, min((t + 1) * 网格, len(y))
        if b <= a:
            break
        r = float(np.sqrt(np.mean(y[a:b] ** 2)) + 1e-15)
        g[t] = float(np.clip(目标[t] / r, 增益限[0], 增益限[1]))
    gs = np.convolve(g, np.ones(3) / 3.0, mode="same")      # 3 格平滑（降增益抖动，确定性）
    for t in range(T):
        a, b = t * 网格, min((t + 1) * 网格, len(出))
        if b > a:
            出[a:b] *= gs[t]
    return 出


def 归一_分位(y: np.ndarray, 目标=0.9) -> np.ndarray:
    s = float(np.percentile(np.abs(y), 分位)) if y.size else 0.0
    if s < 1e-9:
        s = float(np.abs(y).max()) + 1e-12
    return np.clip(y / s * 目标, -0.90, 0.90)


def 归一_峰值(y: np.ndarray, 目标=0.9) -> np.ndarray:   # A2 臂：旧行为
    m = float(np.abs(y).max())
    return (y / m * 目标) if m > 1e-9 else y


# ══════════════════════════ 臂合成 ══════════════════════════
def 合成_臂(控制: dict, 臂: str, n样本: int, 种子: int) -> tuple[np.ndarray, dict]:
    网格 = HOP5
    断气声 = False
    旧归一 = False
    二值 = False
    if 臂 == "P2′_主":
        pass
    elif 臂 == "P1_脉冲":
        pass
    elif 臂 == "A1_断气声":
        断气声 = True
    elif 臂 == "A2_旧归一化":
        旧归一 = True
    elif 臂 == "A3_二值清浊":
        二值 = True
        网格 = HOP10
    else:
        raise ValueError(臂)
    if 二值:
        # A3 臂：清浊二值 + 10ms（只用 03 口径的粒度；激励仍在样本级渲染，但控制量 10ms 常值）
        a10 = 幅度10(控制)
        幅度网格 = np.repeat(a10, 2)[: max(控制["T5"], 1)]
        周期网格 = np.repeat(np.where(控制["浊10"], 1.0, 0.0), 2)[: max(控制["T5"], 1)]
        控 = 控制
    else:
        幅度网格 = 控制["幅度"]
        周期网格 = 控制["周期度"]
        控 = 控制
    if 臂 == "P1_脉冲":
        激源 = 脉冲串(控, n样本, 幅度网格, 周期网格)
    else:
        激源 = 声门脉冲串(控, n样本, 幅度网格, 周期网格)
    激 = 激源 + 噪声激发(控, n样本, 幅度网格, 周期网格, 种子, 断气声=断气声)
    y, 失稳 = 合成_全极点(控制, 激, 网格=网格)
    元 = dict(失稳帧=失稳)
    if 旧归一:
        y = 归一_峰值(逐帧对幅(y, 控制["幅度"], 网格=网格))
    else:
        y = 归一_分位(逐帧对幅(y, 控制["幅度"], 网格=网格))
    return y, 元


# ══════════════════════════ 自检（不载 ASR） ══════════════════════════
def _谱帧(y: np.ndarray) -> np.ndarray:
    w = np.hanning(WIN)
    hop = HOP10
    T = 1 + (len(y) - 1) // hop
    yp = np.pad(y, (WIN // 2, WIN))
    帧 = np.stack([yp[t * hop: t * hop + WIN] * w for t in range(T)])
    return np.abs(np.fft.rfft(帧, 1024)) ** 2


def _清帧掩码(S: np.ndarray, 高=(2000, 8000), 低=(100, 2000)) -> np.ndarray:
    """按谱平坦度前 25% 判清帧（只在 GL 上算一次，跨臂复用 ⇒ 可比）。"""
    频 = np.fft.rfftfreq(1024, 1.0 / SR)
    m = ((频 >= 高[0]) & (频 <= 高[1])) | ((频 >= 低[0]) & (频 < 低[1]))
    平 = np.exp(np.mean(np.log(np.maximum(S[:, m], 1e-20)), axis=1)) / \
        np.maximum(np.mean(S[:, m], axis=1), 1e-20)
    return 平 >= np.percentile(平, 75)


def _带能比(y: np.ndarray, 清: np.ndarray, 高=(2000, 8000), 低=(100, 2000)):
    S = _谱帧(y)
    T = min(len(S), len(清))
    清 = 清[:T]
    if 清.sum() < 5:
        return None
    频 = np.fft.rfftfreq(1024, 1.0 / SR)
    mh = (频 >= 高[0]) & (频 <= 高[1])
    ml = (频 >= 低[0]) & (频 < 低[1])
    高比 = 10 * np.log10(np.maximum(S[:T][清][:, mh].sum(1), 1e-20) /
                        np.maximum(S[:T][清][:, ml].sum(1), 1e-20))
    return float(np.mean(高比))


def 自检(条数: int = 2) -> dict:
    os.makedirs(str(出目录), exist_ok=True)
    V = V1()
    S = V.声码()
    报 = dict(口径=dict(SR=SR, HOP5=HOP5, HOP10=HOP10, 阶=阶, 分位=分位, 失稳倍=失稳倍),
              逐条=[])
    for 行号, 起点 in 短片段表[:条数]:
        x = 读片段(行号, 起点, 段长_短)
        控 = 提取控制_v2(x)
        n = len(x)
        gl = S.mel转波(V.计算logmel(x))
        条 = dict(行号=行号, 起点秒=起点, 周期度中位=round(float(np.median(控["周期度"])), 3),
                 周期度_std=round(float(np.std(控["周期度"])), 3))
        Sgl = _谱帧(gl)
        清掩 = _清帧掩码(Sgl)
        条["清帧占比"] = round(float(清掩.mean()), 3)
        gl高 = _带能比(gl, 清掩)
        条["GL清帧高频比dB"] = round(gl高, 2) if gl高 is not None else None
        for 臂 in ("P2′_主", "P1_脉冲", "A1_断气声", "A3_二值清浊"):
            y, 元 = 合成_臂(控, 臂, n, 1000 + 行号)
            y2, _ = 合成_臂(控, 臂, n, 1000 + 行号)
            高 = _带能比(y, 清掩)
            条[臂] = dict(有限=bool(np.all(np.isfinite(y))), 峰值=round(float(np.abs(y).max()), 3),
                         逐位同=bool(hashlib.sha256(y.tobytes()).hexdigest() ==
                                    hashlib.sha256(y2.tobytes()).hexdigest()),
                         失稳帧=元["失稳帧"],
                         清帧高频比dB=round(高, 2) if 高 is not None else None,
                         帧RMS动态范围dB=round(float(20 * np.log10(
                             (np.percentile(np.abs(y), 99) + 1e-12) /
                             (np.percentile(np.abs(y), 20) + 1e-12))), 1))
        print("[自检] 行%d GL清帧高频比=%.2f dB ｜ P2′=%.2f ｜ P1=%.2f ｜ A1断气声=%.2f ｜ A3二值=%.2f"
              % (行号, 条["GL清帧高频比dB"], 条["P2′_主"]["清帧高频比dB"],
                 条["P1_脉冲"]["清帧高频比dB"], 条["A1_断气声"]["清帧高频比dB"],
                 条["A3_二值清浊"]["清帧高频比dB"]), flush=True)
        print("        确定性: P2′逐位同=%s A3逐位同=%s ｜ 失稳帧 P2′=%.0f A3=%.0f ｜ 峰值 %.3f"
              % (条["P2′_主"]["逐位同"], 条["A3_二值清浊"]["逐位同"],
                 条["P2′_主"]["失稳帧"], 条["A3_二值清浊"]["失稳帧"], 条["P2′_主"]["峰值"]), flush=True)
        报["逐条"].append(条)
    (出目录 / "G0v2_自检.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[自检落盘] %s" % (出目录 / "G0v2_自检.json"))
    return 报


# ══════════════════════════ 探针主流程（载 ASR） ══════════════════════════
臂_短 = ("P2′_主", "P1_脉冲", "A1_断气声", "A2_旧归一化", "A3_二值清浊")


def 探针(输出: Path | None = None, 只短: bool = False) -> dict:
    输出 = 输出 or 默认输出
    os.makedirs(str(出目录), exist_ok=True)
    V = V1()
    量 = V.量具()
    声 = V.声码()
    引 = 量.载引擎()
    print("[ASR] %s" % 引["名"], flush=True)
    转写 = 引["转写"]
    开始 = time.time()
    n调用 = 0
    逐条 = []

    def 一条(行号, 起点, 段长, 集合, 臂s):
        nonlocal n调用
        x = 读片段(行号, 起点, 段长)
        n = len(x)
        控 = 提取控制_v2(x)
        名 = "%s_%d_%s" % (集合, 行号, str(起点).replace(".", "p"))
        路真 = 出目录 / ("G0v2_真_%s.wav" % 名)
        量.写wav(str(路真), x, fs=SR)
        T_h = 转写(str(路真)); n调用 += 1
        条 = dict(集合=集合, 行号=行号, 清单文件=清单()[行号]["文件"], 起点秒=起点,
                  段长秒=段长, 时长秒=round(n / SR, 2), T_h=T_h,
                  周期度中位=round(float(np.median(控["周期度"])), 3), 条件={})
        # GL 对照（护栏 G-1）
        gl = 声.mel转波(V.计算logmel(x))
        路GL = 出目录 / ("G0v2_GL_%s.wav" % 名)
        量.写wav(str(路GL), gl, fs=SR)
        Tgl = 转写(str(路GL)); n调用 += 1
        条["条件"]["真mel_GL"] = dict(CER=量.CER(T_h, Tgl), 转写=Tgl)
        空串 = []
        if not Tgl.strip():
            空串.append("真mel_GL")
        for 臂 in 臂s:
            t0 = time.perf_counter()
            y, 元 = 合成_臂(控, 臂, n, 3000 + 行号)
            合成秒 = time.perf_counter() - t0
            路 = 出目录 / ("G0v2_%s_%s.wav" % (臂.split("_")[0].replace("′", "p"), 名))
            量.写wav(str(路), y, fs=SR)
            Ts = 转写(str(路)); n调用 += 1
            if not Ts.strip():
                空串.append(臂)
            条["条件"][臂] = dict(CER=量.CER(T_h, Ts), 转写=Ts, 合成秒=round(合成秒, 2),
                                实时率=round(合成秒 / (n / SR), 3), 失稳帧=元["失稳帧"],
                                峰值=round(float(np.abs(y).max()), 3),
                                有限=bool(np.all(np.isfinite(y))))
        条["空串"] = 空串
        print("[%s] 行%d %s %.1fs ｜ GL=%.4f ｜ " %
              (集合, 行号, 清单()[行号]["文件"], 条["时长秒"], 条["条件"]["真mel_GL"]["CER"])
              + " ".join("%s=%.4f" % (a.split("_")[0], 条["条件"][a]["CER"]) for a in 臂s)
              + (" ｜ 空串:%s" % 空串 if 空串 else ""), flush=True)
        return 条

    for 行号, 起点 in 短片段表:
        逐条.append(一条(行号, 起点, 段长_短, "短", 臂_短))
    if not 只短:
        for 行号, 起点 in 长片段表:
            逐条.append(一条(行号, 起点, 段长_长, "长", ("P2′_主",)))

    def 汇(子集, 臂):
        vs = [t["条件"][臂]["CER"] for t in 子集 if 臂 in t["条件"]]
        if not vs:
            return None
        return dict(CER均值=round(float(np.mean(vs)), 4), CER中位=round(float(np.median(vs)), 4),
                    CER范围=[round(float(min(vs)), 4), round(float(max(vs)), 4)], n=len(vs))
    短 = [t for t in 逐条 if t["集合"] == "短"]
    长 = [t for t in 逐条 if t["集合"] == "长"]
    汇s = {"短": {}, "长": {}, "合并n=%d" % len(逐条): {}}
    for 臂 in 臂_短:
        汇s["短"][臂] = 汇(短, 臂)
        汇s["合并n=%d" % len(逐条)][臂] = 汇(逐条, 臂)
    for 臂 in ("真mel_GL", "P2′_主"):
        汇s["长"][臂] = 汇(长, 臂)
        汇s["合并n=%d" % len(逐条)][臂] = 汇(逐条, 臂)
    # 护栏与判定
    GL均值 = 汇(逐条, "真mel_GL")["CER均值"]
    P2均值 = 汇(逐条, "P2′_主")["CER均值"]
    空串数 = int(sum(1 for t in 逐条 if t["空串"]))
    判定 = ("PASS" if P2均值 <= 0.20 else ("MARGINAL" if P2均值 <= 0.35 else "FAIL"))
    消融 = {}
    for 臂 in ("A1_断气声", "A2_旧归一化", "A3_二值清浊"):
        v = 汇(短, 臂)
        if v and 汇(短, "P2′_主"):
            消融[臂] = dict(CER均值=v["CER均值"],
                            差vs_P2主=round(v["CER均值"] - 汇(短, "P2′_主")["CER均值"], 4),
                            贡献=bool(v["CER均值"] - 汇(短, "P2′_主")["CER均值"] >= 0.05))
    报 = dict(
        探针="meeting67 G0-v2（v0.1）器官渲染上限复测",
        预注册="meeting67/07_预注册_G0v2.md（2026-10-21T12:08）",
        口径=dict(SR=SR, 网格_控制="5ms（A3 臂 10ms）", LPC分析="10ms/25ms/hamming/阶18",
                 片段=f"短 {len(短)} 条×{段长_短}s + 长 {len(长)} 条×{段长_长}s",
                 短片段表=[dict(行号=r, 起点秒=s) for r, s in 短片段表],
                 长片段表=[dict(行号=r, 起点秒=s) for r, s in 长片段表],
                 参考文本="同段真 wav 的 ASR 转写 T_h", ASR=引["名"],
                 修复范围="07 §0 三类（归一化/气声接线/连续清浊5ms）",
                 时间戳=time.strftime("%Y-%m-%d %H:%M:%S")),
        护栏=dict(G1_GL均值=GL均值, G1_通过=bool(GL均值 <= 0.15),
                 G4_空串数=空串数,
                 G4_空串条目数=int(sum(1 for t in 逐条 if t["空串"])),
                 G4_通过=bool(空串数 == 0),
                 G3_全部有限=bool(all(t["条件"][a]["有限"] for t in 逐条
                                    for a in t["条件"] if a != "真mel_GL")),
                 G6_SR=SR),
        逐条=逐条, 汇总=汇s, 消融=消融,
        判定=dict(主臂="P2′_主", 合并均值=P2均值, 判定=判定, 阈值="≤0.20 PASS / ≤0.35 MARGINAL / >0.35 FAIL"),
        ASR调用次数=n调用, 耗时秒=round(time.time() - 开始, 1))
    _写json(输出, 报)
    print("\n=== 汇总 ===")
    for 集, d in 汇s.items():
        print("[%s]" % 集)
        for 臂, v in d.items():
            if v:
                print("   %-14s 均值=%.4f 中位=%.4f 范围=%s n=%d"
                      % (臂, v["CER均值"], v["CER中位"], v["CER范围"], v["n"]))
    print("护栏 G-1 GL=%.4f(%s) ｜ G-4 空串=%d(%s)"
          % (GL均值, "过" if GL均值 <= 0.15 else "不过", 空串数, "过" if 空串数 == 0 else "不过"))
    print("消融:", json.dumps(消融, ensure_ascii=False))
    print("判定: P2′合并均值=%.4f ⇒ %s" % (P2均值, 判定))
    print("[落盘] %s（ASR %d 次，%.1fs）" % (输出, n调用, 报["耗时秒"]))
    return 报


def _写json(路: Path, o):
    def 转(v):
        if isinstance(v, dict):
            return {str(k): 转(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [转(x) for x in v]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return 转(v.tolist())
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    路.write_text(json.dumps(转(o), ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    派 = argparse.ArgumentParser()
    派.add_argument("模式", nargs="?", default="探针", choices=("自检", "探针"))
    派.add_argument("--条数", type=int, default=2)
    派.add_argument("--只短", action="store_true")
    派.add_argument("--输出", default=None)
    a = 派.parse_args()
    if a.模式 == "自检":
        自检(a.条数)
    else:
        探针(Path(a.输出) if a.输出 else None, 只短=a.只短)
    return 0


if __name__ == "__main__":
    sys.exit(main())
