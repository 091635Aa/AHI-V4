# -*- coding: utf-8 -*-
"""meeting70/实施/混合_H1.py —— H1：v13 谱包络 × 器官激励（零训练，只读 v13）。

mel(v13) → 线性谱包络；LF 声门源 + 噪声（F0 由文本声调模板给）→ STFT 逐帧乘包络 → ISTFT。
"""
from __future__ import annotations

import importlib.util
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M70 = HERE.parent
M69 = M70.parent / "meeting69"
M66 = M70.parent / "meeting66"
出 = M70 / "产物"
SR = 22050
帧样本 = 220          # 10 ms（mel 帧）


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


器 = 载("器官v2", HERE / "器官v2.py")
精 = 载("音频线_v1_精修", M69 / "实施" / "音频线_v1_精修.py")
声 = 精.声
Lib = 声.librosa模块()

声母表 = ["∅", "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s"]
韵母表 = ["a", "o", "e", "i", "u", "v", "er", "ii", "iii", "ai", "ei", "ao", "ou", "ia", "ie", "iao", "iou", "ua", "uo",
        "uai", "uei", "ve", "an", "en", "in", "uen", "vn", "ang", "eng", "ing", "ong", "iong", "ian", "uan", "uang",
        "van", "", "iang", "io"]
# 声调模板（半音，相对基频）
调型 = {0: np.array([0.0, 0.4, 0.7, 0.9]), 1: np.array([-0.6, -0.2, 0.5, 1.0]),
        2: np.array([0.2, -0.6, -0.8, 0.6]), 3: np.array([1.0, 0.4, -0.4, -0.9]),
        4: np.array([0.0, 0.0, 0.0, 0.0])}


def 包络(mel):
    功率 = np.maximum(np.exp(np.clip(np.asarray(mel, np.float64), -40, 40)) - 1e-6, 0.0)
    M = Lib.feature.inverse.mel_to_stft(功率.T, sr=SR, n_fft=1024, power=2.0, fmin=0.0, fmax=SR / 2.0)
    return np.maximum(np.asarray(M, np.float64), 0.0)          # (513, T)


def 文本F0(文本, T, 基F0=320.0):
    from pypinyin import lazy_pinyin, Style
    音节s = [s for s in lazy_pinyin(文本, style=Style.TONE3, neutral_tone_with_five=True)]
    n = max(1, len(音节s))
    每 = T / n
    f0 = np.zeros(T)
    for i, s in enumerate(音节s):
        tone = 5
        if s and s[-1].isdigit():
            tone = int(s[-1]); s = s[:-1]
        tid = max(0, min(4, tone - 1))
        a = int(round(i * 每)); b = int(round((i + 1) * 每))
        b = min(b, T)
        if b <= a:
            continue
        xs = np.linspace(0, 1, len(调型[tid]))
        tpl = np.interp(np.linspace(0, 1, b - a), xs, 调型[tid])
        f0[a:b] = 基F0 * (2.0 ** (tpl / 12.0))
    # 全句轻微下降
    f0 *= np.linspace(1.0, 2.0 ** (-1.0 / 12.0), T)
    return f0


def 激励(F0帧, 噪声比帧, 种子=20261022):
    T = len(F0帧); N = T * 帧样本
    f0s = np.interp(np.arange(N) / 帧样本, np.arange(T) + 0.5, F0帧)
    ph = np.mod(np.cumsum(f0s) / SR, 1.0)
    流, 导 = 器.声门表(0.62, 0.42)
    源 = np.interp(ph * 器._表长, np.arange(器._表长), 导)
    rng = np.random.RandomState(种子)
    噪 = rng.randn(N)
    噪hp = np.append(噪[0], 噪[1:] - 噪[:-1])
    NRs = np.interp(np.arange(N) / 帧样本, np.arange(T) + 0.5, 噪声比帧)
    源 = 源 * 0.6 + 噪hp * NRs * 0.8
    return 源


def 合成(文本, 保存=None, 基F0=320.0, 种子=20261022, 迭代=32, mel覆盖=None):
    mel, 有效, 跳 = 精.推理mel(文本, 24)
    if mel覆盖 is not None:
        mel = np.asarray(mel覆盖, np.float32)
    M = 包络(mel)                                    # (513, T)
    T = M.shape[1]
    # 噪声比：mel 谱平坦度（逐帧）
    功率 = np.maximum(np.exp(np.clip(np.asarray(mel, np.float64), -40, 40)) - 1e-6, 0.0)
    flat = np.exp(np.mean(np.log(功率 + 1e-10), axis=1)) / (np.mean(功率, axis=1) + 1e-10)
    NR = np.clip((flat - 0.10) / 0.60, 0.0, 1.0)
    F0 = 文本F0(文本, T, 基F0)
    源 = 激励(F0, NR, 种子)
    S = Lib.stft(源, n_fft=1024, hop_length=帧样本, win_length=551, window="hann", center=True)
    # 包络对齐到 STFT 帧数
    T2 = S.shape[1]
    if M.shape[1] != T2:
        M2 = np.zeros((M.shape[0], T2))
        for i in range(M.shape[1]):
            j = min(T2 - 1, int(i * T2 / M.shape[1]))
            M2[:, j] = M[:, i]
        # 沿时间插值
        for k in range(T2):
            j0 = min(M.shape[1] - 1, int(k * M.shape[1] / T2))
            M2[:, k] = M[:, j0]
    else:
        M2 = M
    幅 = np.abs(S)
    # 以器官激励的相位为初相，做「保幅度」迭代（GL 式），使相位与幅度自洽
    相位 = np.angle(S)
    y = None
    for _ in range(int(迭代)):
        Y = M2 * np.exp(1j * 相位)
        y = Lib.istft(Y, hop_length=帧样本, win_length=551, n_fft=1024, window="hann", center=True)
        y = np.nan_to_num(np.asarray(y, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        S2 = Lib.stft(y, n_fft=1024, hop_length=帧样本, win_length=551, window="hann", center=True)
        if S2.shape[1] != M2.shape[1]:
            break
        相位 = np.angle(S2)
    if y is None:
        Y = M2 * np.exp(1j * np.angle(S))
        y = Lib.istft(Y, hop_length=帧样本, win_length=551, n_fft=1024, window="hann", center=True)
    y = np.nan_to_num(np.asarray(y, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    p = float(np.abs(y).max())
    if p > 1e-9:
        y = y / p * 0.9
    出.mkdir(parents=True, exist_ok=True)
    if 保存:
        with wave.open(str(保存), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return y.astype(np.float32), dict(有效音节数=有效, 帧=T, sr=SR)


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    for 名, 文 in {
        "H1主句": "今天天气很好我听见了自己的声音我想再说一遍",
        "H1新句1": "今天的风很轻我慢慢说给你听",
        "H1新句2": "窗外的雨停了我该休息一会",
        "H1新句3": "你先别走我还想再说两句",
    }.items():
        p = 出 / f"{名}.wav"
        y, info = 合成(文, 保存=p)
        print("|", 名, info["帧"], "帧", round(len(y) / SR, 2), "s", flush=True)
    print("耗时", round(time.time() - t0, 1))
    return 0


if __name__ == "__main__":
    sys.exit(主())
