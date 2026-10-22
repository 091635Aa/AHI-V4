# -*- coding: utf-8 -*-
"""meeting68/实施/内生耳.py —— 内生听觉嵌入（自研 DSP，绝不使用外部模型；C10）。

定位：给「自听学习引擎」提供 **自己的耳**：
  · 只从波形算声学特征（F0/能量/频谱/语速），不调用 ASR/TTS/任何预训练模型；
  · 确定性：固定帧长/跳长/参数，同波形两趟嵌入逐位同；
  · 输出两层：可读 dict + 固定尺度的向量（供距离/优化用）。

特征（全部可解释）：
  F0中位/F0_90分位/F0_10分位/浊帧占比   —— 「转动频率 / 发声概率」
  能量均值/能量标准差/能量包络周期性      —— 「特殊振幅」
  语速（能量峰/秒）                      —— 「发生概率/时长」
  谱质心/谱带宽/谱滚降/谱平坦度/8 带 log-mel 均值/标准差 —— 音色/明亮度
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SR = 22050
窗 = 551          # 25 ms
跳 = 220          # 10 ms
F0_MIN, F0_MAX = 60.0, 500.0
F0_帧 = 1024

# 固定尺度（把各特征归一到可比数量级；距离用）
尺度 = {
    "F0中位": 60.0, "F0_90": 80.0, "F0_10": 50.0, "浊帧占比": 0.25,
    "能量均值": 0.05, "能量标准差": 0.04, "包络周期性": 0.15,
    "语速": 1.5,
    "谱质心": 400.0, "谱带宽": 500.0, "谱滚降": 600.0, "谱平坦度": 0.08,
}
权重 = {
    "F0中位": 1.0, "F0_90": 0.35, "F0_10": 0.35, "浊帧占比": 0.25,
    "能量均值": 0.60, "能量标准差": 0.25, "包络周期性": 0.15,
    "语速": 0.45,
    "谱质心": 0.50, "谱带宽": 0.20, "谱滚降": 0.25, "谱平坦度": 0.20,
}


def 读波形(路, sr: int = SR):
    import soundfile as sf
    x, s = sf.read(str(路), dtype="float32", always_2d=True)
    m = np.ascontiguousarray(x.mean(axis=1))
    if s != sr:
        import math
        import scipy.signal as sps
        g = math.gcd(int(s), sr)
        m = np.ascontiguousarray(sps.resample_poly(m, sr // g, int(s) // g).astype(np.float32))
    return m, sr


def 帧化(x):
    n = 1 + max(0, (len(x) - 窗) // 跳)
    return np.lib.stride_tricks.sliding_window_view(x, 窗)[:n * 跳:跳][:n]


def F0轨迹(x):
    """自相关法（自己的耳朵，不是外部模型）：40 ms 窗 / 10 ms 跳。"""
    帧长 = 882  # 40 ms @22.05k
    hop = 跳
    n = 1 + max(0, (len(x) - 帧长) // hop)
    if n <= 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    帧 = np.lib.stride_tricks.sliding_window_view(x, 帧长)[: n * hop: hop][:n]
    能量 = np.mean(帧.astype(np.float64) ** 2, axis=1)
    阈 = max(float(np.percentile(能量, 60)) * 0.1, 1e-6)
    帧 = 帧 - 帧.mean(axis=1, keepdims=True)
    自 = np.fft.rfft(帧, n=2 * 帧长)
    自 = np.fft.irfft(np.abs(自) ** 2, n=2 * 帧长)[:, :帧长]
    自 = 自 / (自[:, :1] + 1e-9)
    滞后min = int(SR / F0_MAX); 滞后max = int(SR / F0_MIN)
    段 = 自[:, 滞后min:滞后max]
    最佳 = np.argmax(段, axis=1) + 滞后min
    峰 = 段[np.arange(len(段)), 最佳 - 滞后min]
    f0 = SR / np.maximum(最佳, 1)
    浊 = (峰 > 0.30) & (能量 > 阈) & ((f0 >= F0_MIN) & (f0 <= F0_MAX))
    return f0.astype(np.float64), 浊


def 耳嵌入(路) -> dict:
    x, sr = 读波形(路)
    if x.size < 窗 * 4:
        raise ValueError("音频过短：%d 采样" % x.size)
    # 电平归一（峰值 0.9）：耳比较的是「相对动态/音色/节奏」，不是绝对音量；
    # 目标真人片段与自产语音都按同一口径归一，避免电平差主导距离。
    峰 = float(np.abs(x).max()) if x.size else 0.0
    if 峰 > 1e-9:
        x = np.ascontiguousarray(x / 峰 * 0.9).astype(np.float32)
    # 能量
    帧 = 帧化(x)
    rms = np.sqrt(np.mean(帧.astype(np.float64) ** 2, axis=1) + 1e-12)
    # F0
    f0, 浊 = F0轨迹(x)
    f0v = f0[浊 & np.isfinite(f0)] if len(f0) else np.zeros(0)
    浊帧占比 = float(np.mean(浊)) if len(浊) else 0.0
    if f0v.size >= 5:
        F0中位 = float(np.median(f0v)); F0_90 = float(np.percentile(f0v, 90)); F0_10 = float(np.percentile(f0v, 10))
    else:
        F0中位 = F0_90 = F0_10 = 0.0
    # 能量包络周期性（8–20 Hz 自相关峰；与语速/节奏相关）
    包 = rms - rms.mean()
    自 = np.correlate(包, 包, mode="full")[len(包) - 1:]
    自 = 自 / (自[0] + 1e-12)
    滞后 = np.arange(len(自)) * 跳 / sr
    带 = (滞后 >= 0.05) & (滞后 <= 0.25)
    包络周期性 = float(np.max(自[带])) if 带.any() else 0.0
    # 语速：能量包络的峰/秒（平滑后找峰）
    from scipy.signal import find_peaks
    平 = np.convolve(rms, np.ones(5) / 5, mode="same")
    峰s, _ = find_peaks(平, distance=int(0.08 * sr / 跳), prominence=float(np.std(平) * 0.4 + 1e-9))
    语速 = float(len(峰s) / (len(x) / sr)) if x.size else 0.0
    # 频谱
    import librosa
    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=跳, win_length=窗)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    功 = S + 1e-12
    cen = float(np.mean(np.sum(功 * freqs[:, None], axis=0) / np.sum(功, axis=0)))
    bw = float(np.mean(np.sqrt(np.sum(功 * (freqs[:, None] - cen) ** 2, axis=0) / np.sum(功, axis=0))))
    roll = float(np.mean(freqs[np.argmax(np.cumsum(功, axis=0) >= 0.85 * np.sum(功, axis=0)[None, :], axis=0)]))
    flat = float(np.mean(librosa.feature.spectral_flatness(S=S + 1e-12)))
    mel = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=1024, hop_length=跳, win_length=窗,
                                        n_mels=80, fmin=0, fmax=sr / 2.0, power=2.0, center=True)
    lm = np.log(mel + 1e-6)
    带均值 = [float(np.mean(lm[b * 10:(b + 1) * 10, :])) for b in range(8)]
    带标准差 = [float(np.std(lm[b * 10:(b + 1) * 10, :])) for b in range(8)]
    emb = dict(
        时长秒=round(float(len(x) / sr), 3),
        F0中位=round(F0中位, 3), F0_90=round(F0_90, 3), F0_10=round(F0_10, 3),
        浊帧占比=round(浊帧占比, 4),
        能量均值=round(float(rms.mean()), 5), 能量标准差=round(float(rms.std()), 5),
        包络周期性=round(包络周期性, 4), 语速=round(语速, 4),
        谱质心=round(cen, 2), 谱带宽=round(bw, 2), 谱滚降=round(roll, 2), 谱平坦度=round(flat, 5),
        带均值=[round(v, 4) for v in 带均值], 带标准差=[round(v, 4) for v in 带标准差],
    )
    return emb


def 向量(emb: dict) -> np.ndarray:
    """固定尺度向量（供距离/优化；带均值/标准差各 8 维）。"""
    v = []
    for k, s in 尺度.items():
        v.append(float(emb.get(k, 0.0)) / s)
    v += [float(x) / 8.0 for x in emb.get("带均值", [0.0] * 8)]
    v += [float(x) / 5.0 for x in emb.get("带标准差", [0.0] * 8)]
    return np.asarray(v, dtype=np.float64)


def 距离(embA: dict, embB: dict) -> dict:
    """加权归一化距离（总 + 分项，便于判读）。"""
    import math
    总 = 0.0
    分 = {}
    for k, w in 权重.items():
        a = float(embA.get(k, 0.0)) / 尺度[k]
        b = float(embB.get(k, 0.0)) / 尺度[k]
        d = (a - b) ** 2
        分[k] = round(float(d), 6)
        总 += w * d
    a = np.asarray(embA.get("带均值", [0.0] * 8), dtype=np.float64)
    b = np.asarray(embB.get("带均值", [0.0] * 8), dtype=np.float64)
    d带 = float(np.mean((a - b) ** 2)) / 4.0
    分["带均值"] = round(d带, 6)
    总 += 0.5 * d带
    return dict(总=round(float(总), 8), 分项=分)


def 主(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="内生耳：波形 → 听觉嵌入（自研 DSP）")
    ap.add_argument("wav")
    ap.add_argument("--出", default=None)
    参 = ap.parse_args(argv)
    emb = 耳嵌入(参.wav)
    txt = json.dumps(emb, ensure_ascii=False, indent=1)
    print(txt)
    if 参.出:
        Path(参.出).write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(主())
