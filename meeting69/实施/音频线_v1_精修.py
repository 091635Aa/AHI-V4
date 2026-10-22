# -*- coding: utf-8 -*-
"""meeting69/实施/音频线_v1_精修.py —— 音频线精修 v1（预注册 `26`）：去电音 + 去沙哑 + 收尾。

只做**声码器与移调**的精修（不训练、不改 v13、不改 `meeting68/音频线_v0.py`）：
  · 电音：Griffin-Lim 迭代 64→256、momentum 0.99、可选 mel 时间 ×2 上采样（跳 220→110）、精确 length；
  · 沙哑：高兴/低落的时域 `librosa.pitch_shift`（相位声码器）→ **纯 `scipy.resample_poly` 移调**（时长随率变）；
  · 收尾：尾部静音裁剪 + 5 ms 淡入淡出。
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

N_FFT = 声.N_FFT
窗长 = 声.窗长
跳长 = 声.跳长
MEL带数 = 声.MEL带数
MEL下限, MEL上限 = 声.MEL下限, 声.MEL上限
AMIN = 1e-6
GL种子 = 0
目标峰值 = 0.9

_模型 = None
# 与 meeting68/音频线_v0 同一 ckpt（只读；保证 A0 能逐位复现 v0）
CKPT = M66 / "产物" / "模型_v13_K4_320" / "ckpt_e0150.pt"

def 载模型():
    global _模型
    if _模型 is None:
        _模型 = 声.载ckpt(CKPT, 设备="cpu")
    return _模型


def 推理mel(文本, 每音节帧数=24, 迭代=4, 阻尼=0.5):
    import os
    os.environ["M66_迭代推理"] = str(int(迭代))
    os.environ["M66_阻尼"] = str(float(阻尼))
    mel, 有效, 跳 = 声.推理mel(载模型(), 文本, int(每音节帧数))
    return np.asarray(mel, np.float32), 有效, 跳


def 上采样mel(mel, 倍=2):
    """mel 时间轴线性插值 ×倍（帧率 100 Hz→200 Hz；总时长不变）。"""
    m = np.asarray(mel, np.float64)
    T, B = m.shape
    if 倍 <= 1:
        return m
    t旧 = np.arange(T, dtype=np.float64)
    t新 = np.linspace(0.0, T - 1.0, (T - 1) * 倍 + 1)
    out = np.empty((len(t新), B), np.float64)
    for b in range(B):
        out[:, b] = np.interp(t新, t旧, m[:, b])
    return out


def 平滑mel(mel, 帧=3):
    """mel 时间轴中值平滑（去帧级抖动；默认 3 帧）。"""
    if 帧 <= 1:
        return np.asarray(mel, np.float64)
    from scipy.ndimage import median_filter
    return median_filter(np.asarray(mel, np.float64), size=(int(帧), 1), mode="nearest")


def mel转波_精修(mel, 迭代=256, momentum=0.99, 时间倍率=1, 平滑=0, 种子=GL种子, 峰值=目标峰值,
                 nfft=None, 窗=None):
    """log-mel → 波形（精修 GL）。librosa mel_to_stft 伪逆 + 高迭代/momentum + 精确 length。

    nfft=2048/窗=1024（A4 臂）：更长分析窗 ⇒ 频率分辨更高、相位更稳（jitter 显著下降）。"""
    lib = 声.librosa模块()
    nfft = int(nfft or N_FFT)
    win = int(窗 or min(窗长, nfft))
    m = np.asarray(mel, np.float64)
    m = np.nan_to_num(m, nan=MEL下限, posinf=MEL上限, neginf=MEL下限)
    m = np.clip(m, MEL下限, MEL上限)
    if 平滑 > 1:
        m = 平滑mel(m, 平滑)
    if 时间倍率 > 1:
        m = 上采样mel(m, 时间倍率)
    功率 = np.maximum(np.exp(m) - AMIN, 0.0)
    幅度 = lib.feature.inverse.mel_to_stft(功率.T, sr=int(SR), n_fft=nfft, power=2.0,
                                          fmin=声.FMIN, fmax=SR / 2.0)
    幅度 = np.maximum(np.asarray(幅度, np.float64), 0.0)
    跳 = max(1, int(round(跳长 / max(1, 时间倍率))))
    length = max(1, (幅度.shape[1] - 1) * 跳)
    入口 = getattr(lib, "griffinlim", None) or lib.effects.griffinlim
    波形 = 入口(幅度, n_iter=int(迭代), hop_length=int(跳), win_length=int(win), n_fft=int(nfft),
                window="hann", center=True, pad_mode="constant", momentum=float(momentum),
                init="random", random_state=int(种子), length=int(length))
    波形 = np.nan_to_num(np.asarray(波形, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if 峰值 and 波形.size:
        p = float(np.abs(波形).max())
        if p > 1e-8:
            波形 = 波形 / p * float(峰值)
    return 波形.astype(np.float32)


def 移调_重采样(x, 半音):
    """纯重采样移调（chipmunk）：音高 ×2^(n/12)，时长 ×2^(-n/12)。无相位声码器。"""
    from fractions import Fraction
    from scipy.signal import resample_poly
    if abs(float(半音)) < 1e-9:
        return np.asarray(x, np.float32)
    r = 2.0 ** (float(半音) / 12.0)
    fr = Fraction(r).limit_denominator(1000)
    # 读快 r 倍 ⇒ 长度 /r：resample_poly(up=分母, down=分子) 得 len*分母/分子
    y = resample_poly(np.asarray(x, np.float64), fr.denominator, fr.numerator)
    return np.asarray(y, np.float32)


def 收尾(x, 尾部毫秒=30, 淡毫秒=5):
    """尾部静音裁剪（保留 tail 毫秒）+ 5 ms 淡入淡出。"""
    y = np.asarray(x, np.float64).copy()
    if y.size == 0:
        return y.astype(np.float32)
    p = float(np.abs(y).max())
    if p > 1e-9:
        阈 = 0.01 * p
        非静 = np.where(np.abs(y) > 阈)[0]
        if 非静.size:
            末 = min(len(y), int(非静[-1] + SR * 尾部毫秒 / 1000.0))
            y = y[:末]
    n = int(SR * 淡毫秒 / 1000.0)
    if y.size > 2 * n > 0:
        窗 = np.linspace(0.0, 1.0, n)
        y[:n] *= 窗
        y[-n:] *= 窗[::-1]
    p = float(np.abs(y).max())
    if p > 1e-9:
        y = y / p * 目标峰值
    return y.astype(np.float32)


# ── 情绪/风格预置（与音频线_v0 同一张贴，只读口径） ──
情绪表 = dict(声=None)  # 占位，下面从 v0 读
_v0 = 载("音频线_v0", M69.parent / "meeting68" / "实施" / "音频线_v0.py")
情绪表 = _v0.情绪表
风格表 = _v0.风格表


def 参数(情绪="平静", 风格="正常"):
    p = dict(情绪表.get(情绪, 情绪表["平静"]))
    p.update(风格表.get(风格, {}))
    return p


def 合成_精修(文本, 情绪="平静", 风格="正常", 臂="A4", 移调="P1"):
    """→ dict(波形, mel, 控制, 时长, sha 由调用方算)。

    臂：A0=v0 原样复现（含 v0 的「无标点也插停顿」回退行为，用于 F5 对拍）；
        A1=256 迭代+momentum；A2=A1+mel 时间×2；A3=A2+mel 平滑；
        A4=**n_fft 2048 / 窗 1024**+256 迭代+momentum（去电音候选）；
        A1–A4 均**不再插错误停顿**（v0 的回退缺陷），并做收尾。"""
    p = 参数(情绪, 风格)
    mel, 有效, 跳 = 推理mel(文本, p["每音节帧数"])
    if abs(float(p.get("倾斜", 0.0))) > 1e-9:
        tilt = np.linspace(-1.0, 1.0, mel.shape[1], dtype=np.float32) * float(p["倾斜"])
        mel = (mel + tilt[None, :]).astype(np.float32)
    半音 = float(p.get("音高", 0.0))
    if 臂 == "A0":
        x = 声.mel转波(mel)                                    # v0 原路径（64 迭代、无 momentum、无 length）
    elif 臂 == "A1":
        x = mel转波_精修(mel, 迭代=256, momentum=0.99, 时间倍率=1, 平滑=0)
    elif 臂 == "A2":
        x = mel转波_精修(mel, 迭代=256, momentum=0.99, 时间倍率=2, 平滑=0)
    elif 臂 == "A3":
        x = mel转波_精修(mel, 迭代=256, momentum=0.99, 时间倍率=2, 平滑=3)
    elif 臂 == "A4":
        x = mel转波_精修(mel, 迭代=256, momentum=0.99, 时间倍率=1, 平滑=0, nfft=2048, 窗=1024)
    else:
        raise ValueError(臂)
    if abs(半音) > 1e-9:
        if 移调 == "P0":                                       # v0 原路径：时域相位声码器
            import librosa
            x = librosa.effects.pitch_shift(np.asarray(x, np.float32), sr=SR, n_steps=半音)
        elif 移调 == "P1":                                     # 纯重采样
            x = 移调_重采样(x, 半音)
        else:                                                  # P2：不移调
            pass
    if abs(float(p.get("增益", 0.0))) > 1e-9:
        x = x * (10.0 ** (float(p["增益"]) / 20.0))
    if 臂 == "A0":                                             # v0 原样：按原逻辑插停顿 + 归一
        x = np.asarray(x, np.float32)
        if float(p.get("停顿毫秒", 0.0)) > 1e-6:
            x = _v0._插停顿(x, 文本, float(p["停顿毫秒"]))
        p峰 = float(np.abs(x).max()) if x.size else 0.0
        if p峰 > 1e-9:
            x = x / p峰 * 0.9
    else:
        x = 收尾(x)                                            # A1–A4：不插错误停顿 + 收尾
    return dict(波形=np.asarray(x, np.float32), mel=mel, 控制=p, 有效音节数=有效,
                时长秒=round(float(np.asarray(x).size) / SR, 3))
