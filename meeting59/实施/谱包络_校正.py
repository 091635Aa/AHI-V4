# -*- coding: utf-8 -*-
"""meeting59/实施/谱包络_校正.py —— 7 段静态谱包络校正（纯 numpy 零相位 EQ，非外部模型）。

依据：meeting58 输出层 E2 立场 §2 第 2 条「谱倾斜 + 音色指纹闭环：……或按目标 7 带 log 包络
做最小二乘校正」；本轮把它落成**确定性、零相位、纯 numpy** 的窄带 EQ：
  · 目标 = 参考录音的 7 带平均对数功率谱（活动帧）；
  · 当前 = 同一声线档案原生声带合成探针的 7 带平均对数功率谱；
  · 增益 = 目标 − 当前（dB，截 ±8dB、去均值），按带中心 log-频率线性插值成频率响应；
  · 应用 = rfft → 乘幅度响应（零相位）→ irfft；对 F0/时长/共振峰位置无搬移。

口径：这是**静态音色包络拟合**（EQ match），属于 AHI 原生 DSP 链的一部分；
不引入任何神经网络声音模型（C1/C10），不复制参考波形，参考音频不再分发。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import 声学工具 as T  # noqa: E402

带中心 = (125.0, 354.0, 707.0, 1414.0, 2449.0, 3464.0, 5657.0)
上限dB = 8.0


def 设计(参考带原始: np.ndarray, 合成带原始: np.ndarray) -> np.ndarray:
    """7 带 log 功率差 → 增益 dB（去均值、截幅）。输入是未归一化的 log 带能量。"""
    d = np.asarray(参考带原始, dtype=np.float64) - np.asarray(合成带原始, dtype=np.float64)
    d = d - float(np.mean(d))
    d = np.clip(d * (10.0 / np.log(10.0)), -上限dB, 上限dB)   # ln 功率 → dB
    return d - float(np.mean(d))


def 响应(增益dB: np.ndarray, nfft: int = 4096, fs: int = T.FS) -> np.ndarray:
    """在 rfft 频点上做 log-频率线性插值，返回幅度响应（实数、零相位）。"""
    g = np.asarray(增益dB, dtype=np.float64)
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    lf = np.log2(np.maximum(f, 20.0))
    lc = np.log2(np.asarray(带中心, dtype=np.float64))
    db = np.interp(lf, lc, g)
    # 带外两端保持最近值（低频/高频不额外抬升）
    return 10.0 ** (db / 20.0)


def 应用(x: np.ndarray, 增益dB: np.ndarray) -> np.ndarray:
    """零相位 EQ：首尾各补 0.1s 避免 FFT 回绕。"""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    填 = max(int(0.1 * T.FS), 64)
    y = np.concatenate([np.zeros(填), x, np.zeros(填)])
    nfft = 1 << int(np.ceil(np.log2(len(y))))
    Y = np.fft.rfft(y, nfft)
    out = np.fft.irfft(Y * 响应(增益dB, nfft), nfft)[:len(y)]
    return out[填:填 + x.size]


def 存(路径: str, 增益dB: np.ndarray, 元: dict) -> str:
    os.makedirs(os.path.dirname(路径), exist_ok=True)
    with open(路径, "w", encoding="utf-8") as f:
        json.dump(dict(增益dB=[round(float(v), 3) for v in 增益dB],
                       带中心Hz=list(带中心), 元=元), f, ensure_ascii=False, indent=1)
    return 路径


def 取(路径: str) -> np.ndarray | None:
    if not os.path.exists(路径):
        return None
    with open(路径, encoding="utf-8") as f:
        d = json.load(f)
    return np.asarray(d["增益dB"], dtype=np.float64)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    器 = T.取声带(300)
    from 接口契约_角色声线 import 声线档案
    a = 声线档案(名="自检", 基频=220.0)
    w, _, _ = T.原生合成("你好，我是自检。", a, 300, 器)
    参考 = T.带包络(w, 归一=False)
    合成 = T.带包络(w, 归一=False)
    g = 设计(参考, 合成)
    print("EQ 自检 增益dB:", np.round(g, 2).tolist(), "应用后 RMS 比:",
          round(T.均方根(应用(w, g)) / T.均方根(w), 4))
