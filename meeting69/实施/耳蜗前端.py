# -*- coding: utf-8 -*-
"""meeting69/实施/耳蜗前端.py —— 像人一样的耳蜗前端 v0（自研 DSP；确定性；C10）。

不是 mel/STFT：实现 **ERB 间距滤波器组 → 毛细胞转导（半波整流+幂律压缩）→ 双时间常数适应 → 事件化**。
输出「听觉场」：适应后的时频包络（tonotopic × time，5 ms 步长）+ 事件流（onset/频谱通量峰）。

自检（不需要外部模型）：
  · 确定性：同一 wav 两趟输出 sha256 逐位同；
  · 适应：稳态段（后段）响应显著低于起始段（听而不闻）；
  · 事件：事件率与语音音节率同量级（~3–5/s）；
  · 对照：log-mel 80 无适应（同段响应不衰减）。

用法：
  & $py 耳蜗前端.py 自检 "<path>\\SpeakerA7代模型训练数据\\(1).wav"
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import wave
from pathlib import Path

import numpy as np

SR = 22050
频带数 = 64
FMIN, FMAX = 80.0, 8000.0
跳 = 110          # 5 ms
帧 = 551          # 25 ms（分析窗）
快适应秒 = 0.010
慢适应秒 = 0.200


def ERB(f):
    return 24.7 * (4.37 * f / 1000.0 + 1.0)


def 中心频率():
    # ERB 尺度等距
    lo, hi = math.log(FMIN + ERB(FMIN) / 2), math.log(FMAX + ERB(FMAX) / 2)
    # 用二分在 ERB 尺度等距取点
    def erb位置(f):
        # 数值积分 ERB 的近似位置函数
        return math.log(4.37 * f / 1000.0 + 1.0)
    a, b = erb位置(FMIN), erb位置(FMAX)
    return np.array([ (math.exp(a + (b - a) * i / (频带数 - 1)) - 1.0) * 1000.0 / 4.37
                      for i in range(频带数)])


def 读wav(路):
    with wave.open(str(路), "rb") as w:
        n = w.getnframes(); sr = w.getframerate(); ch = w.getnchannels()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        x = x.reshape(-1, 2).mean(axis=1)
    if sr != SR:
        import scipy.signal as sps
        g = math.gcd(int(sr), SR)
        x = sps.resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(x)


def 滤波组(x):
    import scipy.signal as sps
    fs = [中心频率()][0]
    out = np.zeros((len(fs), len(x)), dtype=np.float32)
    for i, f in enumerate(fs):
        bw = max(ERB(f) * 1.5, 40.0)
        lo = max(20.0, f - bw / 2); hi = min(SR / 2 - 100, f + bw / 2)
        if hi <= lo:
            continue
        sos = sps.butter(2, [lo / (SR / 2), hi / (SR / 2)], btype="band", output="sos")
        b = sps.sosfilt(sos, x).astype(np.float32)
        env = np.abs(sps.hilbert(b)).astype(np.float32)     # 复包络
        out[i] = env
    return out, fs


def 毛细胞与适应(包络):
    """半波整流已由包络保证；幂律压缩 + 快/慢适应。返回 (压缩后, 适应后)。"""
    comp = np.power(np.maximum(包络, 0.0), 0.4).astype(np.float32)
    dt = 跳 / SR
    af = np.zeros(comp.shape[0], dtype=np.float32)
    as_ = np.zeros(comp.shape[0], dtype=np.float32)
    out = np.zeros(comp.shape, dtype=np.float32)
    kf = dt / 快适应秒; ks = dt / 慢适应秒
    for t in range(comp.shape[1]):
        x = comp[:, t]
        af += (x - af) * kf
        as_ += (x - as_) * ks
        适 = x - (0.6 * af + 0.4 * as_)
        out[:, t] = np.maximum(适, 0.0)
    return comp, out


def 事件(适应后, 压缩后):
    # 帧级能量（适应后）与频谱通量（压缩后）
    e = np.log(np.mean(适应后, axis=0) + 1e-6)
    flux = np.sum(np.maximum(np.diff(压缩后, axis=1), 0.0), axis=0)
    flux = np.concatenate([[0.0], flux])
    阈 = np.median(flux) + 2.0 * (np.percentile(flux, 75) - np.median(flux) + 1e-9)
    峰 = []
    for t in range(1, len(flux) - 1):
        if flux[t] > 阈 and flux[t] >= flux[t - 1] and flux[t] >= flux[t + 1]:
            if not 峰 or (t - 峰[-1]) * 跳 / SR > 0.05:      # 最小间隔 50 ms
                峰.append(t)
    return [dict(t=round(t * 跳 / SR, 3), 通量=round(float(flux[t]), 4)) for t in 峰], dict(
        能量均=round(float(e.mean()), 4), 通量均值=round(float(flux.mean()), 4))


def 分析(路):
    x = 读wav(路)
    包络样本, fs = 滤波组(x)
    T = 包络样本.shape[1] // 跳
    包络 = 包络样本[:, : T * 跳].reshape(包络样本.shape[0], T, 跳).mean(axis=2)  # (bands, T)
    压缩, 适应 = 毛细胞与适应(包络)
    事件s, 统 = 事件(适应, 压缩)
    场 = 适应.T.astype(np.float32)                  # (T, bands) = 听觉场
    return dict(
        采样率=SR, 中心频率=[round(float(f), 1) for f in fs],
        场=场, 事件=事件s, 统计=统,
        适应比=_适应比(适应), 时长秒=round(len(x) / SR, 3),
    )


def _适应比(适应):
    """稳态适应：后 60% 响应 / 前 20% 响应（越小=适应越强）。"""
    n = 适应.shape[1]
    if n < 20:
        return 1.0
    前 = float(np.mean(适应[:, : max(1, int(n * 0.2))]))
    后 = float(np.mean(适应[:, int(n * 0.4):]))
    return round(后 / (前 + 1e-9), 4)


def 数组sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def 自检(路):
    r1 = 分析(路); r2 = 分析(路)
    同 = (数组sha(r1["场"]) == 数组sha(r2["场"])
          and [e["t"] for e in r1["事件"]] == [e["t"] for e in r2["事件"]])
    数 = len(r1["事件"]); 时 = r1["时长秒"]
    场 = r1["场"]; n = len(场)
    前场 = float(np.mean(场[: max(1, n // 5)]))
    后场 = float(np.mean(场[int(n * 0.4):]))
    # log-mel 对照（无适应）
    import librosa
    x = 读wav(路)
    S = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=1024, hop_length=跳, win_length=帧,
                                       power=2.0, n_mels=80, fmin=0, fmax=SR / 2.0)
    mel = np.log(S + 1e-6).T
    m = mel.shape[0]
    前 = float(np.mean(mel[: max(1, m // 5)])); 后 = float(np.mean(mel[int(m * 0.4):]))
    报 = dict(工具="耳蜗前端.py", 文件=str(路), 时长秒=时, 频带数=频带数,
              确定性同=同, 事件数=数, 事件率=round(数 / max(时, 1e-9), 2),
              适应比=r1["适应比"], 听觉场前段=round(前场, 4), 听觉场后段=round(后场, 4),
              统计=r1["统计"],
              mel对照=dict(前段=round(前, 3), 后段=round(后, 3), 变化=round(后 - 前, 3)),
              边界="自检只验前端性质，不验可懂度；不承诺像人耳整体")
    return 报


def 主(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="耳蜗前端 v0（ERB 滤波+毛细胞+适应+事件）")
    子 = ap.add_subparsers(dest="命令", required=True)
    p = 子.add_parser("自检"); p.add_argument("wav")
    参 = ap.parse_args(argv)
    if 参.命令 == "自检":
        r = 自检(参.wav)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return 0


if __name__ == "__main__":
    sys.exit(主())
