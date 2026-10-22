# -*- coding: utf-8 -*-
"""meeting63/实施/耳_内生.py —— 内生听觉前段（波形侧，零外挂）。

所有者口径：「我们说话的时候，脑子里可以听到自己说话」。本件就是那对**内生的耳**：
只吃**波形**（自己合成出来的、或参考录音的），用纯 numpy/scipy 的确定性 DSP，
产出**音素段级的可测感知量**（清/浊/静的切段、时长、谱重心、谱峰）。

★ 关键设计（为了可证伪，不许含糊）：
  · 本件**只读波形**，绝不读生成器的内部帧（`声源/噪形/f1/f2…`）。若把内部帧喂进来，
    就等价于「偷看答案」⇒ 预测误差会恒为 0（`学_咬字反馈.py` 的 must-fail M1 专测这一条）。
  · 因此本件是「外侧/内侧听觉」的**可测代理**：波形 → 感知量，中间不经过任何外部模型、
    任何权重、任何参考音频模板（零外挂，C10）。
  · 确定性（C13）：无随机；固定帧长/跳长/阈值；同输入两趟逐位同。

口径：`听(波形)` → 段列 + 汇总；`听文件(path)` → 同上。
本件不产生声音、不改任何既有件（C14）。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

FS = 44100
帧长 = 882          # 20 ms
跳 = 441            # 10 ms
静阈dB = 30.0       # 低于（峰值−30dB）记为静音
浊阈 = 0.35         # 归一化自相关峰 ≥ 此值记为浊（语音学常用浊/清分界量级）

# 类别编码（供外部稳定引用）
静, 清, 浊 = "静", "清", "浊"


def _分帧(x: np.ndarray) -> dict:
    """滑动分帧 → 逐帧 能量/浊度/过零率/谱重心/谱峰（纯 DSP，确定性）。"""
    n = len(x)
    if n < 帧长:
        x = np.pad(x, (0, 帧长 - n))
        n = 帧长
    k = 1 + (n - 帧长) // 跳
    idx = np.arange(帧长)[None, :] + 跳 * np.arange(k)[:, None]
    fr = x[idx]
    fr = fr - fr.mean(axis=1, keepdims=True)
    能量 = np.sqrt(np.mean(fr * fr, axis=1))
    能量dB = 20.0 * np.log10(np.maximum(能量, 1e-9))

    # 浊度：自相关（FFT 加速），归一化到 lag=0
    nfft = 1 << int(np.ceil(np.log2(2 * 帧长)))
    F = np.fft.rfft(fr * np.hanning(帧长)[None, :], n=nfft, axis=1)
    ac = np.fft.irfft(F * np.conj(F), n=nfft, axis=1)
    ac /= np.maximum(ac[:, :1], 1e-20)
    lo, hi = max(1, int(FS / 450.0)), min(nfft - 1, int(FS / 70.0))
    seg = ac[:, lo:hi + 1]
    浊度 = seg.max(axis=1)

    # 过零率
    过零 = np.mean(np.abs(np.diff(np.signbit(fr), axis=1)), axis=1)

    # 谱重心 / 谱峰（300–16000Hz）
    P = np.abs(np.fft.rfft(fr * np.hanning(帧长)[None, :], n=nfft, axis=1)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / FS)
    m = (f >= 300.0) & (f <= 16000.0)
    fm, Pm = f[m], P[:, m]
    Pm = Pm.copy()
    Pm[:, 0] = 0.0
    T = Pm.sum(axis=1)
    T = np.where(T > 0, T, 1.0)
    重心 = (Pm @ fm) / T
    峰 = fm[np.argmax(Pm, axis=1)]
    return dict(k=int(k), 能量dB=能量dB, 浊度=浊度, 过零率=过零, 谱重心=重心, 谱峰=峰)


def 切段(x: np.ndarray, 帧: dict = None) -> list:
    """波形 → 段列 [(类, 起样本, 止样本)]；类∈{静,清,浊}。"""
    if 帧 is None:
        帧 = _分帧(x)
    k = 帧["k"]
    峰dB = float(np.max(帧["能量dB"])) if k else 0.0
    类 = np.empty(k, dtype=object)
    for i in range(k):
        if 帧["能量dB"][i] < 峰dB - 静阈dB:
            类[i] = 静
        elif 帧["浊度"][i] >= 浊阈:
            类[i] = 浊
        else:
            类[i] = 清
    # 单帧碎片并入前段（消除抖动）
    for i in range(1, k - 1):
        if i >= 1 and 类[i] != 类[i - 1] and 类[i] != 类[i + 1]:
            类[i] = 类[i - 1]
    # 合并同类相邻
    段 = []
    i = 0
    while i < k:
        t = 类[i]
        j = i
        while j < k and 类[j] == t:
            j += 1
        a = i * 跳
        b = min((j - 1) * 跳 + 帧长, len(x))
        段.append((str(t), int(a), int(b)))
        i = j
    return 段


def _段统计(帧: dict, 段: list) -> dict:
    """对每段取：时长 ms、谱重心中位、谱峰中位、能量中位。"""
    出 = []
    kf = 帧["k"]
    for t, a, b in 段:
        起帧 = int(round(a / 跳))
        止帧 = min(kf, int(round(b / 跳)) + 1)
        if 止帧 <= 起帧:
            止帧 = min(kf, 起帧 + 1)
        扇 = slice(起帧, 止帧)
        出.append(dict(
            类=t, ms=round((b - a) / FS * 1000.0, 1),
            谱重心Hz=round(float(np.median(帧["谱重心"][扇])), 1) if 止帧 > 起帧 else 0.0,
            谱峰Hz=round(float(np.median(帧["谱峰"][扇])), 1) if 止帧 > 起帧 else 0.0,
            能量dB=round(float(np.median(帧["能量dB"][扇])), 2) if 止帧 > 起帧 else 0.0,
            浊度=round(float(np.median(帧["浊度"][扇])), 3) if 止帧 > 起帧 else 0.0,
        ))
    return 出


def 听(x, 含段=True) -> dict:
    """波形 → 内生感知读数（段列 + 汇总）。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    if np.max(np.abs(x)) > 1.2:
        x = x / 32768.0
    帧 = _分帧(x)
    段 = 切段(x, 帧)
    seg = _段统计(帧, 段)
    各 = {静: [], 清: [], 浊: []}
    for s in seg:
        各[s["类"]].append(s)
    ms中位 = {t: (round(float(np.median([s["ms"] for s in v])), 1) if v else 0.0)
             for t, v in 各.items()}
    总ms = {t: round(float(sum(s["ms"] for s in v)), 1) for t, v in 各.items()}
    清谱 = [s["谱重心Hz"] for s in 各[清]] or [0.0]
    清峰 = [s["谱峰Hz"] for s in 各[清]] or [0.0]
    浊谱 = [s["谱重心Hz"] for s in 各[浊]] or [0.0]
    总 = max(sum(总ms.values()), 1e-9)
    return dict(
        时长秒=round(len(x) / FS, 3),
        段数=len(seg), 静段数=len(各[静]), 清段数=len(各[清]), 浊段数=len(各[浊]),
        **{"%s时长中位ms" % t: ms中位[t] for t in (静, 清, 浊)},
        **{"%s总时长ms" % t: 总ms[t] for t in (静, 清, 浊)},
        清总占比=round(总ms[清] / 总, 4), 浊总占比=round(总ms[浊] / 总, 4),
        静总占比=round(总ms[静] / 总, 4),
        清谱重心中位Hz=round(float(np.median(清谱)), 1),
        清谱峰中位Hz=round(float(np.median(清峰)), 1),
        浊谱重心中位Hz=round(float(np.median(浊谱)), 1),
        清浊总时长比=round(总ms[清] / max(总ms[浊], 1e-9), 4),
        段=seg if 含段 else [])


def 听文件(路径: str) -> dict:
    fs, x = wavfile.read(路径)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.max(np.abs(x)) > 1.2:
        x = x / 32768.0
    if fs != FS:
        g = math.gcd(int(fs), FS)
        x = resample_poly(x, FS // g, fs // g)
    r = 听(x)
    r["文件"] = os.path.basename(路径)
    r["源采样率"] = int(fs)
    return r


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    import json
    AHI4 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    M61 = os.path.join(AHI4, "meeting61")
    M62 = os.path.join(AHI4, "meeting62")
    print("=== 内生耳 自检（波形侧切段） ===")
    样 = [
        ("真人·EmoCompanion·01", os.path.join(M61, "素材", "EmoCompanion", "ref44_EmoCompanion_01.wav")),
        ("真人·SpeakerA·01", os.path.join(M61, "素材", "SpeakerA", "ref44_SpeakerA_01.wav")),
        ("真人·洛天依·清唱01", os.path.join(M61, "素材", "洛天依", "ref44_洛天依_清唱_01.wav")),
        ("合成v7·EmoCompanion", os.path.join(M62, "产物", "交付_v7", "说话", "说话_v7_EmoCompanion.wav")),
        ("合成v7·SpeakerA", os.path.join(M62, "产物", "交付_v7", "说话", "说话_v7_SpeakerA.wav")),
        ("合成v7·洛天依", os.path.join(M62, "产物", "交付_v7", "说话", "说话_v7_洛天依.wav")),
    ]
    for 名, p in 样:
        if not os.path.exists(p):
            print("%-22s [缺]" % 名)
            continue
        r = 听文件(p)
        print("%-22s 时长%5.2fs 段%3d 清%2d/浊%2d/静%2d | 清中位%6.1fms 浊中位%6.1fms "
              "清占比%.3f 清浊比%.3f 清重心%6.1fHz"
              % (名, r["时长秒"], r["段数"], r["清段数"], r["浊段数"], r["静段数"],
                 r["清时长中位ms"], r["浊时长中位ms"], r["清总占比"],
                 r["清浊总时长比"], r["清谱重心中位Hz"]))
    print("-" * 96)
    print("确定性：同输入两趟逐位同 =",
          json.dumps(听文件(样[0][1])["段"]) == json.dumps(听文件(样[0][1])["段"]))
