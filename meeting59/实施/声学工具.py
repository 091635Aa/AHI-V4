# -*- coding: utf-8 -*-
"""meeting59/实施/声学工具.py —— 参考录音分析 + 原生声带合成 的公用工具（纯 numpy/scipy）。

口径（C1/C10）：
  · 只做经典 DSP 分析：自相关基频、倒谱包络共振峰、FFT 频带能量、能量包络调制率；
  · 合成只调用 `AHI4/meeting32/声带.py` 的原生接口（Rosenberg 声门源 + Klatt 谐振器）；
  · **不引入任何神经网络声音模型/外部声线模型**；参考录音只用于估参数，不复制、不复用波形。

只读 import：`meeting32/声带.py`、`meeting58/实施/接口契约_角色声线.py`。不改任何既有件。
"""
from __future__ import annotations

import hashlib
import os
import sys
import wave

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
MEET59 = os.path.dirname(HERE)
AHI4 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(AHI4, "meeting32"))
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))

import 声带 as 声带模块  # noqa: E402  只读复用
from 接口契约_角色声线 import 声线档案  # noqa: E402

FS = 16000
帧率 = 100
每帧样本 = FS // 帧率
临时目录 = os.path.join(MEET59, "产物", "临时")
os.makedirs(os.path.join(临时目录, "发声"), exist_ok=True)


# ───────────────────────── 读写 ─────────────────────────
def 读wav16(路径: str) -> np.ndarray:
    """任意采样率/声道 → 16k 单声道 float64（[-1,1]）。"""
    sr, x = wavfile.read(路径)
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.dtype == np.uint8:
        x = (x.astype(np.float64) - 128.0) / 128.0
    elif x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float64) / 2147483648.0
    else:
        x = x.astype(np.float64)
        m = float(np.max(np.abs(x))) if x.size else 0.0
        if m > 1.5:
            x = x / 32768.0
    x = x - float(np.mean(x)) if x.size else x
    if int(sr) != FS:
        g = np.gcd(int(sr), FS)
        x = resample_poly(x, FS // g, int(sr) // g)
    return np.ascontiguousarray(x, dtype=np.float64)


def 写wav16(路径: str, x: np.ndarray) -> str:
    os.makedirs(os.path.dirname(路径), exist_ok=True)
    y = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    pcm = (y * 32767.0).round().astype("<i2")
    with wave.open(路径, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(pcm.tobytes())
    return 路径


def 文件sha256(路径: str, 上限=64_000_000) -> str:
    h = hashlib.sha256()
    with open(路径, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def 均方根(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)) + 1e-12)


def 半音距(a: float, b: float) -> float:
    return float(12.0 * np.log2(max(float(a), 1e-9) / max(float(b), 1e-9)))


# ───────────────────────── 逐帧分析 ─────────────────────────
def _帧切片(x: np.ndarray, 帧长秒: float = 0.04, 帧移秒: float = 0.01):
    长 = max(int(round(帧长秒 * FS)), 64)
    移 = max(int(round(帧移秒 * FS)), 1)
    n = max(1, (len(x) - 长) // 移 + 1) if len(x) >= 长 else 0
    for i in range(n):
        yield i, x[i * 移:i * 移 + 长]


def 基频轨迹(x: np.ndarray) -> tuple:
    """逐帧自相关基频 + 谐噪度（归一化自相关峰值）。

    返回 (f0, hnr, 能量)，nan 表示清音/静音。FFT 求自相关，长音频也快。
    """
    n = max(1, (len(x) - int(0.04 * FS)) // int(0.01 * FS) + 1) if len(x) >= int(0.04 * FS) else 0
    f0 = np.full(n, np.nan)
    hnr = np.full(n, np.nan)
    能 = np.zeros(n)
    长 = int(0.04 * FS)
    移 = int(0.01 * FS)
    nfft = 2048
    lo = max(int(FS / 450.0), 16)      # 上限约 450Hz，避开 F1 造成的假基频
    hi = min(int(FS / 70.0), 长 - 1)   # 下限 70Hz
    for i in range(n):
        seg = x[i * 移:i * 移 + 长]
        if seg.size < 长:
            break
        seg = seg - float(np.mean(seg))
        能[i] = float(np.sqrt(np.mean(seg * seg)))
        if 能[i] < 1e-5:
            continue
        spec = np.fft.rfft(seg, nfft)
        ac = np.fft.irfft(np.abs(spec) ** 2, nfft)[:长]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        段 = ac[lo:hi]
        if 段.size == 0:
            continue
        # 取「第一个足够高的局部峰」：抑制次谐波（八度下移）误判；无峰则退回最大峰
        峰列 = []
        for k in range(1, 段.size - 1):
            if 段[k] >= 段[k - 1] and 段[k] >= 段[k + 1] and 段[k] > 0.25:
                峰列.append(k)
        if 峰列:
            最高 = max(段[k] for k in 峰列)
            候选 = [k for k in 峰列 if 段[k] >= 0.92 * 最高]
            k = min(候选)
        else:
            k = int(np.argmax(段))
        peak = float(段[k])
        if peak < 0.28:
            continue
        # 抛物线插值（更稳的基频）
        a = float(段[max(k - 1, 0)])
        b = float(段[min(k + 1, 段.size - 1)])
        den = a - 2.0 * peak + b
        off = 0.5 * (a - b) / den if abs(den) > 1e-9 else 0.0
        lag = lo + k + float(np.clip(off, -0.5, 0.5))
        f0[i] = FS / lag
        hnr[i] = peak
    return f0, hnr, 能


def _倒谱峰(seg: np.ndarray) -> list:
    """倒谱 lifter(30) 包络峰 → F1/F2/F3 候选（Hz）。"""
    w = seg * np.hanning(len(seg))
    logmag = np.log(np.abs(np.fft.rfft(w, 2048)) + 1e-9)
    cep = np.fft.irfft(logmag)
    cep[30:-30] = 0.0
    env = np.fft.rfft(cep).real
    频 = np.fft.rfftfreq(2048, 1.0 / FS)
    峰 = []
    for i in range(2, len(env) - 2):
        if env[i] > env[i - 1] and env[i] > env[i + 1] and 200.0 < 频[i] < 4200.0:
            峰.append(float(频[i]))
    return 峰[:3]


def 活动片段(x: np.ndarray, 门: float = 0.15) -> np.ndarray:
    """按 10ms 能量门切出活动段并拼接（段间留 0.10s 静音），减少长静音对统计的偏置。"""
    n = len(x) // 160
    if n < 4:
        return x
    env = np.array([均方根(x[i * 160:(i + 1) * 160]) for i in range(n)])
    参考电平 = max(float(np.percentile(env, 90)), 1e-6)
    活 = env > 门 * 参考电平
    if not np.any(活):
        return x
    # 向两侧扩张 5 帧（50ms）
    扩 = 活.copy()
    for d in range(1, 6):
        扩[d:] |= 活[:-d]
        扩[:-d] |= 活[d:]
    segs = []
    起 = None
    for i, v in enumerate(扩):
        if v and 起 is None:
            起 = i
        elif not v and 起 is not None:
            segs.append(x[起 * 160:min(i * 160, len(x))])
            起 = None
    if 起 is not None:
        segs.append(x[起 * 160:])
    if not segs:
        return x
    out = []
    for s in segs:
        out.extend([s, np.zeros(int(0.10 * FS))])
    return np.concatenate(out) if out else x


def 共振峰中位(x: np.ndarray, f0: np.ndarray = None, 能: np.ndarray = None) -> tuple:
    """浊音帧倒谱包络峰的 F1/F2/F3 中位；不足返回 (nan,nan,nan)。可复用已算的 F0/能量。"""
    if f0 is None or 能 is None:
        f0, _, 能 = 基频轨迹(x)
    if 能.size == 0:
        return (float("nan"),) * 3
    门 = max(1e-4, 0.20 * float(np.percentile(能, 85)))
    Fs = []
    for i, (idx, seg) in enumerate(_帧切片(x)):
        if i >= len(f0) or not np.isfinite(f0[i]) or 能[i] < 门:
            continue
        p = _倒谱峰(seg)
        if len(p) >= 3:
            Fs.append(p[:3])
    if len(Fs) < 3:
        return (float("nan"),) * 3
    a = np.asarray(Fs, dtype=np.float64)
    return tuple(float(np.median(a[:, j])) for j in range(3))


带边 = (0, 250, 500, 1000, 2000, 3000, 4000, 8000)


def _帧功率列(x: np.ndarray, 长=512, 移=160):
    """返回 (每帧 RMS, 每帧功率谱 list)；帧长 32ms/移 10ms。"""
    n = max(1, (len(x) - 长) // 移 + 1) if len(x) >= 长 else 0
    均 = []
    谱 = []
    for i in range(n):
        seg = x[i * 移:i * 移 + 长]
        if seg.size < 长:
            break
        r = 均方根(seg)
        均.append(r)
        谱.append(np.abs(np.fft.rfft(seg * np.hanning(长), 2048)) ** 2)
    return np.asarray(均), 谱


def 带包络(x: np.ndarray, 归一=True) -> np.ndarray:
    """7 带（E2 同口径）平均对数功率；默认减均值（只留形状）；只统计活动帧。"""
    均, 谱列 = _帧功率列(x)
    if len(谱列) == 0:
        return np.zeros(len(带边) - 1)
    门 = max(1e-4, 0.20 * float(np.percentile(均, 85)))
    频 = np.fft.rfftfreq(2048, 1.0 / FS)
    掩s = [(频 >= 带边[j]) & (频 < 带边[j + 1]) for j in range(len(带边) - 1)]
    累 = np.zeros(len(带边) - 1, dtype=np.float64)
    计 = 0
    for r, p in zip(均, 谱列):
        if r < 门:
            continue
        for j, m in enumerate(掩s):
            累[j] += float(np.mean(p[m])) if np.any(m) else 1e-12
        计 += 1
    if 计 == 0:
        return np.zeros(len(带边) - 1)
    v = np.log(累 / 计 + 1e-12)
    return v - float(np.mean(v)) if 归一 else v


def 谱倾斜(x: np.ndarray) -> float:
    """平均功率谱在 300–4000Hz 的 log-log 斜率（dB/倍频程）；只统计活动帧。"""
    均, 谱列 = _帧功率列(x)
    if len(谱列) == 0:
        return 0.0
    门 = max(1e-4, 0.20 * float(np.percentile(均, 85)))
    累 = np.zeros(1025)
    计 = 0
    for r, p in zip(均, 谱列):
        if r < 门:
            continue
        累 += p
        计 += 1
    if 计 == 0:
        return 0.0
    频 = np.fft.rfftfreq(2048, 1.0 / FS)
    m = (频 >= 300.0) & (频 <= 4000.0) & (累 > 0)
    if np.count_nonzero(m) < 8:
        return 0.0
    lf = np.log2(频[m])
    lp = 10.0 * np.log10(累[m] / 计 + 1e-12)
    A = np.vstack([lf, np.ones_like(lf)]).T
    k = np.linalg.lstsq(A, lp, rcond=None)[0][0]
    return float(k)


def 语速估计(x: np.ndarray) -> float:
    """能量包络调制率（音节/秒）：10ms 帧 RMS → 去均值 → 2–8Hz 峰。失败给 4.0。"""
    n = max(1, (len(x) - 160) // 160 + 1) if len(x) >= 160 else 0
    if n < 20:
        return 4.0
    env = np.array([均方根(x[i * 160:i * 160 + 160]) for i in range(n)])
    env = env - float(np.mean(env))
    if np.max(np.abs(env)) < 1e-6:
        return 4.0
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)), 8192))
    频 = np.fft.rfftfreq(8192, 0.01)
    m = (频 >= 2.0) & (频 <= 8.0)
    if not np.any(m):
        return 4.0
    return float(频[m][int(np.argmax(spec[m]))])


def 抖动估计(x: np.ndarray) -> float:
    """连续浊音帧 F0 的半音跳变中位（黏膜波规则性代理）。"""
    f0, _, _ = 基频轨迹(x)
    v = f0[np.isfinite(f0)]
    if v.size < 8:
        return 0.0
    d = np.abs(12.0 * np.log2(v[1:] / v[:-1]))
    return float(np.median(d))


def 谐噪比中位(x: np.ndarray) -> float:
    _, hnr, _ = 基频轨迹(x)
    v = hnr[np.isfinite(hnr)]
    return float(np.median(v)) if v.size else 0.0


def 浊音比(x: np.ndarray) -> float:
    f0, _, 能 = 基频轨迹(x)
    if f0.size == 0:
        return 0.0
    act = 能 > 1e-4
    return float(np.mean(np.isfinite(f0) & act) / max(np.mean(act), 1e-9))


# ───────────────────────── 档案 ↔ 原生声带 ─────────────────────────
def 档案律(档案: 声线档案, 语速倍: float = 1.0) -> dict:
    return dict(
        基频=float(np.clip(档案.基频, 70.0, 400.0)),
        语速=float(np.clip(档案.语速 * float(语速倍), 1.5, 7.0)),
        幅度=float(np.clip(档案.幅度, 0.1, 1.0)),
        气声度=float(np.clip(档案.气声度, 0.0, 0.8)),
        紧喉度=float(np.clip(档案.紧喉度, 0.0, 1.0)),
        粗糙度=float(np.clip(档案.粗糙度, 0.0, 1.0)),
        送气量=float(np.clip(档案.送气量, 0.0, 1.0)),
        声区=float(np.clip(档案.声区, 0.0, 1.0)),
        _ñ=0.5,
    )


def 档案音色(档案: 声线档案, 增益倍: float = 1.0) -> dict:
    return dict(
        基频=float(档案.基频),
        语速=float(档案.语速),
        幅度=float(档案.幅度),
        气声度=float(档案.气声度),
        紧喉度=float(档案.紧喉度),
        粗糙度=float(档案.粗糙度),
        送气量=float(档案.送气量),
        声区=float(档案.声区),
        共振峰缩放=tuple(float(x) for x in 档案.共振峰缩放),
        带宽缩放=1.0,
        谱倾斜=float(档案.谱倾斜),
        送气增益=0.0,
        增益=float(档案.增益) * float(增益倍),
    )


def 取声带(种子: int = 300):
    return 声带模块.声带器官(
        seed=int(种子), D=1024,
        目录=os.path.join(临时目录, "发声"),
        台账路径=os.path.join(临时目录, "发声台账.jsonl"))


def 原生合成(文本: str, 档案: 声线档案, 种子: int = 300, 器=None) -> tuple:
    """原生声带（不含人类化韵律）整句合成；返回 (波形, 元信息, 帧)。"""
    器 = 器 or 取声带(种子)
    律 = 档案律(档案)
    音 = 档案音色(档案)
    律["_音色"] = 音
    qs = 器.分位读数(np.zeros(1024, dtype=np.float64))
    帧 = 器.音节帧(文本, 律, qs)
    波形, 元 = 器.合成波形_快(帧, 音)
    return 波形, 元, 帧


def 参考统计(路径s: list) -> dict:
    """多段参考 → 汇总声学统计（先按能量门切活动段；片段间静音不影响中位）。"""
    xs = [读wav16(p) for p in 路径s]
    x0 = np.concatenate([np.concatenate([v, np.zeros(int(0.15 * FS))]) for v in xs])
    x = 活动片段(x0)
    f0, hnr, 能 = 基频轨迹(x)
    f0v = f0[np.isfinite(f0)]
    ref = dict(
        F0中位=float(np.median(f0v)) if f0v.size else 0.0,
        F0=float(np.median(f0v)) if f0v.size else 0.0,
        F0IQR=float(np.percentile(f0v, 75) - np.percentile(f0v, 25)) if f0v.size else 0.0,
        共振峰=共振峰中位(x),
        带包络=带包络(x).tolist(),
        带=带包络(x).tolist(),
        谱倾斜=谱倾斜(x),
        谐噪比=谐噪比中位(x),
        语速=语速估计(x),
        抖动=抖动估计(x),
        均方根=均方根(x),
        浊音比=浊音比(x),
        时长秒=round(len(x) / FS, 2),
        段数=len(路径s),
    )
    return ref


# ───────────────────────── 频带余弦（对增益不变） ─────────────────────────
def 带余弦(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - np.mean(a)
    b = b - np.mean(b)
    返 = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return 返


def 包络距离(a, b) -> float:
    return float(1.0 - 带余弦(a, b))


if __name__ == "__main__":
    import sys as _s
    try:
        _s.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print("声学工具自检：FS=%d 带边=%s" % (FS, 带边))
    器 = 取声带(300)
    档 = 声线档案(名="自检", 基频=200.0, 共振峰缩放=(0.95, 0.95, 0.95))
    w, _, _ = 原生合成("你好，我是自检。", 档, 300, 器)
    f0, _, _ = 基频轨迹(w)
    v = f0[np.isfinite(f0)]
    print("合成时长=%.2fs RMS=%.4f F0=%.1f 共振峰=%s 带包络=%s"
          % (len(w) / FS, 均方根(w), float(np.median(v)) if v.size else 0.0,
             共振峰中位(w), np.round(带包络(w), 2).tolist()))
