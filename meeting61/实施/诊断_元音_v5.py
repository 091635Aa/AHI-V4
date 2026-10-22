# -*- coding: utf-8 -*-
"""meeting61/实施/诊断_元音_v5.py —— 元音层专项诊断（波形级，无听感通道）。

用户反馈：v5 的「辅音表」已能听懂，其余（元音表 / 整句）仍模糊。
本诊断只回答一个问题：**元音共振峰在合成波形里到底有没有成形、够不够突出**，
以及整句里每个音节的元音是否真的到达了目标。

量测项（全部在输出波形上取，不读内部参数）：
  1) 倒谱平滑包络 → 在目标 ±25% 窗内取峰，报 F1/F2/F3 实测与误差；
  2) 峰突出度 prominence = dB(峰) − dB(周围谱底中位)，< 6dB 时听感上「没有元音」；
  3) 峰间调制度 = dB(F1峰) − dB(F1/F2 之间的谷)，衡量两个共振峰是否分开；
  4) 谐波斜率（dB/oct）：声门源 + 唇辐射的净谱倾斜，与参考原声对照；
  5) 元音空间两两距离（F1,F2 平面），< 300Hz 时两个元音会被听成同一个；
  6) 整句逐音节元音段：目标 vs 实测 F1/F2 轨迹。

用法：``python 诊断_元音_v5.py`` → 打印 + 写 产物/诊断/诊断_元音_v5.json
纯 numpy/scipy（C1/C10），确定性（C13）。
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
MEET61 = os.path.dirname(HERE)
AHI4 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))

import 音韵_v5 as Y  # noqa: E402
import 韵律_v5 as H5  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

FS = 44100
出目录 = os.path.join(MEET61, "产物", "诊断")
os.makedirs(出目录, exist_ok=True)

元音表 = ["a1", "o1", "e1", "i1", "u1", "v1", "er2", "ai1", "ei1",
        "ao1", "ou1", "an1", "en1", "ang1", "eng1"]


def 读档(名="EmoCompanion") -> 声线档案:
    with open(os.path.join(MEET61, "产物", "声线档案", 名 + ".json"), encoding="utf-8") as f:
        return 声线档案.从dict(json.load(f))


def _db(a):
    return 20.0 * math.log10(max(float(a), 1e-12))


def _倒谱包络(x, 提升秒=0.0022, fs=FS):
    """倒谱平滑 → 谱包络（压掉谐波，保留共振峰形状）。"""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 256:
        return np.zeros(0), np.zeros(0)
    x = x - float(np.mean(x))
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    logP = np.log(np.abs(X) + 1e-12)
    cep = np.fft.irfft(logP)
    L = max(4, int(提升秒 * fs))
    cep[L:-L] = 0.0
    env = np.exp(np.fft.rfft(cep).real)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return f, env


def _峰(env, f, lo, hi):
    m = (f >= lo) & (f <= hi)
    if not np.any(m):
        return 0.0, 0.0
    ff, ee = f[m], env[m]
    k = int(np.argmax(ee))
    return float(ff[k]), float(ee[k])


def _谷(env, f, lo, hi):
    m = (f >= lo) & (f <= hi)
    if not np.any(m):
        return 0.0
    return float(np.min(env[m]))


def _谱底(env, f, 目标F, 带=(150.0, 4000.0)):
    m = (f >= 带[0]) & (f <= 带[1])
    for t in 目标F:
        if t:
            m &= ~((f >= t * 0.72) & (f <= t * 1.28))
    if not np.any(m):
        return 1e-9
    return float(np.median(env[m]))


def _谐波斜率(x, fs=FS, fmax=4000.0):
    """峰值拾取谐波 → dB / oct 线性拟合（相对 1kHz）。"""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 1024:
        return None
    x = x - float(np.mean(x))
    n = len(x)
    X = np.abs(np.fft.rfft(x * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    # 自相关求 F0
    fr = x * np.hanning(n)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    ac = np.fft.irfft(np.abs(np.fft.rfft(fr, n=nfft)) ** 2, n=nfft)
    ac /= max(ac[0], 1e-20)
    最短, 最长 = int(fs / 450.0), min(nfft - 1, int(fs / 70.0))
    lag = int(np.argmax(ac[最短:最长 + 1])) + 最短
    f0 = fs / lag
    k = 1
    点 = []
    while k * f0 < min(fmax, fs * 0.45):
        c = k * f0
        窗 = (f >= c * 0.94) & (f <= c * 1.06)
        if np.any(窗):
            pk = float(np.max(X[窗]))
            if pk > 1e-9:
                点.append((c, 20 * math.log10(pk)))
        k += 1
    if len(点) < 4:
        return None
    xs = np.log2(np.array([p[0] for p in 点]) / 1000.0)
    ys = np.array([p[1] for p in 点])
    ys = ys - float(np.mean(ys))
    w = np.polyfit(xs, ys, 1)
    return dict(f0=round(f0, 1), 斜率dB每倍频程=round(float(w[0]), 2),
                点数=len(点), 相对1kHz偏移dB=round(float(np.polyval(w, 0.0)), 2))


def _浊区间(帧):
    return [i for i, t in enumerate(帧["声源"]) if int(t) == 1]


def 量元音(w, 帧, 期望F3, 取窗=(0.35, 0.75)):
    """在浊音段中段取窗，报峰位/突出度/调制度。"""
    idx = _浊区间(帧)
    if not idx:
        return {}
    i0, i1 = idx[0], idx[-1]
    n = i1 - i0 + 1
    a = (i0 + int(n * 取窗[0])) * 441
    b = (i0 + int(n * 取窗[1])) * 441 + 441
    seg = w[a:min(b, len(w))]
    if len(seg) < 1024:
        return {}
    f, env = _倒谱包络(seg)
    实 = []
    for t in 期望F3:
        if not t:
            实.append(0.0)
            continue
        c, v = _峰(env, f, t * 0.75, t * 1.25)
        实.append(round(c, 1))
    底 = _谱底(env, f, 期望F3)
    突 = []
    for t, c in zip(期望F3, 实):
        if not t or not c:
            突.append(None)
            continue
        _, v = _峰(env, f, c * 0.85, c * 1.15)
        突.append(round(20 * math.log10(max(v, 1e-12) / 底), 1))
    mod = None
    if 实[0] and 实[1] and 实[1] > 实[0]:
        谷 = _谷(env, f, 实[0] * 1.10, 实[1] * 0.90)
        _, v1 = _峰(env, f, 实[0] * 0.85, 实[0] * 1.15)
        if 谷 > 0 and v1 > 0:
            mod = round(20 * math.log10(v1 / 谷), 1)
    return dict(F1=实[0], F2=实[1], F3=实[2], 突出度dB=突, 峰谷调制度dB=mod,
                噪比=_谐波斜率(seg) or {})


def 短句元音检查(档, 句, 种子=300):
    """整句：逐音节取浊音段中段，比目标与实测。"""
    w, 元, 帧 = H5.人类韵律_v5(档, 种子).说(句)
    音s = Y.音节列表(句)
    idx = _浊区间(帧)
    if not idx:
        return []
    段s = []
    cur = [idx[0]]
    for k in idx[1:]:
        if k == cur[-1] + 1:
            cur.append(k)
        else:
            段s.append(cur)
            cur = [k]
    段s.append(cur)
    出 = []
    for s, 段 in zip(音s, 段s):
        目标 = s["目标s"][-1]
        a = (段[0] + int(len(段) * 0.4)) * 441
        b = (段[0] + int(len(段) * 0.85)) * 441 + 441
        seg = w[a:min(b, len(w))]
        if len(seg) < 512:
            continue
        f, env = _倒谱包络(seg)
        f1, _ = _峰(env, f, 目标[0] * 0.7, 目标[0] * 1.4)
        f2, _ = _峰(env, f, 目标[1] * 0.7, 目标[1] * 1.4)
        出.append(dict(拼音=s["拼音"], 韵=s["韵"], 浊帧=len(段),
                       目标F1=round(目标[0], 1), 实测F1=round(f1, 1),
                       目标F2=round(目标[1], 1), 实测F2=round(f2, 1)))
    return 出


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    报 = dict(工具="诊断_元音_v5.py", 采样率=FS)
    档 = 读档("EmoCompanion")

    # ---------- 1) 元音表（声线 F0 ≈ 350Hz）----------
    器 = H5.人类韵律_v5(档, 300)
    A = {}
    print("=" * 96)
    print("元音表 · EmoCompanion（合成 F0≈350Hz）  [目标 F1/F2/F3 → 实测 | 突出度 dB | 峰谷调制度 dB]")
    print("=" * 96)
    for p in 元音表:
        w, _, 帧 = H5.人类韵律_v5(档, 300).说音节(p)
        s = Y.解析音节(p)
        目标 = [t * v for t, v in zip(s["目标s"][-1], (档.共振峰缩放[0],
                                                   档.共振峰缩放[1], 档.共振峰缩放[2]))]
        r = 量元音(w, 帧, 目标)
        A[p] = dict(韵=s["韵"], 目标=[round(t, 1) for t in 目标], **r)
        print("%-5s %-5s 目标 %s → 实测 %s | 突出 %s | 调制度 %s | 谐波 %s"
              % (p, s["韵"],
                 " ".join("%5.0f" % t for t in 目标),
                 " ".join("%5.0f" % (r.get(k) or 0) for k in ("F1", "F2", "F3")),
                 r.get("突出度dB"), r.get("峰谷调制度dB"),
                 (r.get("噪比") or {}).get("斜率dB每倍频程")))
    报["元音表_EmoCompanion"] = A

    # ---------- 2) 元音表（120Hz 测试声线，排除高 F0 干扰）----------
    档低 = 档.派生(基频=120.0)
    B = {}
    print("-" * 96)
    print("元音表 · 120Hz 测试声线（谐波够密，可检验共振峰是否成形）")
    for p in 元音表:
        w, _, 帧 = H5.人类韵律_v5(档低, 300).说音节(p)
        s = Y.解析音节(p)
        目标 = [t * v for t, v in zip(s["目标s"][-1], (档.共振峰缩放[0],
                                                   档.共振峰缩放[1], 档.共振峰缩放[2]))]
        r = 量元音(w, 帧, 目标)
        B[p] = dict(韵=s["韵"], 目标=[round(t, 1) for t in 目标], **r)
        误差 = []
        for t, c in zip(目标, [r.get("F1"), r.get("F2"), r.get("F3")]):
            if t and c:
                误差.append(abs(c - t) / t)
        B[p]["最大峰位误差"] = round(max(误差), 3) if 误差 else None
        print("%-5s %-5s 目标 %s → 实测 %s | 突出 %s | 调制度 %s | 误差 %s"
              % (p, s["韵"],
                 " ".join("%5.0f" % t for t in 目标),
                 " ".join("%5.0f" % (r.get(k) or 0) for k in ("F1", "F2", "F3")),
                 r.get("突出度dB"), r.get("峰谷调制度dB"),
                 B[p]["最大峰位误差"]))
    报["元音表_120Hz"] = B

    # ---------- 3) 元音空间可分性（120Hz 声线）----------
    def 取(名_):
        r = B.get(名_)
        return (r.get("F1") or 0.0, r.get("F2") or 0.0)
    单 = {"a": "a1", "o": "o1", "e": "e1", "i": "i1", "u": "u1", "v": "v1"}
    距离 = {}
    for k1 in 单:
        for k2 in 单:
            if k1 < k2:
                x1, y1 = 取(单[k1])
                x2, y2 = 取(单[k2])
                距离["%s-%s" % (k1, k2)] = round(math.hypot(x1 - x2, y1 - y2), 1)
    报["元音空间_120Hz"] = 距离
    print("-" * 96)
    print("元音空间距离（F1,F2 平面，Hz）：", json.dumps(距离, ensure_ascii=False))
    最小 = min(距离.values()) if 距离 else 0.0
    print("最小两两距离 %.0fHz（<300Hz = 两个元音会被听成同一个）" % 最小)

    # ---------- 4) 整句逐音节 ----------
    报["整句逐音节"] = {}
    for 句 in ("你好，我是EmoCompanion。", "今天的风很轻，适合慢慢地说一会儿话。"):
        r = 短句元音检查(档, 句, 300)
        报["整句逐音节"][句] = r
        print("-" * 96)
        print("整句逐音节（%s）：" % 句)
        for d in r:
            print("   %-6s 韵=%-5s 浊%3d帧 目标F1/F2 %5.0f/%5.0f → 实测 %5.0f/%5.0f"
                  % (d["拼音"], d["韵"], d["浊帧"], d["目标F1"], d["目标F2"],
                     d["实测F1"], d["实测F2"]))

    # ---------- 5) 参考原声谐波斜率 ----------
    参考 = {}
    for 名 in ("EmoCompanion", "SpeakerA", "洛天依"):
        ps = sorted(glob.glob(os.path.join(MEET61, "素材", 名, "ref44_*.wav")))
        if not ps:
            continue
        fs, x = wavfile.read(ps[0])
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if np.max(np.abs(x)) > 1.2:
            x = x / 32768.0
        if fs != FS:
            g = math.gcd(int(fs), FS)
            x = resample_poly(x, FS // g, fs // g)
        # 取能量最高的一段 0.4s
        n = int(0.4 * FS)
        e = np.array([np.mean(x[i:i + n] ** 2) for i in range(0, len(x) - n, n // 2)])
        k = int(np.argmax(e))
        seg = x[k * (n // 2): k * (n // 2) + n]
        参考[名] = _谐波斜率(seg) or {}
    报["参考谐波"] = 参考
    print("-" * 96)
    print("参考原声谐波斜率：", json.dumps(参考, ensure_ascii=False))

    # ---------- 6) 判据 ----------
    突s = [r["突出度dB"][0] for r in B.values() if r.get("突出度dB") and r["突出度dB"][0]]
    判 = dict(
        元音空间最小距离Hz=round(最小, 1),
        判_空间可分=bool(最小 >= 300.0),
        F1突出度中位dB=round(float(np.median(突s)), 1) if 突s else None,
        判_F1成形=bool(突s and float(np.median(突s)) >= 6.0),
        峰位误差中位=round(float(np.median([r["最大峰位误差"] for r in B.values()
                                          if r.get("最大峰位误差")])), 3),
    )
    报["判据"] = 判
    print("=" * 96)
    print("判据：", json.dumps(判, ensure_ascii=False))
    out = os.path.join(出目录, "诊断_元音_v5.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)
    print("JSON:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
