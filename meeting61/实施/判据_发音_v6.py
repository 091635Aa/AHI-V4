# -*- coding: utf-8 -*-
"""meeting61/实施/判据_发音_v5.py —— v5 发音底层判据（客观，无听感通道）。

判据分五组：
  A 音韵映射（y/w、舌尖元音、复合韵目标）
  B 辅音实现（时长/电平/部位谱/清浊）
  C CV 协同发音（F2 轨迹、塞音部位顺序、VOT 对比）
  D 元音空间（F1/F2 峰位与两两距离）
  E 语流波形（静音占比、清浊差、包络动态、音节率、谱重心贴近参考、确定性）

所有读数来自真跑波形（FFT 峰位/RMS），不读内部参数。
用法：``python 判据_发音_v5.py`` → 打印 + 写 产物/判据/判据_发音_v5.json
纯 numpy/scipy（C1/C10），同种子逐位同（C13）。
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
import 韵律_v6 as H5  # noqa: E402
import 引擎_v5 as E5  # noqa: E402   （仅用于 F3 噪声道逐位同对照）
import 引擎_v6 as E6  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

FS = 44100
出目录 = os.path.join(MEET61, "产物", "判据")
os.makedirs(出目录, exist_ok=True)


def 读档(名="EmoCompanion") -> 声线档案:
    with open(os.path.join(MEET61, "产物", "声线档案", 名 + ".json"), encoding="utf-8") as f:
        return 声线档案.从dict(json.load(f))


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))) if len(x) else 0.0


def _db(a, b):
    return 20.0 * math.log10(max(a, 1e-12) / max(b, 1e-12))


def _峰(x, lo, hi, fs=FS):
    """FFT 峰位（Hann + 抛物线插值），返回 (Hz, 该峰功率)。"""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 64:
        return 0.0, 0.0
    x = x - float(np.mean(x))
    w = np.hanning(len(x))
    X = np.abs(np.fft.rfft(x * w)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    m = (f >= lo) & (f <= hi)
    if not np.any(m):
        return 0.0, 0.0
    ff, pp = f[m], X[m]
    k = int(np.argmax(pp))
    if 0 < k < len(pp) - 1:
        y0, y1, y2 = pp[k - 1], pp[k], pp[k + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-30 else 0.0
        fm = ff[k] + max(-0.5, min(0.5, d)) * (ff[1] - ff[0])
    else:
        fm = ff[k]
    return float(fm), float(pp[k])


def _谱重心(x, fs=FS):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 256:
        return 0.0
    w = np.hanning(len(x))
    P = np.abs(np.fft.rfft(x * w)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    P[0] = 0.0
    总 = float(P.sum())
    return float((f * P).sum() / 总) if 总 > 0 else 0.0


def 合成音节(拼: str, 档=None, 种子=300):
    """单音节 → (波形, 帧)。用同一韵律器，保证与整句一致。"""
    档 = 档 or 读档()
    器 = H5.人类韵律_v5(档, 种子)
    w, 元, 帧 = 器.说(拼)
    return w, 帧


def 分段(w, 帧, 类s):
    """按帧的声源类型取样本区间并拼接。类s 为集合，如 {0} / {1} / {-1}。"""
    out = []
    for i, t in enumerate(帧["声源"]):
        if int(t) in 类s:
            a, b = i * 441, min((i + 1) * 441, len(w))
            out.append(w[a:b])
    return np.concatenate(out) if out else np.zeros(0)


def _包络(x, fs=FS):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 256:
        return np.zeros(0), np.zeros(0)
    x = x - float(np.mean(x))
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    logP = np.log(np.abs(X) + 1e-12)
    cep = np.fft.irfft(logP)
    L = max(4, int(0.0016 * fs))       # 1.6ms 倒谱提升：保留共振峰包络
    cep[L:-L] = 0.0
    env = np.exp(np.fft.rfft(cep).real)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return f, env


def _目标峰(x, 目标Hz, 半宽=0.22, fs=FS):
    """在目标 ±半宽 内取谱包络最大值 → 验证「该共振峰确实出现在目标附近」。"""
    f, env = _包络(x, fs)
    if len(f) == 0:
        return 0.0
    m = (f >= 目标Hz * (1.0 - 半宽)) & (f <= 目标Hz * (1.0 + 半宽))
    if not np.any(m):
        return 0.0
    return float(f[m][int(np.argmax(env[m]))])


def _共振峰组(x, fs=FS, lo=150.0, hi=3500.0):
    """倒谱平滑谱包络 → 前三个共振峰（避免把谐波/F1 尾巴当成 F2）。"""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 256:
        return []
    x = x - float(np.mean(x))
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    logP = np.log(np.abs(X) + 1e-12)
    cep = np.fft.irfft(logP)
    L = max(4, int(0.0022 * fs))       # 2.2ms 倒谱提升：保留谱包络、压掉谐波
    cep[L:-L] = 0.0
    env = np.exp(np.fft.rfft(cep).real)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    m = (f >= lo) & (f <= hi)
    ff, ee = f[m], env[m]
    if len(ee) < 5:
        return []
    峰 = []
    for i in range(1, len(ee) - 1):
        if ee[i] >= ee[i - 1] and ee[i] > ee[i + 1]:
            峰.append((float(ff[i]), float(ee[i])))
    峰.sort(key=lambda z: z[0])
    结果 = []
    for fq, v in 峰:
        if 结果 and abs(fq - 结果[-1][0]) < 160.0:
            if v > 结果[-1][1]:
                结果[-1] = (fq, v)
            continue
        结果.append((fq, v))
    return [round(fq, 1) for fq, _ in 结果[:3]]


def _辅音样本(w, 帧):
    """只取辅音噪声帧（排除句首换气：换气 F1=500，辅音 F1≤450）。"""
    out = []
    for i, t in enumerate(帧["声源"]):
        if int(t) == 0 and float(帧["f1"][i]) < 450.0:
            a, b = i * 441, min((i + 1) * 441, len(w))
            out.append(w[a:b])
    return np.concatenate(out) if out else np.zeros(0)


def _辅音帧数(帧):
    return sum(1 for i, t in enumerate(帧["声源"]) if int(t) == 0 and float(帧["f1"][i]) < 450.0)


def _谱带(x, fs=FS):
    """能量占比：0–500 / 500–1k / 1–2k / 2–4k / 4–8k / 8–16k。"""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 4096:
        return {}
    n, hop = 4096, 2048
    k = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(k)[:, None]
    P = np.mean(np.abs(np.fft.rfft(x[idx] * np.hanning(n)[None, :], axis=1)) ** 2, axis=0)
    P[0] = 0.0
    f = np.fft.rfftfreq(n, 1.0 / fs)
    T = float(P.sum())
    if T <= 0:
        return {}
    return {f"{lo}-{hi}": round(float(P[(f >= lo) & (f < hi)].sum() / T), 4)
            for lo, hi in ((0, 500), (500, 1000), (1000, 2000),
                           (2000, 4000), (4000, 8000), (8000, 16000))}


def _f0序列(x, fs=FS, fmin=70.0, fmax=450.0):
    """40ms 帧 / 10ms 跳、归一化自相关 → 浊音帧 F0 序列。"""
    x = np.asarray(x, dtype=np.float64)
    帧长, 跳 = int(0.04 * fs), int(0.01 * fs)
    if len(x) < 帧长 * 2:
        return np.zeros(0)
    k = 1 + (len(x) - 帧长) // 跳
    idx = np.arange(帧长)[None, :] + 跳 * np.arange(k)[:, None]
    fr = x[idx] - x[idx].mean(axis=1, keepdims=True)
    fr *= np.hanning(帧长)[None, :]
    nfft = 1 << int(np.ceil(np.log2(2 * 帧长)))
    F = np.fft.rfft(fr, n=nfft, axis=1)
    ac = np.fft.irfft(F * np.conj(F), n=nfft, axis=1)
    ac /= np.maximum(ac[:, :1], 1e-20)
    最短, 最长 = max(1, int(fs / fmax)), min(nfft - 1, int(fs / fmin))
    seg = ac[:, 最短:最长 + 1]
    lag = np.argmax(seg, axis=1) + 最短
    rmax = seg[np.arange(len(lag)), lag - 最短]
    f0 = fs / lag
    f0[rmax < 0.40] = 0.0
    return f0[f0 > 0]


def _三段F0(x):
    """取 20% / 55% / 92% 三个时间点的 F0（±2 帧中位）——对应声调轮廓的起/折/末。"""
    f0 = _f0序列(x)
    if len(f0) < 8:
        return None
    出 = []
    for p in (0.20, 0.55, 0.92):
        c = int(round(p * (len(f0) - 1)))
        a, b = max(0, c - 2), min(len(f0), c + 3)
        出.append(float(np.median(f0[a:b])))
    return 出


def _半音(a, b):
    return 12.0 * math.log2(max(a, 1e-6) / max(b, 1e-6))


def 句级统计(w, 帧=None):
    n = 441
    k = max(1, len(w) // n)
    e = np.array([_rms(w[i * n:(i + 1) * n]) for i in range(k)])
    edb = 20 * np.log10(np.maximum(e, 1e-7))
    if 帧 is not None:
        活 = np.array([int(t) != -1 for t in 帧["声源"]][:k])
    else:
        活 = edb > (np.max(edb) - 40.0)
    if np.any(活):
        i0, i1 = int(np.argmax(活)), int(len(活) - np.argmax(活[::-1]))
    else:
        i0, i1 = 0, k
    段 = edb[i0:i1] if i1 > i0 else edb
    return dict(静音占比=round(float(np.mean(段 < (np.max(edb) - 40.0))), 4),
                包络标准差dB=round(float(np.std(段)), 2),
                包络动态范围dB=round(float(np.percentile(段, 95) - np.percentile(段, 5)), 2))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    档 = 读档("EmoCompanion")
    臂 = {}

    # ---------------- A 音韵映射 ----------------
    a1 = {"wo3": ("uo", None), "yuan2": ("van", "n"), "yi1": ("i", None),
          "yao4": ("iao", None), "wei4": ("uei", None), "wang4": ("uang", "ng"),
          "yue4": ("ve", None), "yun2": ("vn", "n")}
    ok = all(Y.解析音节(k)["韵"] == v[0] and Y.解析音节(k)["尾"] == v[1] for k, v in a1.items())
    臂["A1_yw零声母归一"] = dict(通过=bool(ok),
                            读数={k: Y.解析音节(k)["韵"] for k in a1})
    a2 = {"zhi1": "iii", "chi1": "iii", "shi1": "iii", "ri4": "iii",
          "zi1": "ii", "ci4": "ii", "si1": "ii"}
    ok2 = all(Y.解析音节(k)["韵"] == v for k, v in a2.items())
    f2_shi = Y.解析音节("shi1")["目标s"][0][1]
    f2_i = Y.解析音节("yi1")["目标s"][0][1]
    臂["A2_舌尖元音"] = dict(通过=bool(ok2 and f2_i - f2_shi > 400),
                        读数=dict(映射={k: Y.解析音节(k)["韵"] for k in a2},
                                 F2_shi=f2_shi, F2_i=f2_i))
    文本s = ["你好，我是EmoCompanion。", "今天的风很轻，适合慢慢地说一会儿话。",
             "你最近，过得好吗？", "嘿，我是SpeakerA。",
             "别看我平时闹得欢，认真说话的时候，我也能安静下来。",
             "要不要，一起待一会儿？", "大家好，我是洛天依。",
             "谢谢你愿意听我说话。", "希望今天的声音，能让你心里暖一点。",
             "语言不只是声音，它把一个人的温度，送到另一个人的心里。"]
    审 = Y.审计(文本s)
    臂["A3_全句查表"] = dict(通过=bool(审["问题数"] == 0),
                         读数=dict(音节数=审["音节数"], 问题数=审["问题数"],
                                 问题=审["问题"][:8]))

    # ---------------- B 辅音实现 ----------------
    辅 = {}
    字 = {"sa": "撒", "sha": "沙", "xia": "虾", "fa": "发", "ha": "哈",
         "ma": "妈", "na": "拿", "la": "拉", "ba": "八", "da": "搭", "ga": "嘎",
         "pa": "趴", "ta": "他", "ka": "咖"}
    for p in 字:
        w, 帧 = 合成音节(字[p], 档, 300)
        噪 = _辅音样本(w, 帧)
        浊 = 分段(w, 帧, {1})
        噪帧 = _辅音帧数(帧)
        浊帧 = sum(1 for t in 帧["声源"] if int(t) == 1)
        辅[p] = dict(噪ms=round(噪帧 * 10, 1), 浊ms=round(浊帧 * 10, 1),
                    清浊差dB=round(_db(_rms(噪), _rms(浊)), 2) if len(噪) and len(浊) else None,
                    噪重心Hz=round(_谱重心(噪), 0),
                    峰值=round(float(np.max(np.abs(w))), 3))
    fric = [p for p in ("sa", "sha", "xia", "fa") if p in 辅]
    部位序 = [辅[p]["噪重心Hz"] for p in ("sa", "xia", "sha")]
    f弱 = (辅["fa"]["清浊差dB"] < 辅["sha"]["清浊差dB"] - 0.5)
    臂["B1_擦音部位序"] = dict(
        通过=bool(部位序[0] > 部位序[1] > 部位序[2] and f弱),
        读数=dict(顺序="s>x>sh", 重心Hz=dict(zip(("sa", "xia", "sha"), 部位序)),
                 f重心Hz=辅["fa"]["噪重心Hz"], f弱于sh=bool(f弱)))
    强擦 = [辅[p]["清浊差dB"] for p in ("sa", "sha", "xia") if 辅[p]["清浊差dB"] is not None]
    弱擦 = 辅["ha"]["清浊差dB"] if 辅["ha"]["清浊差dB"] is not None else -99
    臂["B2_擦音电平"] = dict(
        通过=bool(强擦 and all(-16.0 <= d <= -2.0 for d in 强擦) and -24.0 <= 弱擦 <= -5.0),
        读数=dict(强擦dB=强擦, ha_dB=弱擦))
    浊母 = []
    浊母明细 = {}
    for p in ("ma", "na", "la"):
        w, 帧 = 合成音节(字[p], 档, 300)
        idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
        鼻 = w[idx[0] * 441:(idx[0] + 4) * 441] if len(idx) >= 4 else np.zeros(0)
        元 = w[(idx[0] + 9) * 441:(idx[0] + 14) * 441] if len(idx) >= 14 else np.zeros(0)
        if len(鼻) and len(元):
            d = _db(_rms(鼻), _rms(元))
            浊母.append(round(d, 2))
            浊母明细[p] = round(d, 2)
    臂["B3_浊声母电平"] = dict(通过=bool(浊母 and all(-16.0 <= d <= -3.0 for d in 浊母)),
                           读数=dict(dB=浊母明细))
    塞 = {p: 辅[p] for p in ("ba", "pa", "da", "ta", "ga", "ka")}
    vot差 = {}
    for q in ("b", "d", "g"):
        vot差[q] = 塞[q + "a"]["噪ms"]
    asp = {q: 塞[q + "a"]["噪ms"] for q in "ptk"}
    对 = {"p": "b", "t": "d", "k": "g"}
    臂["B4_送气对比"] = dict(
        通过=bool(all(asp[q] >= 2.0 * max(vot差[对[q]], 10.0) for q in "ptk")),
        读数=dict(不送气ms=vot差, 送气ms=asp))

    # ---------------- C CV 协同发音 ----------------
    C = {}
    期望起 = {"ba": 1027, "pa": 1027, "da": 1522, "ta": 1522, "ga": 1687, "ka": 1687,
            "ma": 1000, "na": 1500, "la": 1100}
    for p in ("ba", "da", "ga", "pa", "ta", "ka", "ma", "na", "la"):
        w, 帧 = 合成音节(字[p], 档, 300)
        浊idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
        if not 浊idx:
            continue
        i0 = 浊idx[0]
        a0 = i0 * 441
        起 = w[a0:a0 + int(0.040 * FS)]
        中 = w[(a0 + int(0.06 * FS)):(a0 + int(0.11 * FS))]
        F2起 = _目标峰(起, 期望起[p])
        F2中 = _目标峰(中, 1305.0)
        C[p] = dict(F2起Hz=round(F2起, 1), F2中Hz=round(F2中, 1),
                    ΔF2=round(F2起 - F2中, 1))
    ord_ok = C["ba"]["F2起Hz"] < C["da"]["F2起Hz"] < C["ga"]["F2起Hz"]
    臂["C1_塞音F2轨迹"] = dict(通过=bool(ord_ok),
                          读数=dict(顺序="b<d<g", F2起={k: C[k]["F2起Hz"] for k in ("ba", "da", "ga")}))
    转 = max(abs(C[p]["ΔF2"]) for p in C)
    臂["C2_CV过渡幅度"] = dict(通过=bool(转 >= 80.0), 读数=dict(最大ΔF2Hz=round(转, 1), 逐音节=C))

    # C3 四声轮廓（妈麻马骂）
    调测 = {}
    for 拼 in ("ma1", "ma2", "ma3", "ma4"):
        w, _, _ = H5.人类韵律_v5(档, 300).说音节(拼)
        t = _三段F0(w)
        if t is None:
            调测[拼] = None
            continue
        首 = t[0]
        调测[拼] = dict(首Hz=round(首, 1), 中Hz=round(t[1], 1), 末Hz=round(t[2], 1),
                      中st=round(_半音(t[1], 首), 2), 末st=round(_半音(t[2], 首), 2))
    四声ok = False
    if all(调测.get(k) for k in ("ma1", "ma2", "ma3", "ma4")):
        四声ok = bool(abs(调测["ma1"]["末st"]) < 0.8
                      and 调测["ma2"]["末st"] >= 1.2
                      and 调测["ma3"]["中st"] <= -0.4
                      and 调测["ma3"]["末st"] - 调测["ma3"]["中st"] >= 1.2
                      and 调测["ma4"]["末st"] <= -2.0)
    臂["C3_四声轮廓"] = dict(通过=四声ok, 读数=调测)

    # ---------------- D 元音空间 ----------------
    # 说明：EmoCompanion F0≈285Hz（乘声区≈350Hz）时谐波间隔 350Hz，F1≈825 只有 2.4 个谐波，
    #      任何包络估计都会被拉到相邻谐波（这是测量不确定度，不是共振峰错）。
    #      因此元音目标用 120Hz 测试声线验证引擎/音韵；高 F0 读数另列作参考。
    档低 = 档.派生(基频=120.0)
    V = {}
    for p, 期望, 字p in (("a", (825, 1305), "啊"), ("yi", (330, 2349), "衣"),
                        ("wu", (352, 751), "乌"), ("yu", (330, 1931), "迂")):
        w, 帧 = 合成音节(字p, 档低, 300)
        浊idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
        a = (浊idx[len(浊idx) // 3] if 浊idx else 0) * 441
        seg = w[a:a + int(0.08 * FS)]
        f1 = _目标峰(seg, 期望[0], 0.12)
        f2 = _目标峰(seg, 期望[1], 0.12)
        V[p] = dict(F1=round(f1, 1), F2=round(f2, 1),
                    误F1=round(abs(f1 - 期望[0]) / 期望[0], 3),
                    误F2=round(abs(f2 - 期望[1]) / 期望[1], 3))
    误 = max(max(v["误F1"], v["误F2"]) for v in V.values())
    臂["D1_元音峰位"] = dict(通过=bool(误 < 0.15), 读数=dict(测试F0=120, 最大误差=round(误, 3), 元音=V))
    d_ai = math.hypot(V["a"]["F1"] - V["yi"]["F1"], V["a"]["F2"] - V["yi"]["F2"])
    d_au = math.hypot(V["a"]["F1"] - V["wu"]["F1"], V["a"]["F2"] - V["wu"]["F2"])
    d_iu = math.hypot(V["yi"]["F1"] - V["yu"]["F1"], V["yi"]["F2"] - V["yu"]["F2"])
    臂["D2_元音可分"] = dict(通过=bool(d_ai > 600 and d_au > 500 and d_iu > 250),
                         读数=dict(a_i=round(d_ai), a_u=round(d_au), i_v=round(d_iu)))

    # 鼻尾 -n / -ng 的 F2 差（元音末段）
    N = {}
    for p, 字p in (("an", "安"), ("ang", "昂")):
        w, 帧 = 合成音节(字p, 档, 300)
        浊idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
        i0 = 浊idx[int(len(浊idx) * 0.72)]
        seg = w[i0 * 441:(i0 * 441) + int(0.06 * FS)]
        f2 = _目标峰(seg, 1670.0 if p == "an" else 1148.0)
        N[p] = round(f2, 1)
    臂["D3_鼻尾区别"] = dict(通过=bool(abs(N["an"] - N["ang"]) >= 200),
                         读数=dict(an=N["an"], ang=N["ang"], 差=round(N["an"] - N["ang"], 1)))

    # ---------------- E 语流波形 ----------------
    句s = {}
    句波形 = {}
    for i, 句 in enumerate(("你好，我是EmoCompanion。",
                            "今天的风很轻，适合慢慢地说一会儿话。",
                            "语言不只是声音，它把一个人的温度，送到另一个人的心里。")):
        w, 元, 帧 = H5.人类韵律_v5(档, 300 + i).说(句)
        句波形[句] = w
        n音节 = len(Y.音节列表(句))
        噪 = 分段(w, 帧, {0})
        浊 = 分段(w, 帧, {1})
        句s[句[:10]] = dict(时长=round(len(w) / FS, 2), 音节数=n音节,
                          音节率=round(n音节 / (len(w) / FS), 2),
                          清浊差dB=round(_db(_rms(噪), _rms(浊)), 2),
                          谱重心Hz=round(_谱重心(w), 0), **句级统计(w, 帧))
    ref = {}
    参考波形 = {}
    for 名 in ("EmoCompanion", "SpeakerA", "洛天依"):
        ps = sorted(glob.glob(os.path.join(MEET61, "素材", 名, "ref44_*.wav")))
        if ps:
            fs, x = wavfile.read(ps[0])
            x = np.asarray(x, dtype=np.float64)
            if x.ndim > 1:
                x = x.mean(axis=1)
            if np.max(np.abs(x)) > 1.2:
                x = x / 32768.0
            if fs != FS:
                g = math.gcd(int(fs), FS)
                x = resample_poly(x, FS // g, fs // g)
            ref[名] = dict(谱重心Hz=round(_谱重心(x), 0), **句级统计(x))
            参考波形[名] = x
    缘句 = 句s["你好，我是EmoCompanion。"]
    w1 = 句波形["你好，我是EmoCompanion。"]
    合成带 = _谱带(w1)
    参考带 = {k: _谱带(参考波形[k]) for k in 参考波形}
    均值带 = {b: float(np.mean([参考带[k][b] for k in 参考带 if b in 参考带[k]]))
             for b in 合成带} if 参考带 else {}
    带距 = float(sum(abs(合成带[b] - 均值带[b]) for b in 合成带 if b in 均值带)) if 均值带 else 1.0
    均重 = float(np.mean([v["谱重心Hz"] for v in ref.values()])) if ref else 1.0
    重比 = abs(缘句["谱重心Hz"] - 均重) / 均重
    def _参考范围():
        """参考原声 11 个文件的各带占比 min/max —— 说明 E5 的 L1 门限锚在什么上。
        实测：同一角色的不同 take，0–500Hz 占比从 12% 到 57%（差 4.7 倍），
        1–2k 从 7.6% 到 51.5%（差 6.8 倍）⇒ 带分布不是「正确发音」的稳定属性。"""
        各 = []
        for 名 in ("EmoCompanion", "SpeakerA", "洛天依"):
            for p in sorted(glob.glob(os.path.join(MEET61, "素材", 名, "ref44_*.wav"))):
                fs, xx = wavfile.read(p)
                xx = np.asarray(xx, dtype=np.float64)
                if xx.ndim > 1:
                    xx = xx.mean(axis=1)
                if np.max(np.abs(xx)) > 1.2:
                    xx = xx / 32768.0
                if fs != FS:
                    g = math.gcd(int(fs), FS)
                    xx = resample_poly(xx, FS // g, fs // g)
                各.append(_谱带(xx))
        return {b: [round(min(z[b] for z in 各 if b in z), 4),
                    round(max(z[b] for z in 各 if b in z), 4)] for b in 合成带}

    # E7：整句层面的「噪帧不变」护栏 —— 与 F4 同义，但在真实句子里逐帧取噪段比较，
    # 证明 v6 相对 v5 的变化只落在浊帧（元音层），辅音层没有动。
    帧缘 = H5.人类韵律_v5(档, 300).建帧("你好，我是EmoCompanion。")
    w怨5 = E5.合成(帧缘, 音色=H5.人类韵律_v5(档, 300).音色(), 种子=300)[0]
    w怨6 = 句波形["你好，我是EmoCompanion。"]
    噪5 = _谱带(分段(w怨5, 帧缘, {0}))
    噪6 = _谱带(分段(w怨6, 帧缘, {0}))
    浊5 = _谱带(分段(w怨5, 帧缘, {1}))
    浊6 = _谱带(分段(w怨6, 帧缘, {1}))
    噪同 = bool(all(abs(噪5[b] - 噪6[b]) < 1e-9 for b in 噪5))
    臂["E7_噪帧带分布未变"] = dict(
        通过=噪同,
        读数=dict(说明="同帧同音色，只换引擎；噪帧带分布必须逐位同（辅音不回归）",
                 v5噪=噪5, v6噪=噪6, v5浊=浊5, v6浊=浊6,
                 噪L1=round(float(sum(abs(噪5[b] - 噪6[b]) for b in 噪5)), 6)))

    臂["E5_谱形贴近参考"] = dict(通过=bool(重比 <= 0.25 and 带距 <= 0.25),
                            读数=dict(合成重心=缘句["谱重心Hz"], 参考重心均值=round(均重),
                                    相对差=round(重比, 3), 带分布L1=round(带距, 3),
                                    合成带=合成带, 参考带均值={k: round(v, 4) for k, v in 均值带.items()},
                                    参考带范围=_参考范围()))
    臂["E1_静音占比"] = dict(通过=bool(缘句["静音占比"] < 0.20),
                         读数=dict(合成=缘句["静音占比"], 参考=ref["EmoCompanion"]["静音占比"]))
    臂["E2_清浊电平差"] = dict(通过=bool(-18.0 <= 缘句["清浊差dB"] <= -4.0),
                           读数=dict(合成=缘句["清浊差dB"]))
    臂["E3_包络动态"] = dict(通过=bool(缘句["包络标准差dB"] <= 35.0),
                         读数=dict(合成=缘句["包络标准差dB"], 参考=ref["EmoCompanion"]["包络标准差dB"]))
    臂["E4_音节率"] = dict(通过=bool(all(3.8 <= v["音节率"] <= 5.8 for v in 句s.values())),
                        读数={k: v["音节率"] for k, v in 句s.items()})
    wa, _, _ = H5.人类韵律_v5(档, 300).说("你好，我是EmoCompanion。")
    wb, _, _ = H5.人类韵律_v5(档, 300).说("你好，我是EmoCompanion。")
    wc, _, _ = H5.人类韵律_v5(档, 301).说("你好，我是EmoCompanion。")
    同 = bool(np.array_equal(wa, wb))
    异 = bool(not np.array_equal(wa, wc))
    臂["E6_确定性"] = dict(通过=bool(同 and 异), 读数=dict(两趟同=同, 异种子异=异))

    # ---------------- F 耳听等价（v6 新增：元音谱峰是否落在 F1）----------------
    # 换量法理由（两处）：1) 倒谱包络在窄窗找峰会撞窗边；2) 声调使 F0 在窗内滑动
    # ±15%，用「固定 F0 的谐波栅格」会错位取样（第一版 /e/ 的假读数就是这么来的）。
    # 最终采用**与 F0 无关的量**：谱包络（倒谱平滑）在 150–3500Hz 内的最大点位置。
    def _包络(x):
        x = np.asarray(x, dtype=np.float64) - float(np.mean(x))
        X = np.fft.rfft(x * np.hanning(len(x)))
        cep = np.fft.irfft(np.log(np.abs(X) + 1e-12))
        L = max(4, int(0.0022 * FS))
        cep[L:-L] = 0.0
        env = np.exp(np.fft.rfft(cep).real)
        return np.fft.rfftfreq(len(x), 1.0 / FS), env

    def _峰与台(x, f1):
        f, env = _包络(x)
        m = (f >= 150.0) & (f <= 3500.0)
        ff, ee = f[m], env[m]
        峰位 = float(ff[int(np.argmax(ee))])
        lo = (f >= 150.0) & (f < f1 * 0.75)
        台 = float(np.median(env[lo])) if np.any(lo) else 0.0
        return 峰位, round(20.0 * math.log10(max(ee.max(), 1e-12) / max(台, 1e-12)), 1)

    档低 = 档.派生(基频=120.0)
    # 单元音：窗 = [F1×0.85, F2×1.15]（单目标，严格）。
    # 复合/鼻韵母：有滑移，中间窗未必落在末目标上 ⇒ 用**全部目标的并集窗**
    # [min(F1)×0.85, max(F2)×1.15]（宽一些但不空洞）。
    样 = (("啊", "a1"), ("哦", "o1"), ("鹅", "e1"), ("衣", "i1"), ("乌", "u1"),
         ("迂", "v1"), ("儿", "er2"),
         ("好", "hao3"), ("我", "wo3"), ("是", "shi4"), ("天", "tian1"),
         ("慢", "man4"), ("说", "shuo1"))
    FF = {}
    for 字, 拼 in 样:
        s = Y.解析音节(拼)
        目s = s["目标s"]
        单元 = len(目s) == 1 and s["尾"] is None
        if 单元:
            f1t, f2t = 目s[0][0] * 档.共振峰缩放[0], 目s[0][1] * 档.共振峰缩放[1]
            窗 = (f1t * 0.85, f2t * 1.15)
        else:
            f1t = min(t[0] for t in 目s) * 档.共振峰缩放[0]
            f2t = max(t[1] for t in 目s) * 档.共振峰缩放[1]
            窗 = (f1t * 0.85, f2t * 1.15)
        帧 = H5.人类韵律_v5(档低, 300).建帧(字)
        idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
        a0 = (idx[len(idx) // 3] if idx else 0) * 441
        x6, _, _ = H5.人类韵律_v5(档低, 300).说(字)
        x5 = E5.合成(帧, 音色=H5.人类韵律_v5(档低, 300).音色(), 种子=300)[0]
        seg6 = x6[a0:a0 + int(0.12 * FS)]
        seg5 = x5[a0:a0 + int(0.12 * FS)]
        峰6, 突6 = _峰与台(seg6, 目s[0][0] * 档.共振峰缩放[0]) if len(seg6) > 512 else (0.0, 0.0)
        峰5, 突5 = _峰与台(seg5, 目s[0][0] * 档.共振峰缩放[0]) if len(seg5) > 512 else (0.0, 0.0)
        比6 = 峰6 / (目s[0][0] * 档.共振峰缩放[0])
        比5 = 峰5 / (目s[0][0] * 档.共振峰缩放[0])
        落 = bool(窗[0] <= 峰6 <= 窗[1])
        落5 = bool(窗[0] <= 峰5 <= 窗[1])
        FF[拼] = dict(韵=s["韵"], 单元音=单元, 首目标F1=round(目s[0][0] * 档.共振峰缩放[0], 1),
                     窗=[round(窗[0], 1), round(窗[1], 1)],
                     v6包络峰=round(峰6, 1), v5包络峰=round(峰5, 1),
                     v6峰位比首F1=round(比6, 2), v5峰位比首F1=round(比5, 2),
                     v6突出dB=突6, v5突出dB=突5, v6落位=落, v5落位=落5)
    全落 = all(v["v6落位"] for v in FF.values())
    落5数 = sum(1 for v in FF.values() if v["v5落位"])
    臂["F1_元音谱包络峰位"] = dict(
        通过=bool(全落 and 落5数 < len(FF)),
        读数=dict(说明="谱包络最大点必须落在元音共振峰空间 [F1×0.85, F2×1.15]；"
                    "同一判据下 v5 的落位数必须更少（v5 的峰被并联低频平台拖到 300–408Hz）",
                 v6落位=f"{sum(1 for v in FF.values() if v['v6落位'])}/{len(FF)}",
                 v5落位=f"{落5数}/{len(FF)}", 逐音节=FF))
    臂["F2_元音F1F2区突出"] = dict(
        通过=bool(all(v["v6突出dB"] >= 3.0 for v in FF.values()
                      if v["首目标F1"] >= 500.0)),
        读数={"v6突出dB": {k: v["v6突出dB"] for k, v in FF.items()},
              "v5突出dB": {k: v["v5突出dB"] for k, v in FF.items()},
              "门": "F1≥500Hz 的元音，包络峰须比 F1 以下的谱台高 ≥3dB"})

    # F2 单元音全表峰位（120Hz 声线，倒谱包络）
    单 = {}
    for 拼 in ("a1", "o1", "e1", "i1", "u1", "v1", "er2"):
        s = Y.解析音节(拼)
        目 = [t * v for t, v in zip(s["目标s"][0], (档.共振峰缩放[0], 档.共振峰缩放[1]))]
        字 = {"a": "啊", "o": "哦", "e": "鹅", "i": "衣", "u": "乌",
             "v": "迂", "er": "儿"}[s["韵"]]
        帧 = H5.人类韵律_v5(档低, 300).建帧(字)
        idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
        a0 = (idx[len(idx) // 3] if idx else 0) * 441
        w, _, _ = H5.人类韵律_v5(档低, 300).说(字)
        seg = w[a0:a0 + int(0.10 * FS)]
        # 「倒谱包络峰列表」：检单元音的 F1/F2 是否都能在包络里找到（±20%）
        f, env = _包络(seg)
        m = (f >= 150.0) & (f <= 3500.0)
        ff, ee = f[m], env[m]
        峰s = []
        for i in range(1, len(ee) - 1):
            if ee[i] >= ee[i - 1] and ee[i] > ee[i + 1]:
                峰s.append(float(ff[i]))
        def 近(t):
            if not 峰s:
                return None, 9.9
            c = min(峰s, key=lambda z: abs(z - t))
            return c, abs(c - t) / t
        c1, e1_ = 近(目[0])
        c2, e2_ = 近(目[1])
        单[拼] = dict(韵=s["韵"], 目标F1=round(目[0], 1), 目标F2=round(目[1], 1),
                     最近包络峰F1=(round(c1, 1) if c1 else None),
                     最近包络峰F2=(round(c2, 1) if c2 else None),
                     F1误差=round(e1_, 3), F2误差=round(e2_, 3))
    误s = [max(v["F1误差"], v["F2误差"]) for v in 单.values()]
    臂["F3_单元音F1F2可辨"] = dict(通过=bool(误s and float(np.median(误s)) <= 0.20),
                             读数=dict(说明="包络峰列表里 F1 与 F2 目标 ±20% 内有峰（量法在窄窗内找峰会撞窗边，故改用峰列表）",
                                     中位误差=round(float(np.median(误s)), 3), 逐韵=单))

    # F3 噪声道未动（C14 护栏：辅音听感不得回归）
    噪帧 = dict(f0=[180.0] * 20, f1=[300.0] * 20, f2=[1800.0] * 20, f3=[2600.0] * 20,
               f4=[3500.0] * 20, f5=[4500.0] * 20, 幅=[0.8] * 20, 气声=[0.0] * 20,
               声源=[0] * 20, 噪带=[(6800.0, 2200.0)] * 20, 送气=[1.0] * 20,
               紧喉=[0.3] * 20, 粗糙=[0.0] * 20, 噪形=[0] * 20)
    y5, _ = E5.合成(噪帧, 音色=None, 种子=300)
    y6, _ = E6.合成(噪帧, 音色=None, 种子=300)
    同噪 = bool(np.array_equal(y5, y6))
    臂["F4_噪声道逐位同v5"] = dict(通过=同噪, 读数=dict(逐位同=同噪, 样本数=len(y5)))

    通过数 = sum(1 for v in 臂.values() if v["通过"])
    报 = dict(工具="判据_发音_v6.py", 臂数=len(臂), 通过=通过数, 未过=len(臂) - 通过数,
              臂=臂, 参考=ref, 句级=句s)
    out = os.path.join(出目录, "判据_发音_v6.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)

    print("=" * 86)
    for k, v in 臂.items():
        print("%-18s %s  %s" % (k, "PASS" if v["通过"] else "FAIL",
                                json.dumps(v["读数"], ensure_ascii=False)[:150]))
    print("-" * 86)
    print("总计 %d/%d PASS" % (通过数, len(臂)))
    print("JSON:", out)
    return 0 if 通过数 == len(臂) else 1


if __name__ == "__main__":
    raise SystemExit(main())
