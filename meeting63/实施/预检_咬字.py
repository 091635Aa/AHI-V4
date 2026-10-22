# -*- coding: utf-8 -*-
"""meeting63/实施/预检_咬字.py —— 「咬字不清」客观预检（纯内生声学量，无外挂模型）。

目的：把「声音模糊、咬字不清」拆成**可测的辅音/音节边界缺陷**，逐项给读数与判据。
只读 import meeting62/61/58 既有件；只写 meeting63/产物/。C14：不改既有任何件。

量法（全部来自引擎的**帧标签**声源∈{-1 静音, 0 噪声, 1 浊} 与 噪形∈{0 噪带成形, 1 共振峰成形}）：
  塞音：塞闭(静音)ms → 爆破(噪形0)ms/SPL → VOT(噪形1)ms/SPL → 元音
  擦音：擦段(噪形0/1) ms、噪谱峰 Hz、噪谱重心 Hz、噪电平 vs 后随元音
  塞擦：塞闭 → 短爆破 → 擦段
  鼻/边：浊声母段 ms 与电平
  音节：元音时长 ms、CV 过渡 F2 起-中差 Hz
输出：meeting63/产物/预检_咬字.json（两趟逐位同 = C13 自检）
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M63 = os.path.dirname(HERE)
AHI4 = os.path.dirname(M63)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AHI4, "meeting62", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting61", "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))

import 韵律_v7 as H7  # noqa: E402
import 引擎_v7 as E7  # noqa: E402
import 音韵_v5 as Y  # noqa: E402
import 声线覆盖层_v7 as 覆  # noqa: E402

FS = 44100
每帧 = 441
出目录 = os.path.join(M63, "产物")
os.makedirs(出目录, exist_ok=True)


def _rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def _db(a, b):
    return 20.0 * math.log10(max(abs(a), 1e-12) / max(abs(b), 1e-12))


def _谱(x, fs=FS):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 256:
        return dict(峰=0.0, 重心=0.0, 带宽=0.0)
    x = x - float(np.mean(x))
    P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    P[0] = 0.0
    T = float(P.sum())
    if T <= 0:
        return dict(峰=0.0, 重心=0.0, 带宽=0.0)
    lo, hi = 300.0, 16000.0
    m = (f >= lo) & (f <= hi)
    ff, pp = f[m], P[m]
    k = int(np.argmax(pp))
    峰 = float(ff[k])
    重心 = float((f * P).sum() / T)
    带 = float(math.sqrt(((f - 重心) ** 2 * P).sum() / T))
    return dict(峰=round(峰, 1), 重心=round(重心, 1), 带宽=round(带, 1))


def _包络(x, lif=0.0022, fs=FS):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 256:
        return np.zeros(0), np.zeros(0)
    x = x - float(np.mean(x))
    X = np.fft.rfft(x * np.hanning(len(x)))
    logP = np.log(np.abs(X) + 1e-12)
    cep = np.fft.irfft(logP)
    L = max(4, int(lif * fs))
    cep[L:-L] = 0.0
    return np.fft.rfftfreq(len(x), 1.0 / fs), np.exp(np.fft.rfft(cep).real)


def _峰近(x, 目标Hz, 半宽=0.20):
    f, env = _包络(x)
    if f.size == 0:
        return 0.0
    m = (f >= 目标Hz * (1 - 半宽)) & (f <= 目标Hz * (1 + 半宽))
    if not np.any(m):
        return 0.0
    return float(f[m][int(np.argmax(env[m]))])


def _帧段(帧, i0, i1):
    """帧区间 [i0,i1) 的样本（按帧号×441 切，末样本截齐）。"""
    a, b = i0 * 每帧, min(i1 * 每帧, len(帧["_w"]))
    return 帧["_w"][a:b]


def _标签序列(帧):
    s = np.array(帧["声源"], dtype=int)
    nz = np.array(帧["噪形"], dtype=int)
    return s, nz


def 分析音节(档, 拼, 字, 种子=300):
    器 = H7.人类韵律_v5(档, 种子)
    w, 元, 帧 = 器.说音节(拼)
    帧["_w"] = w
    s, nz = _标签序列(帧)
    幅 = np.array(帧["幅"], dtype=np.float64)
    A = float(getattr(器, "幅度", 0.55))
    N = len(s)

    # ① 跳过句首「换气」段：**噪声帧**(typ==0) 且 幅 很小；静音(typ==-1)不跳
    起 = 0
    while 起 < N and int(s[起]) == 0 and 幅[起] <= A * 0.15:
        起 += 1
    if 起 >= N:
        起 = 0

    # ② 元音起点：浊帧段里第一个「幅 达到满幅」的帧（浊声母 murmur 幅明显更低）
    元起 = N
    for i in range(起, N):
        if int(s[i]) == 1 and 幅[i] >= A * 0.95:
            元起 = i
            break
    if 元起 >= N:                      # 兜底：首个浊帧
        idx = np.where(s[起:] == 1)[0]
        元起 = 起 + int(idx[0]) if idx.size else N

    # ③ 辅音分段（在 [起, 元起) 内按 声源/噪形 切）
    段s = []
    i = 起
    while i < 元起:
        typ, m = int(s[i]), int(nz[i])
        j = i
        while j < 元起 and int(s[j]) == typ and int(nz[j]) == m:
            j += 1
        段s.append(dict(typ=typ, 噪形=m, 帧数=j - i, 起=i, 止=j))
        i = j
    for d in 段s:
        seg = _帧段(帧, d["起"], d["止"])
        d["ms"] = round(d["帧数"] * 10.0, 1)
        d["rms"] = round(_rms(seg), 6)
        d["谱"] = _谱(seg)
        d["噪带"] = 帧["噪带"][d["起"]]
        d["送气"] = round(float(帧["送气"][d["起"]]), 3)

    # 元音段（元起 + 3 帧 ≈ 30ms 后，取 10 帧）
    元s = _帧段(帧, min(元起 + 3, N), min(元起 + 13, N))
    元rms = _rms(元s)
    辅s = _帧段(帧, 起, 元起)
    辅rms = _rms(辅s)
    清浊差 = round(_db(辅rms, 元rms), 2) if 辅rms > 0 and 元rms > 0 else None

    # ④ CV 过渡：F2 起(元起 起 4 帧) 与 中(+30ms)
    曲 = Y.解析音节(拼)
    目 = 曲["目标s"]
    F2目标 = float(目[0][1]) if 曲["韵"] else 1250.0
    起段 = _帧段(帧, 元起, min(元起 + 4, N))
    中段 = _帧段(帧, min(元起 + 3, N), min(元起 + 8, N))
    F2起 = _峰近(起段, F2目标) if len(起段) > 200 else 0.0
    F2中 = _峰近(中段, F2目标) if len(中段) > 200 else 0.0

    浊声母帧 = 元起 - (起 + sum(d["帧数"] for d in 段s if d["typ"] != 1))
    return dict(拼=拼, 字=字, 方式=曲["方式"], 声=曲["声"], 韵=曲["韵"],
                呼吸ms=round(起 * 10.0, 1),
                辅段=段s, 辅ms=round(sum(d["ms"] for d in 段s), 1),
                浊声母ms=round(max(浊声母帧, 0) * 10.0, 1),
                辅rms=round(辅rms, 6), 元rms=round(元rms, 6), 清浊差dB=清浊差,
                元音ms=round((N - 元起) * 10.0, 1),
                F2起Hz=round(F2起, 1), F2中Hz=round(F2中, 1),
                ΔF2=round(F2起 - F2中, 1),
                峰值=round(float(np.max(np.abs(w))), 4))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    档 = 覆.取("EmoCompanion")
    字表 = {
        # 塞音（不送气 / 送气）
        "ba1": "八", "pa1": "趴", "da1": "搭", "ta1": "他", "ga1": "嘎", "ka1": "咖",
        # 擦音
        "fa1": "发", "ha1": "哈", "sa1": "撒", "sha1": "沙", "xia1": "虾",
        # 塞擦
        "za1": "匝", "ca1": "擦", "zha1": "渣", "cha1": "叉", "jia1": "家", "qia1": "掐",
        # 浊声母
        "ma1": "妈", "na1": "拿", "la1": "拉", "ra1": "呥",
    }
    读 = {}
    for 拼, 字 in 字表.items():
        try:
            读[拼] = 分析音节(档, 拼, 字)
        except Exception as e:  # noqa: BLE001
            读[拼] = dict(错误="%s: %s" % (type(e).__name__, e))

    # —— 汇总：送气/不送气 VOT 对照 ——
    def vot(拼):
        for d in 读[拼]["辅段"]:
            if d["typ"] == 0 and d["噪形"] == 1:
                return d["ms"]
        return None

    def 爆破(拼):
        for d in 读[拼]["辅段"]:
            if d["typ"] == 0 and d["噪形"] == 0:
                return d
        return None

    def 闭(拼):
        for d in 读[拼]["辅段"]:
            if d["typ"] == -1:
                return d["ms"]
        return 0.0

    擦拼 = {"f": "fa1", "h": "ha1", "s": "sa1", "sh": "sha1", "x": "xia1"}
    汇 = lambda q: 读[(擦拼.get(q) or (q + "a1"))]
    汇总 = dict(
        送气VOT_ms={q: vot(q + "a1") for q in "ptk"},
        不送气VOT_ms={q: vot(q + "a1") for q in "bdg"},
        塞音闭_ms={q: 闭(q + "a1") for q in ("b", "p", "d", "t", "g", "k")},
        塞擦闭_ms={q: 闭(k) for q, k in (("z", "za1"), ("c", "ca1"), ("zh", "zha1"),
                                        ("ch", "cha1"), ("j", "jia1"), ("q", "qia1"))},
        擦音时长_ms={q: 汇(q)["辅ms"] for q in ("f", "h", "s", "sh", "x")},
        擦音谱峰Hz={q: (爆破(擦拼[q]) or dict(谱=dict(峰=0.0)))["谱"]["峰"]
                  for q in ("f", "h", "s", "sh", "x")},
        塞音清浊差dB={q: 读[q + "a1"]["清浊差dB"] for q in ("b", "p", "d", "t", "g", "k")},
        擦音清浊差dB={q: 汇(q)["清浊差dB"] for q in ("f", "h", "s", "sh", "x")},
        浊声母时长_ms={q: 读[q + "a1"]["浊声母ms"] for q in ("m", "n", "l", "r")},
        元音时长_ms={q: 读[q + "a1"]["元音ms"] for q in ("b", "p", "s", "sh")},
        CV过渡ΔF2={q: 读[q + "a1"]["ΔF2"] for q in ("b", "d", "g", "p", "t", "k")},
    )
    报 = dict(工具="预检_咬字.py", 版本="v7-44k(EmoCompanion覆盖声线)", 逐音节=读, 汇总=汇总)
    out = os.path.join(出目录, "预检_咬字.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)

    print("=" * 92)
    for 拼, v in 读.items():
        if "错误" in v:
            print("%-6s ERR %s" % (拼, v["错误"]))
            continue
        print("%-6s 方式=%-4s 辅%5.1fms 元%5.1fms 清浊%6s | 段=%s"
              % (拼, v["方式"], v["辅ms"], v["元音ms"], v["清浊差dB"],
                 " ".join("%d:%d*%s" % (d["typ"], d["噪形"], d["ms"]) for d in v["辅段"])))
    print("-" * 92)
    print("送气VOT p/t/k =", 汇总["送气VOT_ms"])
    print("不送气VOT b/d/g =", 汇总["不送气VOT_ms"])
    print("擦音时长 f/h/s/sh/x =", 汇总["擦音时长_ms"])
    print("擦音谱峰 =", 汇总["擦音谱峰Hz"])
    print("塞音清浊差 =", 汇总["塞音清浊差dB"])
    print("擦音清浊差 =", 汇总["擦音清浊差dB"])
    print("浊声母时长 m/n/l/r =", 汇总["浊声母时长_ms"])
    print("CV ΔF2 b/d/g =", 汇总["CV过渡ΔF2"])
    print("JSON:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
