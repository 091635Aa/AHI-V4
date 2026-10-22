# -*- coding: utf-8 -*-
"""meeting62/实施/判据_发音_v7.py —— v7 发音底层判据（可证伪版）。

纯新增件（C14）：只读 import meeting61/58/62 既有件；只写 `meeting62/产物/判据/`。
继承 `判据_发音_v6.py` 的 **24 个既有臂（保号保定义，量法/阈值一字不改）**，但**在 v7 产出上重跑**；
并**新增**（按 Task 4 裁决）：

  新增臂
    · N1_元音身份可分(判别版) —— 修红队「窗宽买通」：每元音按 F1/F2 逐共振峰 ±15% 窄窗取包络峰，
      用 (F1est,F2est) 最近邻分类，交叉落位 ≤1。
    · N2_音色身份可分        —— 三角色经 `声线覆盖层_v7`（共振峰缩放 3 轴 + 谱倾斜，由 meeting61
      参考录音统计派生）同句合成，按 (**F0 半音 ≥1.0 或 浊谱带 L1 ≥0.20**，阈值口径未改）判两两可分，须 3/3。
    · N3_唱歌音准            —— 谱驱动（谱管线_v7）逐音 f0 vs 谱音符，误差 ≤25 音分。
    · N4_唱歌节奏            —— 逐音 f0 命中时长 vs 谱时值，误差 ≤0.30；整句时长误差 ≤0.15。
    · N5_长音F0颤音          —— 去颤中心音准 ≤25 音分、去趋势幅度 ∈ [0.4,1.6] 半音（修①可证伪）。

  must-fail（必过；不满足则判据/读数作废）
    · M1_白噪sham   —— 白噪喂进元音峰位管线必须 FAIL。
    · M2_静音sham   —— 静音喂进元音峰位管线必须 FAIL。
    · M3_窗交叉sham —— 复现红队「v6 legacy 宽窗 7 单元音交叉落位 7/7」，并断言判别版修复后 ≤1。

  准入门（任何新 PASS 的前提）
    · P0_外部真人正对照 —— 同一套元音峰位量法跑 `meeting61/素材/*/ref44_*.wav`（共 11 文件），
      真人必须 PASS（峰位中位落合理 F1 域 + F1 区突出 ≥3dB）。

确定性（C13）：同种子两趟逐位同、异种子不同（E6 臂 + 全局 sha256 自检）。
纯 numpy/scipy（C1/C9/C10）；同机 CPU-only。
用法：``python 判据_发音_v7.py`` → 打印 + 写 产物/判据/判据_发音_v7.json
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
M62 = os.path.dirname(HERE)
AHI4 = os.path.dirname(M62)
MEET61 = os.path.join(AHI4, "meeting61")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(MEET61, "实施"))
sys.path.insert(0, os.path.join(AHI4, "meeting58", "实施"))

import 音韵_v5 as Y  # noqa: E402
import 韵律_v7 as H7  # noqa: E402   （v7 韵律）
import 引擎_v7 as E7  # noqa: E402   （v7 引擎）
import 韵律_v6 as H6  # noqa: E402   （只读：legacy 对照/回归）
import 引擎_v6 as E6  # noqa: E402
import 引擎_v5 as E5  # noqa: E402
import 声线覆盖层_v7 as 覆  # noqa: E402   （N2：三角色声线档案覆盖层）
from 接口契约_角色声线 import 声线档案  # noqa: E402

FS = 44100
出目录 = os.path.join(M62, "产物", "判据")
os.makedirs(出目录, exist_ok=True)

带名 = ("0-500", "500-1000", "1000-2000", "2000-4000", "4000-8000", "8000-16000")


def 读档(名="EmoCompanion") -> 声线档案:
    with open(os.path.join(MEET61, "产物", "声线档案", 名 + ".json"), encoding="utf-8") as f:
        return 声线档案.从dict(json.load(f))


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))) if len(x) else 0.0


def _db(a, b):
    return 20.0 * math.log10(max(a, 1e-12) / max(b, 1e-12))


def _峰(x, lo, hi, fs=FS):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 64:
        return 0.0, 0.0
    x = x - float(np.mean(x))
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
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
    P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    P[0] = 0.0
    总 = float(P.sum())
    return float((f * P).sum() / 总) if 总 > 0 else 0.0


def 合成音节(拼: str, 档=None, 种子=300):
    档 = 档 or 读档()
    器 = H7.人类韵律_v5(档, 种子)
    w, 元, 帧 = 器.说(拼)
    return w, 帧


def 分段(w, 帧, 类s):
    out = []
    for i, t in enumerate(帧["声源"]):
        if int(t) in 类s:
            a, b = i * 441, min((i + 1) * 441, len(w))
            out.append(w[a:b])
    return np.concatenate(out) if out else np.zeros(0)


def _包络(x, fs=FS, lif=0.0016):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 256:
        return np.zeros(0), np.zeros(0)
    x = x - float(np.mean(x))
    X = np.fft.rfft(x * np.hanning(len(x)))
    logP = np.log(np.abs(X) + 1e-12)
    cep = np.fft.irfft(logP)
    L = max(4, int(lif * fs))
    cep[L:-L] = 0.0
    env = np.exp(np.fft.rfft(cep).real)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return f, env


def _目标峰(x, 目标Hz, 半宽=0.22, fs=FS):
    f, env = _包络(x, fs)
    if len(f) == 0:
        return 0.0
    m = (f >= 目标Hz * (1.0 - 半宽)) & (f <= 目标Hz * (1.0 + 半宽))
    if not np.any(m):
        return 0.0
    return float(f[m][int(np.argmax(env[m]))])


def _辅音样本(w, 帧):
    out = []
    for i, t in enumerate(帧["声源"]):
        if int(t) == 0 and float(帧["f1"][i]) < 450.0:
            a, b = i * 441, min((i + 1) * 441, len(w))
            out.append(w[a:b])
    return np.concatenate(out) if out else np.zeros(0)


def _辅音帧数(帧):
    return sum(1 for i, t in enumerate(帧["声源"]) if int(t) == 0 and float(帧["f1"][i]) < 450.0)


def _谱带(x, fs=FS):
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


# ═══════════════════════════ 新增：元音身份量法（判别版 / 真人对照） ═══════════════════════════
单韵 = (("啊", "a1"), ("哦", "o1"), ("鹅", "e1"), ("衣", "i1"), ("乌", "u1"), ("迂", "v1"), ("儿", "er2"))


def _共鸣峰(x, 目标Hz, 半宽=0.15):
    """逐共振峰窄窗包络峰（±半宽）与对比度（窗内 max/中位 dB）。"""
    f, env = _包络(x, FS, lif=0.0022)
    if len(f) == 0:
        return 0.0, 0.0
    m = (f >= 目标Hz * (1 - 半宽)) & (f <= 目标Hz * (1 + 半宽))
    if not np.any(m):
        return 0.0, 0.0
    ee = env[m]
    对比 = 20.0 * math.log10(max(float(np.max(ee)), 1e-12) / max(float(np.median(ee)), 1e-12))
    return float(f[m][int(np.argmax(ee))]), float(对比)


def _元音签名(档, 种子=300):
    """7 单元音 → {拼: (F1est, F2est, F1对比dB)}（120Hz 派生声线）。"""
    档低 = 档.派生(基频=120.0)
    缩 = np.asarray(档.共振峰缩放, dtype=np.float64)
    出 = {}
    for 字, 拼 in 单韵:
        s = Y.解析音节(拼)
        目 = s["目标s"][0]
        F1t, F2t = 目[0] * 缩[0], 目[1] * 缩[1]
        帧 = H7.人类韵律_v5(档低, 种子).建帧(字)
        idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
        a0 = (idx[len(idx) // 3] if idx else 0) * 441
        w, _, _ = H7.人类韵律_v5(档低, 种子).说(字)
        seg = w[a0:a0 + int(0.10 * FS)]
        f1, c1 = _共鸣峰(seg, F1t)
        f2, _ = _共鸣峰(seg, F2t)
        出[拼] = dict(目标F1=round(F1t, 1), 目标F2=round(F2t, 1),
                     F1est=round(f1, 1), F2est=round(f2, 1), F1对比dB=round(c1, 2))
    return 出


def 判别交叉(签名) -> (int, list):
    """(F1est,F2est) 最近邻分类 → 交叉数、明细。"""
    交叉 = 0
    明细 = []
    for _, pi in 单韵:
        f1, f2 = 签名[pi]["F1est"], 签名[pi]["F2est"]

        def d(pj):
            return abs(f1 - 签名[pj]["目标F1"]) / 签名[pj]["目标F1"] + \
                   abs(f2 - 签名[pj]["目标F2"]) / 签名[pj]["目标F2"]
        近 = min([p for _, p in 单韵], key=d)
        交叉 += int(近 != pi)
        明细.append("%s→%s" % (pi, 近))
    return 交叉, 明细


def legacy交叉(档, 种子=300) -> (int, list):
    """复现红队量法（**与红队同一管线：韵律_v6 + 引擎_v6** + 宽窗 + 全局包络峰）。"""
    档低 = 档.派生(基频=120.0)
    缩 = np.asarray(档.共振峰缩放, dtype=np.float64)
    窗 = {}
    for _, 拼 in 单韵:
        s = Y.解析音节(拼)
        目 = s["目标s"][0]
        窗[拼] = (目[0] * 缩[0] * 0.85, 目[1] * 缩[1] * 1.15)
    实质 = 0
    明细 = []
    for 字, pi in 单韵:
        帧 = H6.人类韵律_v5(档低, 种子).建帧(字)
        idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
        a0 = (idx[len(idx) // 3] if idx else 0) * 441
        w, _, _ = H6.人类韵律_v5(档低, 种子).说(字)
        seg = w[a0:a0 + int(0.12 * FS)]
        f, env = _包络(seg, FS, lif=0.0022)
        m = (f >= 150.0) & (f <= 3500.0)
        p = float(f[m][int(np.argmax(env[m]))])
        他 = [pj for _, pj in 单韵 if pj != pi and 窗[pj][0] <= p <= 窗[pj][1]]
        实质 += int(len(他) > 0)
        明细.append("%s 峰=%.0f 落他窗=%d" % (pi, p, len(他)))
    return 实质, 明细


def _voiced起始(x, fs=FS, fmin=70.0, fmax=450.0):
    x = np.asarray(x, dtype=np.float64)
    帧长, 跳 = int(0.04 * fs), int(0.01 * fs)
    if len(x) < 帧长 * 2:
        return np.zeros(0, dtype=int)
    k = 1 + (len(x) - 帧长) // 跳
    idx = np.arange(帧长)[None, :] + 跳 * np.arange(k)[:, None]
    fr = x[idx] - x[idx].mean(axis=1, keepdims=True)
    fr *= np.hanning(帧长)[None, :]
    nfft = 1 << int(np.ceil(np.log2(2 * 帧长)))
    F = np.fft.rfft(fr, n=nfft, axis=1)
    ac = np.fft.irfft(F * np.conj(F), n=nfft, axis=1)
    ac /= np.maximum(ac[:, :1], 1e-20)
    lo, hi = max(1, int(fs / fmax)), min(nfft - 1, int(fs / fmin))
    seg = ac[:, lo:hi + 1]
    lag = np.argmax(seg, axis=1) + lo
    r = seg[np.arange(len(lag)), lag - lo]
    return (跳 * np.where(r >= 0.40)[0]).astype(int)


def _读wav(p):
    fs, x = wavfile.read(p)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.max(np.abs(x)) > 1.2:
        x = x / 32768.0
    if fs != FS:
        g = math.gcd(int(fs), FS)
        x = resample_poly(x, FS // g, fs // g)
    return x


def 峰位统计(x, 上限帧=300):
    """浊音帧 → 包络峰位列表 + F1 区(200–1000Hz) 突出度列表。"""
    起s = _voiced起始(x)
    峰s, 突s = [], []
    for a in 起s[:上限帧]:
        seg = x[a:a + int(0.04 * FS)]
        if len(seg) < 1024:
            continue
        f, env = _包络(seg, FS, lif=0.0022)
        m = (f >= 150.0) & (f <= 3500.0)
        峰s.append(float(f[m][int(np.argmax(env[m]))]))
        m1 = (f >= 200.0) & (f <= 1000.0)
        lo = f < 200.0
        突s.append(20.0 * math.log10(max(float(np.max(env[m1])), 1e-12)
                                  / max(float(np.median(env[lo])), 1e-12)))
    return 峰s, 突s


# ═══════════════════════════ 主流程 ═══════════════════════════
def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    档 = 读档("EmoCompanion")
    臂 = {}

    # ---------------- A 音韵映射（保号保定义；只测查表器，与引擎无关） ----------------
    a1 = {"wo3": ("uo", None), "yuan2": ("van", "n"), "yi1": ("i", None),
          "yao4": ("iao", None), "wei4": ("uei", None), "wang4": ("uang", "ng"),
          "yue4": ("ve", None), "yun2": ("vn", "n")}
    ok = all(Y.解析音节(k)["韵"] == v[0] and Y.解析音节(k)["尾"] == v[1] for k, v in a1.items())
    臂["A1_yw零声母归一"] = dict(通过=bool(ok), 读数={k: Y.解析音节(k)["韵"] for k in a1})
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
                         读数=dict(音节数=审["音节数"], 问题数=审["问题数"], 问题=审["问题"][:8]))

    # ---------------- B 辅音实现（保号保定义；v7 噪声道逐位同 v6） ----------------
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
    部位序 = [辅[p]["噪重心Hz"] for p in ("sa", "xia", "sha")]
    f弱 = (辅["fa"]["清浊差dB"] < 辅["sha"]["清浊差dB"] - 0.5)
    臂["B1_擦音部位序"] = dict(
        通过=bool(部位序[0] > 部位序[1] > 部位序[2] and f弱),
        读数=dict(顺序="s>x>sh", 重心Hz=dict(zip(("sa", "xia", "sha"), 部位序)),
                 f重心Hz=辅["fa"]["噪重心Hz"], f弱于sh=bool(f弱)))
    强擦 = [辅[p]["清浊差dB"] for p in ("sa", "sha", "xia") if 辅[p]["清浊差dB"] is not None]
    弱擦 = 辅["ha"]["清浊差dB"] if 辅["ha"]["清浊差dB"] is not None else -99
    臂["B2_擦音电平"] = dict(通过=bool(强擦 and all(-16.0 <= d <= -2.0 for d in 强擦) and -24.0 <= 弱擦 <= -5.0),
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
    vot差 = {q: 塞[q + "a"]["噪ms"] for q in "bdg"}
    asp = {q: 塞[q + "a"]["噪ms"] for q in "ptk"}
    对 = {"p": "b", "t": "d", "k": "g"}
    臂["B4_送气对比"] = dict(通过=bool(all(asp[q] >= 2.0 * max(vot差[对[q]], 10.0) for q in "ptk")),
                         读数=dict(不送气ms=vot差, 送气ms=asp))

    # ---------------- C CV 协同发音（保号保定义） ----------------
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
        C[p] = dict(F2起Hz=round(_目标峰(起, 期望起[p]), 1),
                    F2中Hz=round(_目标峰(中, 1305.0), 1))
        C[p]["ΔF2"] = round(C[p]["F2起Hz"] - C[p]["F2中Hz"], 1)
    ord_ok = C["ba"]["F2起Hz"] < C["da"]["F2起Hz"] < C["ga"]["F2起Hz"]
    臂["C1_塞音F2轨迹"] = dict(通过=bool(ord_ok),
                          读数=dict(顺序="b<d<g", F2起={k: C[k]["F2起Hz"] for k in ("ba", "da", "ga")}))
    转 = max(abs(C[p]["ΔF2"]) for p in C)
    臂["C2_CV过渡幅度"] = dict(通过=bool(转 >= 80.0), 读数=dict(最大ΔF2Hz=round(转, 1), 逐音节=C))
    调测 = {}
    for 拼 in ("ma1", "ma2", "ma3", "ma4"):
        w, _, _ = H7.人类韵律_v5(档, 300).说音节(拼)
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

    # ---------------- D 元音空间（保号保定义；120Hz 派生声线） ----------------
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
    N = {}
    for p, 字p in (("an", "安"), ("ang", "昂")):
        w, 帧 = 合成音节(字p, 档, 300)
        浊idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
        i0 = 浊idx[int(len(浊idx) * 0.72)]
        seg = w[i0 * 441:(i0 * 441) + int(0.06 * FS)]
        N[p] = round(_目标峰(seg, 1670.0 if p == "an" else 1148.0), 1)
    臂["D3_鼻尾区别"] = dict(通过=bool(abs(N["an"] - N["ang"]) >= 200),
                         读数=dict(an=N["an"], ang=N["ang"], 差=round(N["an"] - N["ang"], 1)))

    # ---------------- E 语流波形（保号保定义） ----------------
    句s = {}
    句波形 = {}
    for i, 句 in enumerate(("你好，我是EmoCompanion。",
                            "今天的风很轻，适合慢慢地说一会儿话。",
                            "语言不只是声音，它把一个人的温度，送到另一个人的心里。")):
        w, 元, 帧 = H7.人类韵律_v5(档, 300 + i).说(句)
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
            x = _读wav(ps[0])
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
        各 = []
        for 名 in ("EmoCompanion", "SpeakerA", "洛天依"):
            for p in sorted(glob.glob(os.path.join(MEET61, "素材", 名, "ref44_*.wav"))):
                各.append(_谱带(_读wav(p)))
        return {b: [round(min(z[b] for z in 各 if b in z), 4),
                    round(max(z[b] for z in 各 if b in z), 4)] for b in 合成带}

    帧缘 = H7.人类韵律_v5(档, 300).建帧("你好，我是EmoCompanion。")
    w缘6 = E6.合成(帧缘, 音色=H6.人类韵律_v5(档, 300).音色(), 种子=300)[0]
    w缘7 = 句波形["你好，我是EmoCompanion。"]
    噪6 = _谱带(分段(w缘6, 帧缘, {0}))
    噪7 = _谱带(分段(w缘7, 帧缘, {0}))
    噪同 = bool(all(abs(噪6[b] - 噪7[b]) < 1e-9 for b in 噪6))
    臂["E7_噪帧带分布未变"] = dict(
        通过=噪同,
        读数=dict(说明="同帧同音色，只换引擎；v7 噪帧带分布必须逐位同 v6（辅音不回归）",
                 v6噪=噪6, v7噪=噪7, 噪L1=round(float(sum(abs(噪6[b] - 噪7[b]) for b in 噪6)), 6)))

    臂["E5_谱形贴近参考"] = dict(通过=bool(重比 <= 0.25 and 带距 <= 0.25),
                            读数=dict(合成重心=缘句["谱重心Hz"], 参考重心均值=round(均重),
                                    相对差=round(重比, 3), 带分布L1=round(带距, 3),
                                    合成带=合成带, 参考带均值={k: round(v, 4) for k, v in 均值带.items()},
                                    参考带范围=_参考范围(),
                                    注="[裁决 §6 附] E5 不作耳听满意的门，见 说明_v7.md"))
    臂["E1_静音占比"] = dict(通过=bool(缘句["静音占比"] < 0.20),
                         读数=dict(合成=缘句["静音占比"], 参考=ref["EmoCompanion"]["静音占比"]))
    臂["E2_清浊电平差"] = dict(通过=bool(-18.0 <= 缘句["清浊差dB"] <= -4.0),
                           读数=dict(合成=缘句["清浊差dB"]))
    臂["E3_包络动态"] = dict(通过=bool(缘句["包络标准差dB"] <= 35.0),
                         读数=dict(合成=缘句["包络标准差dB"], 参考=ref["EmoCompanion"]["包络标准差dB"]))
    臂["E4_音节率"] = dict(通过=bool(all(3.8 <= v["音节率"] <= 5.8 for v in 句s.values())),
                        读数={k: v["音节率"] for k, v in 句s.items()})
    wa, _, _ = H7.人类韵律_v5(档, 300).说("你好，我是EmoCompanion。")
    wb, _, _ = H7.人类韵律_v5(档, 300).说("你好，我是EmoCompanion。")
    wc, _, _ = H7.人类韵律_v5(档, 301).说("你好，我是EmoCompanion。")
    同 = bool(np.array_equal(wa, wb))
    异 = bool(not np.array_equal(wa, wc))
    臂["E6_确定性"] = dict(通过=bool(同 and 异), 读数=dict(两趟同=同, 异种子异=异))

    # ---------------- F 耳听等价（保号保定义，v6 口径） ----------------
    def _峰与台(x, f1):
        f, env = _包络(x, FS, lif=0.0022)
        m = (f >= 150.0) & (f <= 3500.0)
        峰位 = float(f[m][int(np.argmax(env[m]))])
        lo = (f >= 150.0) & (f < f1 * 0.75)
        台 = float(np.median(env[lo])) if np.any(lo) else 0.0
        return 峰位, round(20.0 * math.log10(max(env[m].max(), 1e-12) / max(台, 1e-12)), 1)

    样 = (("啊", "a1"), ("哦", "o1"), ("鹅", "e1"), ("衣", "i1"), ("乌", "u1"),
         ("迂", "v1"), ("儿", "er2"), ("好", "hao3"), ("我", "wo3"), ("是", "shi4"),
         ("天", "tian1"), ("慢", "man4"), ("说", "shuo1"))
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
        帧 = H7.人类韵律_v5(档低, 300).建帧(字)
        idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
        a0 = (idx[len(idx) // 3] if idx else 0) * 441
        x7, _, _ = H7.人类韵律_v5(档低, 300).说(字)
        x5 = E5.合成(帧, 音色=H6.人类韵律_v5(档低, 300).音色(), 种子=300)[0]
        seg7 = x7[a0:a0 + int(0.12 * FS)]
        seg5 = x5[a0:a0 + int(0.12 * FS)]
        峰7, 突7 = _峰与台(seg7, 目s[0][0] * 档.共振峰缩放[0]) if len(seg7) > 512 else (0.0, 0.0)
        峰5, 突5 = _峰与台(seg5, 目s[0][0] * 档.共振峰缩放[0]) if len(seg5) > 512 else (0.0, 0.0)
        落 = bool(窗[0] <= 峰7 <= 窗[1])
        落5 = bool(窗[0] <= 峰5 <= 窗[1])
        FF[拼] = dict(韵=s["韵"], 单元音=单元, 首目标F1=round(目s[0][0] * 档.共振峰缩放[0], 1),
                     窗=[round(窗[0], 1), round(窗[1], 1)],
                     v7包络峰=round(峰7, 1), v5包络峰=round(峰5, 1),
                     v7突出dB=突7, v5突出dB=突5, v7落位=落, v5落位=落5)
    全落 = all(v["v7落位"] for v in FF.values())
    落5数 = sum(1 for v in FF.values() if v["v5落位"])
    臂["F1_元音谱包络峰位"] = dict(
        通过=bool(全落 and 落5数 < len(FF)),
        读数=dict(说明="保号保定义（同 v6）：谱包络最大点须落在 [F1×0.85, F2×1.15]；"
                    "同一判据下 **v5（并联低频平台版）** 的落位数必须更少",
                v7落位=f"{sum(1 for v in FF.values() if v['v7落位'])}/{len(FF)}",
                v5落位=f"{落5数}/{len(FF)}", 逐音节=FF))
    臂["F2_元音F1F2区突出"] = dict(
        通过=bool(all(v["v7突出dB"] >= 3.0 for v in FF.values() if v["首目标F1"] >= 500.0)),
        读数={"v7突出dB": {k: v["v7突出dB"] for k, v in FF.items()},
              "v5突出dB": {k: v["v5突出dB"] for k, v in FF.items()},
              "门": "F1≥500Hz 的元音，包络峰须比 F1 以下的谱台高 ≥3dB"})
    单 = {}
    for 拼 in ("a1", "o1", "e1", "i1", "u1", "v1", "er2"):
        s = Y.解析音节(拼)
        目 = [t * v for t, v in zip(s["目标s"][0], (档.共振峰缩放[0], 档.共振峰缩放[1]))]
        字 = {"a": "啊", "o": "哦", "e": "鹅", "i": "衣", "u": "乌", "v": "迂", "er": "儿"}[s["韵"]]
        帧 = H7.人类韵律_v5(档低, 300).建帧(字)
        idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
        a0 = (idx[len(idx) // 3] if idx else 0) * 441
        w, _, _ = H7.人类韵律_v5(档低, 300).说(字)
        seg = w[a0:a0 + int(0.10 * FS)]
        f, env = _包络(seg, FS, lif=0.0022)
        m = (f >= 150.0) & (f <= 3500.0)
        ff, ee = f[m], env[m]
        峰s = [float(ff[i]) for i in range(1, len(ee) - 1) if ee[i] >= ee[i - 1] and ee[i] > ee[i + 1]]

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
                             读数=dict(中位误差=round(float(np.median(误s)), 3), 逐韵=单))
    噪帧 = dict(f0=[180.0] * 20, f1=[300.0] * 20, f2=[1800.0] * 20, f3=[2600.0] * 20,
               f4=[3500.0] * 20, f5=[4500.0] * 20, 幅=[0.8] * 20, 气声=[0.0] * 20,
               声源=[0] * 20, 噪带=[(6800.0, 2200.0)] * 20, 送气=[1.0] * 20,
               紧喉=[0.3] * 20, 粗糙=[0.0] * 20, 噪形=[0] * 20)
    y5, _ = E5.合成(噪帧, 音色=None, 种子=300)
    y7, _ = E7.合成(噪帧, 音色=None, 种子=300)
    同噪 = bool(np.array_equal(y5, y7))
    臂["F4_噪声道逐位同v5"] = dict(通过=同噪, 读数=dict(逐位同=同噪, 样本数=len(y5),
                                                 说明="v7 噪声/清辅音声道与 v5 逐位同（C14 护栏）"))

    # ═════════════ 新增臂（Task 4） ═════════════
    签名 = _元音签名(档, 300)
    交叉, 交叉明细 = 判别交叉(签名)
    臂["N1_元音身份可分(判别版)"] = dict(
        通过=bool(交叉 <= 1),
        读数=dict(说明="每元音按 F1/F2 逐共振峰 ±15% 窄窗取包络峰 → (F1est,F2est) 最近邻分类；"
                    "交叉落位须 ≤1（修红队『窗宽买通』）",
                交叉数=交叉, 明细=交叉明细, 签名=签名))

    # N2 音色身份可分：三角色同句合成，按 (F0 半音, 谱包络距离) 判两两可分
    角色 = ("洛天依", "SpeakerA", "EmoCompanion")
    音色句 = "今天的风很轻，适合慢慢地说一会儿话。"
    音色读数 = {}
    波形色 = {}
    for n in 角色:
        档n = 覆.取(n)     # N2：经 meeting62「声线档案覆盖层」派生（不改 meeting61 档案、不改阈值/口径）
        w, _, 帧n = H7.人类韵律_v5(档n, 300).说(音色句)
        波形色[n] = w
        浊 = 分段(w, 帧n, {1})
        # F0 取**引擎帧序列的浊帧**（构造上 = 瞬时频率：相位 = cumsum(f0/FS)，见 引擎_v7.合成）；
        # 自相关对高 F0/长句不稳定（实测见 说明_v7 §未测），故此处不用波形侧估计。
        f0v = np.array([帧n["f0"][i] for i, t in enumerate(帧n["声源"]) if int(t) == 1])
        音色读数[n] = dict(F0中位Hz=round(float(np.median(f0v)), 1) if f0v.size else 0.0,
                          有效F0=round(float(档n.基频 * (1.0 + 0.5 * 档n.声区)), 1),
                          谱带=_谱带(w), 浊谱带=_谱带(浊))
    refF0 = {}
    for n in 角色:
        拟合 = json.load(open(os.path.join(MEET61, "产物", "指标", "拟合44_%s.json" % n), encoding="utf-8"))
        refF0[n] = round(float(拟合["参考统计"]["F0中位"]), 1)
    两两 = {}
    可分对 = 0
    for i in range(len(角色)):
        for j in range(i + 1, len(角色)):
            a, b = 角色[i], 角色[j]
            st = abs(_半音(音色读数[a]["F0中位Hz"], 音色读数[b]["F0中位Hz"]))
            ba = 音色读数[a]["浊谱带"]
            bb = 音色读数[b]["浊谱带"]
            谱距 = float(sum(abs(ba[k] - bb[k]) for k in ba if k in bb))
            可分 = bool(st >= 1.0 or 谱距 >= 0.20)
            两两["%s-%s" % (a, b)] = dict(F0半音=round(st, 3), 浊谱带L1=round(谱距, 4),
                                        判定="可分" if 可分 else "听不出")
            可分对 += int(可分)
    臂["N2_音色身份可分"] = dict(
        通过=bool(可分对 == 3),
        读数=dict(说明="三角色同句 v7 合成；对每对判 (F0 半音 ≥1.0 或 浊谱带 L1 ≥0.20) 才『可分』；"
                    "须 3/3 对可分才算通过（阈值 [提案]，未真人盲评）。F0 取引擎帧序列浊帧（=瞬时频率）",
                可分对数=可分对, 两两=两两,
                合成F0中位={n: 音色读数[n]["F0中位Hz"] for n in 角色},
                有效F0={n: 音色读数[n]["有效F0"] for n in 角色},
                参考F0中位=refF0,
                覆盖层=覆.说明(),
                档案层恒同维=覆.恒同维诊断(),
                注="修复：三角色经 meeting62「声线档案覆盖层」派生（共振峰缩放 3 轴 + 谱倾斜，"
                   "值由 meeting61 已固化的参考录音统计确定性派生）⇒ 经引擎真实合成路径生效，非事后 EQ。"
                   "**阈值/口径一字未改**（仍 ≥1.0 半音 / ≥0.20 L1，仍 3/3 才算过）。"
                   "登–缘 F0 0.19 半音来自参考本身（322.1/318.5）⇒ F0 路不可达，仅靠参考可分的共振峰层达 L1≥0.20。"
                   "本臂未真人 A/B 验证"))

    # N3/N4/N5 唱歌（谱管线_v7 驱动 声带后端_v7）
    import 谱_亚版本_声库_v7 as 唱管
    合唱 = 唱管.主(os.path.join(M62, "产物", "谱管线_v7"))
    音符s = [x for x in 合唱["f0实测"]]
    音分s = [abs(x["音分"]) for x in 音符s if x["音分"] is not None]
    臂["N3_唱歌音准"] = dict(
        通过=bool(len(音分s) == len(音符s) and max(音分s) <= 25.0),
        读数=dict(说明="谱音符决定音高；去颤中心（中位 f0）与谱音高误差须 ≤25 音分",
                逐音=音符s, 最大音分=round(max(音分s), 1) if 音分s else None))
    # 逐音 f0 命中时长（±60 音分）vs 谱时值
    wav1 = 合唱["wav"]["趟1"]
    fs, x1 = wavfile.read(wav1)
    x1 = np.asarray(x1, dtype=np.float64) / 32768.0
    tss, f0s2 = 唱管._f0轨(x1)
    命中 = {}
    for s in 唱管.契.建亚版本(唱管.契.谱.从JSON文件(os.path.join(M62, "素材", "小谱_样本.json"))).音节s:
        if s.休止:
            continue
        lo = s.音高Hz * 2 ** (-0.6 / 12)
        hi = s.音高Hz * 2 ** (0.6 / 12)
        m = (f0s2 >= lo) & (f0s2 <= hi)
        测长 = float(np.sum(m) * 0.010)
        命中["%s@%.1fHz" % (s.歌词, s.音高Hz)] = dict(谱时值=round(s.时长秒, 3),
                                                    命中时长=round(测长, 3),
                                                    误差=round(abs(测长 - s.时长秒) / s.时长秒, 3))
    节奏ok = all(v["误差"] <= 0.30 for v in 命中.values())
    总时长误 = abs(len(x1) / FS - 3.6) / 3.6
    臂["N4_唱歌节奏"] = dict(
        通过=bool(节奏ok and 总时长误 <= 0.15),
        读数=dict(说明="逐音 f0 命中时长(±60音分) vs 谱时值 误差 ≤0.30；整句时长误差 ≤0.15",
                逐音=命中, 总时长秒=round(len(x1) / FS, 3), 谱总时长秒=3.6,
                总时长误差=round(总时长误, 4)))

    # N5 长音 F0 颤音（修①可证伪）：用「长音变体谱」渲染
    long谱路 = os.path.join(M62, "素材", "小谱_长音.json")
    颤音读数 = None
    if os.path.exists(long谱路):
        长谱 = 唱管.契.谱.从JSON文件(long谱路)
        长亚 = 唱管.契.建亚版本(长谱)
        长r = 唱管.谱声带V7(种子=300, 目录=os.path.join(M62, "产物", "谱管线_v7", "长音")).唱(
            长亚, 名="长音_v7_小谱")
        长w = 长r["波形"]
        tt, ff = 唱管._f0轨(长w)
        非休 = [i for i, s in enumerate(长亚.音节s) if not s.休止]
        末 = 长亚.音节s[非休[-1]]
        t0 = sum(s.时长秒 for s in 长亚.音节s[:非休[-1]])
        m = (tt >= t0 + 0.20) & (tt <= t0 + 末.时长秒 + 0.05) & (ff > 0)
        v = ff[m]
        if v.size >= 8:
            tt2 = tt[m]
            c = np.polyfit(tt2, v, 2)
            d = v - np.polyval(c, tt2)
            amp = 12.0 * math.log2(1.0 + float(np.std(d)) * math.sqrt(2.0) / max(float(np.median(v)), 1e-6))
            中心 = float(np.median(v))
            音分 = 1200.0 * math.log2(max(中心, 1e-6) / 末.音高Hz)
            dd = d - d.mean()
            pk = sum(1 for i in range(1, len(dd) - 1) if dd[i] > dd[i - 1] and dd[i] >= dd[i + 1])
            速率 = pk / max(tt2[-1] - tt2[0], 1e-6)
            颤音读数 = dict(长音时长秒=round(末.时长秒, 3), 期望Hz=round(末.音高Hz, 1),
                         去颤中心Hz=round(中心, 1), 中心音分=round(音分, 1),
                         去趋势幅度半音=round(amp, 3), 速率Hz=round(速率, 2),
                         波峰数=pk, 窗秒=[round(tt2[0], 2), round(tt2[-1], 2)])
            okN5 = bool(abs(音分) <= 25.0 and 0.4 <= amp <= 1.6 and 4.0 <= 速率 <= 9.0)
        else:
            okN5 = False
    臂["N5_长音F0颤音"] = dict(
        通过=bool(颤音读数 and okN5),
        读数=dict(说明="修①：长音 f0 去颤中心音准 ≤25 音分、去趋势幅度 ∈[0.4,1.6] 半音、速率 ∈[4,9]Hz",
                读数=颤音读数, 路径=(长r["wav"] if 颤音读数 else None)))

    # ═════════════ must-fail（必过） ═════════════
    # M1/M2 白噪/静音 sham：喂进**元音峰位同一管线**（voiced 门 + 包络），须 FAIL
    rng = np.random.default_rng(62019)
    白噪 = rng.standard_normal(int(0.30 * FS))
    静音 = np.zeros(int(0.30 * FS))
    子窗 = {拼: (Y.解析音节(拼)["目标s"][0][0] * 档.共振峰缩放[0] * 0.85,
               Y.解析音节(拼)["目标s"][0][1] * 档.共振峰缩放[1] * 1.15) for _, 拼 in 单韵}

    def sham落位(x):
        起s = _voiced起始(x)
        if len(起s) == 0:
            return dict(浊帧=0, 落位=False, 峰=None)
        seg = x[int(起s[0]):int(起s[0]) + int(0.12 * FS)]
        if len(seg) < 512:
            return dict(浊帧=len(起s), 落位=False, 峰=None)
        f, env = _包络(seg, FS, lif=0.0022)
        m = (f >= 150.0) & (f <= 3500.0)
        p = float(f[m][int(np.argmax(env[m]))])
        落 = any(lo <= p <= hi for lo, hi in 子窗.values())
        return dict(浊帧=int(len(起s)), 峰=round(p, 1), 落位=bool(落))
    m1 = sham落位(白噪)
    m2 = sham落位(静音)
    臂["M1_白噪sham"] = dict(通过=bool(not m1["落位"]), 读数=dict(说明="白噪喂元音峰位管线须 FAIL（不落任何元音窗）", **m1))
    臂["M2_静音sham"] = dict(通过=bool(not m2["落位"]), 读数=dict(说明="静音喂元音峰位管线须 FAIL（无浊帧/不落窗）", **m2))

    # M3 窗交叉 sham：复现 v6 legacy 7/7（宽窗）；断言判别版 ≤1
    legacy数, legacy明细 = legacy交叉(档, 300)
    臂["M3_窗交叉sham"] = dict(
        通过=bool(legacy数 >= 6 and 交叉 <= 1),
        读数=dict(说明="复现红队【v6 legacy 宽窗 7 单元音交叉落位 7/7】；断言判别版（逐共振峰窄窗最近邻）≤1",
                legacy交叉数=legacy数, legacy明细=legacy明细,
                判别交叉数=交叉, 判别明细=交叉明细))

    # ═════════════ 准入门：外部真人正对照 ═════════════
    真人 = {}
    真人峰中位 = []
    真人突中位 = []
    for 名 in ("EmoCompanion", "SpeakerA", "洛天依"):
        for p in sorted(glob.glob(os.path.join(MEET61, "素材", 名, "ref44_*.wav"))):
            x = _读wav(p)
            峰s, 突s = 峰位统计(x)
            if not 峰s:
                真人[os.path.basename(p)] = dict(通过=False, 浊帧=0)
                continue
            中位 = float(np.median(峰s))
            突 = float(np.median(突s))
            真人[os.path.basename(p)] = dict(通过=bool(len(峰s) >= 30 and 180.0 <= 中位 <= 1300.0 and 突 >= 3.0),
                                          浊帧=len(峰s), 峰位中位Hz=round(中位, 1),
                                          F1区突出中位dB=round(突, 2))
            真人峰中位.append(中位)
            真人突中位.append(突)
    真人全过 = bool(真人 and all(v["通过"] for v in 真人.values()))
    # 合成对照（同量）：v7 元音空间也须落在真人峰位域内
    合成峰s, 合成突s = 峰位统计(波形色["EmoCompanion"])
    合成中位 = float(np.median(合成峰s)) if 合成峰s else 0.0
    臂["P0_外部真人正对照"] = dict(
        通过=真人全过,
        读数=dict(说明="同一套元音峰位量法跑 11 个人类 ref44 wav；每文件须 峰位中位∈[180,1300]Hz 且 "
                    "F1区(200–1000Hz)突出中位≥3dB 且 浊帧≥30 ⇒ 真人必须 PASS（任何新 PASS 的准入门）",
                文件数=len(真人), 全过=真人全过, 逐文件=真人,
                真人峰位中位域=[round(min(真人峰中位), 1), round(max(真人峰中位), 1)] if 真人峰中位 else None,
                真人F1突中位域=[round(min(真人突中位), 2), round(max(真人突中位), 2)] if 真人突中位 else None,
                合成EmoCompanion峰位中位Hz=round(合成中位, 1),
                合成EmoCompanionF1突中位dB=round(float(np.median(合成突s)), 2) if 合成突s else None,
                注="本臂验证『量法对真实语音可用』；噪声拒绝由 M1/M2 承担"))

    # ═════════════ 汇总 ═════════════
    通过数 = sum(1 for v in 臂.values() if v["通过"])
    报 = dict(工具="判据_发音_v7.py", 版本="v7-44k(颤音入F0+单调高支)",
              臂数=len(臂), 通过=通过数, 未过=len(臂) - 通过数,
              臂=臂, 参考=ref, 句级=句s,
              must_fail=dict(M1=臂["M1_白噪sham"]["通过"], M2=臂["M2_静音sham"]["通过"],
                             M3=臂["M3_窗交叉sham"]["通过"]),
              准入门=dict(P0=臂["P0_外部真人正对照"]["通过"]),
              合成带=合成带)
    out = os.path.join(出目录, "判据_发音_v7.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(报, f, ensure_ascii=False, indent=1)

    print("=" * 96)
    for k, v in 臂.items():
        print("%-22s %s  %s" % (k, "PASS" if v["通过"] else "FAIL",
                               json.dumps(v["读数"], ensure_ascii=False)[:130]))
    print("-" * 96)
    print("总计 %d/%d PASS | must-fail M1/M2/M3=%s/%s/%s | 真人准入门 P0=%s"
          % (通过数, len(臂), 臂["M1_白噪sham"]["通过"], 臂["M2_静音sham"]["通过"],
             臂["M3_窗交叉sham"]["通过"], 臂["P0_外部真人正对照"]["通过"]))
    print("JSON:", out)
    return 0 if (通过数 == len(臂)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
