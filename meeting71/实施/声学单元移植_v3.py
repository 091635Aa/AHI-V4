# -*- coding: utf-8 -*-
"""meeting71/实施/声学单元移植_v3.py —— 用**声学切分**给特征点定界（修 v1/v2 的根因）。

根因（`32` 已测）：`数据_对齐asr` 的音节帧边界与谱通量无关（flux≈随机）⇒ 按它切的「音节」
是半音节。v3：**标签仍用 ASR 字序（粗）**，但**边界改用能量核 + 能量谷（细）**，
得到真正的音节特征点；再按 v2 的频域包络移植 + 40 ms 插值合成。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import sys
import time
import wave
from pathlib import Path

import numpy as np
import librosa

HERE = Path(__file__).resolve().parent
M71 = HERE.parent
M70 = M71.parent / "meeting70"
M67 = M71.parent / "meeting67"
M66 = M71.parent / "meeting66"
M64 = M71.parent / "meeting64" / "实施"
出 = M71 / "产物"
SR = 22050
帧样本 = 110
D2目录 = M67 / "数据" / "子集_D2_le30min" / "样本"
音频根 = Path(r"<path>\SpeakerA7代模型训练数据")
插值帧 = 6
库缓存 = 出 / "库_声学_D2.pkl"
句表 = {
    "V3主句": "今天天气很好我听见了自己的声音我想再说一遍",
    "V3新句1": "今天的风很轻我慢慢说给你听",
    "V3新句2": "窗外的雨停了我该休息一会",
    "V3新句3": "你先别走我还想再说两句",
}


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


器 = 载("器官v2", M70 / "实施" / "器官v2.py")
量 = 载("量具_v9", M64 / "量具_v9.py")


def 写(p, a):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def 找单元(x, 最短=0.06, 最长=0.45):
    """能量核 + 能量谷 → 声学单元边界（秒）。"""
    hop = 220
    e = librosa.feature.rms(y=np.asarray(x, np.float64), frame_length=1024, hop_length=hop)[0]
    e = np.convolve(e, np.ones(3) / 3, mode="same")
    if e.size < 5:
        return []
    thr = max(0.06 * float(np.percentile(e, 95)), 1e-5)
    cand = [i for i in range(1, len(e) - 1) if e[i] >= e[i - 1] and e[i] >= e[i + 1] and e[i] > thr]
    # 最小间距
    keep = []
    for i in cand:
        if not keep or (i - keep[-1]) * hop / SR >= 最短:
            keep.append(i)
        elif e[i] > e[keep[-1]]:
            keep[-1] = i
    边界 = [0]
    for a, b in zip(keep[:-1], keep[1:]):
        seg = e[a:b + 1]
        j = int(a + np.argmin(seg))
        边界.append(j)
    边界.append(len(e) - 1)
    单元 = []
    最长短 = 最短 * SR / hop
    for a, b in zip(边界[:-1], 边界[1:]):
        if (b - a) < 最长短:
            continue
        if (b - a) * hop / SR > 最长:
            continue
        单元.append((a * hop, b * hop))
    return 单元


def 包络_由LPC(A, n_fft=1024):
    out = np.empty((len(A), n_fft // 2 + 1), np.float64)
    for i, a in enumerate(A):
        H = np.fft.rfft(a, n=n_fft)
        out[i] = -np.log(np.abs(H) + 1e-6)
    return out


def 对齐标签(单元中心秒, 字中心秒, 罚=0.20):
    """DP 单调对齐：match / skip-unit / skip-char，返回每单元的字下标（未匹配给最近字）。"""
    n = len(单元中心秒); m = len(字中心秒)
    if n == 0 or m == 0:
        return [0] * n
    INF = 1e18
    dp = np.full((n + 1, m + 1), INF)
    bk = np.zeros((n + 1, m + 1), np.int8)
    dp[0, 0] = 0.0
    cu = np.asarray(单元中心秒, np.float64); cc = np.asarray(字中心秒, np.float64)
    for i in range(n + 1):
        for j in range(m + 1):
            v = dp[i, j]
            if v >= INF:
                continue
            if i < n and j < m:
                c = abs(cu[i] - cc[j])
                if v + c < dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = v + c; bk[i + 1, j + 1] = 1
            if i < n and v + 罚 < dp[i + 1, j]:
                dp[i + 1, j] = v + 罚; bk[i + 1, j] = 2
            if j < m and v + 罚 < dp[i, j + 1]:
                dp[i, j + 1] = v + 罚; bk[i, j + 1] = 3
    标签 = [-1] * n
    i, j = n, m
    while i > 0 or j > 0:
        op = bk[i, j]
        if op == 1:
            标签[i - 1] = j - 1; i -= 1; j -= 1
        elif op == 2:
            i -= 1
        else:
            j -= 1
    for k in range(n):
        if 标签[k] < 0:
            标签[k] = int(np.argmin(np.abs(cc - cu[k])))
    return 标签


def 建库_声学():
    meta = json.loads((M66 / "产物" / "数据_对齐asr" / "meta.json").read_text(encoding="utf-8"))
    样本 = {r["id"]: r for r in meta["样本"]}
    对齐j = json.loads((M67 / "产物" / "P2" / "对齐证据.json").read_text(encoding="utf-8"))
    偏移 = {r["id"]: int(r["偏移帧"]) for r in 对齐j["per_sample"]}
    库 = []
    for f in sorted(D2目录.glob("*.npz")):
        sid = int(f.stem)
        z = np.load(f, allow_pickle=True)
        x = 器.耳.读wav(音频根 / 样本[sid]["wav"])
        c = 器.分析控制(x)
        chs = [ch for ch in 样本[sid]["文"] if "\u4e00" <= ch <= "\u9fff"]
        声, 韵, 调id, 帧数 = z["声母"], z["韵母"], z["调"], z["音节帧数"]
        n = min(len(chs), len(声), len(帧数))
        if n < 2:
            continue
        # ASR 字中心：10 ms 帧 + **偏移帧**（修 bug：之前漏了偏移，错位可达 11.9 s）
        cen = (偏移.get(sid, 0) + np.cumsum(帧数[:n]) - 帧数[:n] / 2) * 0.01
        单元秒 = 找单元(x)
        if len(单元秒) < 2:
            continue
        中心秒 = np.array([(a + b) / 2 / SR for a, b in 单元秒])
        标签 = 对齐标签(中心秒, cen)
        for (a, b), k in zip(单元秒, 标签):
            a2 = max(0, a // 帧样本 - 1); b2 = min(c["T"], b // 帧样本 + 1)
            if b2 - a2 < 6:
                continue
            库.append(dict(sid=sid, sm=int(声[k]), ym=int(韵[k]), tone=int(调id[k]), ch=chs[k],
                           f0=c["f0"][a2:b2].astype(np.float32), F=c["F"][a2:b2].astype(np.float32),
                           B=c["B"][a2:b2].astype(np.float32), E=c["E"][a2:b2].astype(np.float32),
                           NR=c["NR"][a2:b2].astype(np.float32), A=c["A"][a2:b2].astype(np.float32)))
    return 库


def 载库():
    出.mkdir(parents=True, exist_ok=True)
    if 库缓存.exists():
        with open(库缓存, "rb") as f:
            return pickle.load(f)
    库 = 建库_声学()
    with open(库缓存, "wb") as f:
        pickle.dump(库, f, protocol=4)
    return 库


def 建字表(库):
    from collections import Counter
    表 = {}
    for u in 库:
        if u.get("ch"):
            表.setdefault(u["ch"], []).append((u["sm"], u["ym"], u["tone"]))
    return {ch: Counter(v).most_common(1)[0][0] for ch, v in 表.items()}


def 选单元(库, 键, ch):
    sm, ym, tone = 键
    for 条件 in ((lambda u: u["sm"] == sm and u["ym"] == ym and u["tone"] == tone and u.get("ch") == ch),
                 (lambda u: u["sm"] == sm and u["ym"] == ym and u["tone"] == tone),
                 (lambda u: u["sm"] == sm and u["ym"] == ym)):
        候 = [u for u in 库 if 条件(u)]
        if 候:
            候.sort(key=lambda u: abs(len(u["f0"]) - 36))
            return 候[0]
    return None


def 键_of(文本, 字表):
    from pypinyin import lazy_pinyin, Style
    声母表 = ["∅", "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s"]
    韵母表 = ["a", "o", "e", "i", "u", "v", "er", "ii", "iii", "ai", "ei", "ao", "ou", "ia", "ie", "iao", "iou", "ua", "uo",
             "uai", "uei", "ve", "an", "en", "in", "uen", "vn", "ang", "eng", "ing", "ong", "iong", "ian", "uan", "uang",
             "van", "", "iang", "io"]
    字s = [ch for ch in 文本 if "\u4e00" <= ch <= "\u9fff"]
    keys = []
    for ch in 字s:
        if ch in 字表:
            keys.append((ch, 字表[ch], "字表")); continue
        ys = lazy_pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True)
        if not ys:
            continue
        s = ys[0]; tone = 5
        if s and s[-1].isdigit():
            tone = int(s[-1]); s = s[:-1]
        sm = "∅"; rest = s
        for ini in ("zh", "ch", "sh"):
            if s.startswith(ini):
                sm, rest = ini, s[len(ini):]; break
        if sm == "∅":
            for ini in "bpmfdtnlgkhjqxrzcs":
                if s.startswith(ini):
                    sm, rest = ini, s[1:]; break
        if sm in 声母表 and rest in 韵母表:
            keys.append((ch, (声母表.index(sm), 韵母表.index(rest), max(0, min(4, tone - 1))), "拼音"))
    return keys


def 合成序列(文本, 库, 字表, 基F0=300.0):
    keys = 键_of(文本, 字表)
    单元 = []; 状态 = []
    for ch, 键, 来源 in keys:
        u = 选单元(库, 键, ch)
        if u is None:
            状态.append((ch, "缺")); continue
        状态.append((ch, 来源)); 单元.append(u)
    if not 单元:
        return None, 状态
    n = len(单元)
    for j, u in enumerate(单元):
        v = u["f0"][u["f0"] > 0]
        med = float(np.median(v)) if len(v) else 基F0
        u["f0"] = np.where(u["f0"] > 0, u["f0"] * (基F0 / max(med, 50.0)), 0.0)
        pos = j / max(n - 1, 1)
        u["f0"] = u["f0"] * (2.0 ** ((-1.5 * pos) / 12.0))
        u["E"] = u["E"] * (1.0 - 0.35 * pos)
        u["包络"] = 包络_由LPC(u["A"])
    seq_f0 = []; seq_E = []; seq_NR = []; seq_M = []
    prev = None
    for u in 单元:
        if prev is not None:
            w = np.linspace(0, 1, 插值帧 + 2)[1:-1].reshape(-1, 1)
            seq_f0.append((1 - w).ravel() * prev["f0"][-1] + w.ravel() * u["f0"][0])
            seq_E.append((1 - w).ravel() * prev["E"][-1] + w.ravel() * u["E"][0])
            seq_NR.append((1 - w).ravel() * prev["NR"][-1] + w.ravel() * u["NR"][0])
            seq_M.append((1 - w) * prev["包络"][-1][None, :] + w * u["包络"][0][None, :])
        seq_f0.append(u["f0"]); seq_E.append(u["E"]); seq_NR.append(u["NR"]); seq_M.append(u["包络"])
        prev = u
    f0 = np.concatenate(seq_f0); E = np.concatenate(seq_E); NR = np.concatenate(seq_NR)
    M = np.concatenate(seq_M, axis=0)
    return dict(f0=f0, E=E, NR=NR, M=M, T=len(f0)), 状态


def 激励(f0帧, NR帧, 种子=20261022):
    T = len(f0帧); N = T * 帧样本
    f0s = np.interp(np.arange(N) / 帧样本, np.arange(T) + 0.5, f0帧)
    ph = np.mod(np.cumsum(f0s) / SR, 1.0)
    流, 导 = 器.声门表(0.62, 0.42)
    源 = np.interp(ph * 器._表长, np.arange(器._表长), 导)
    rng = np.random.RandomState(种子)
    噪 = rng.randn(N)
    噪hp = np.append(噪[0], 噪[1:] - 噪[:-1])
    NRs = np.interp(np.arange(N) / 帧样本, np.arange(T) + 0.5, NR帧)
    return 源 * 0.6 + 噪hp * NRs * 0.8


def 合成(序列, 迭代=24):
    T = 序列["T"]
    源 = 激励(序列["f0"], 序列["NR"])
    S = librosa.stft(源, n_fft=1024, hop_length=220, win_length=551, window="hann", center=True)
    T2 = S.shape[1]
    idx = np.clip(np.round(np.arange(T2) * 2).astype(int), 0, T - 1)
    M = np.exp(np.clip(序列["M"][idx], -30, 30)) * 序列["E"][idx][:, None]
    M = M.T
    相位 = np.angle(S)
    y = None
    for _ in range(int(迭代)):
        Y = M * np.exp(1j * 相位)
        y = librosa.istft(Y, hop_length=220, win_length=551, n_fft=1024, window="hann", center=True)
        y = np.nan_to_num(np.asarray(y, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        S2 = librosa.stft(y, n_fft=1024, hop_length=220, win_length=551, window="hann", center=True)
        if S2.shape[1] != T2:
            break
        相位 = np.angle(S2)
    p = float(np.abs(y).max())
    if p > 1e-9:
        y = y / p * 0.9
    return y.astype(np.float32)


def 主(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    print("载库（首次会建库，约 2–4 分钟）…", flush=True)
    库 = 载库()
    字表 = 建字表(库)
    print("库音节数", len(库), "字表", len(字表), "耗时", round(time.time() - t0, 1), flush=True)
    引 = 量.载引擎()
    报 = dict(工具="声学单元移植_v3.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              库音节数=len(库), 字表字数=len(字表), 插值帧=插值帧, 项=[])
    for 名, 文 in 句表.items():
        序列, 状态 = 合成序列(文, 库, 字表)
        if 序列 is None:
            报["项"].append(dict(样本=名, 失败="无单元")); continue
        y = 合成(序列)
        p = 出 / f"{名}.wav"; 写(p, y)
        sha = hashlib.sha256((y * 32767).astype(np.int16).tobytes()).hexdigest()[:16]
        q = 引["转写"](str(p))
        缺 = sum(1 for _, s in 状态 if s == "缺")
        报["项"].append(dict(样本=名, 文本=文, 路径=str(p), 秒=round(len(y) / SR, 3),
                            帧=序列["T"], 缺=缺, sha=sha, 转写=q, CER=round(量.CER(文, q), 4)))
        print("|", 名, "CER=%.3f" % 报["项"][-1]["CER"], "缺", 缺, "|", q[:30], flush=True)
    r0 = [r for r in 报["项"] if "CER" in r]
    报["均值CER"] = round(float(np.mean([r["CER"] for r in r0])), 4) if r0 else None
    报["耗时秒"] = round(time.time() - t0, 1)
    (出 / "V3_读数.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("均值", 报["均值CER"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(主())
