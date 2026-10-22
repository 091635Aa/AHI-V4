# -*- coding: utf-8 -*-
"""meeting71/实施/音节控制移植_v1.py —— 特征点检索 + 移植 + 40 ms 控制域插值（不训练）。

所有者思路（meeting71/00）：搜索数据库特征点 → 复制 → 多组对比 → 填入兴趣特征。
P0：音节级特征点包 + 音节间 40 ms 线性插值（不是硬拼）；器官 v2.1 发声。
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

HERE = Path(__file__).resolve().parent
M71 = HERE.parent
M70 = M71.parent / "meeting70"
M67 = M71.parent / "meeting67"
M66 = M71.parent / "meeting66"
M64 = M71.parent / "meeting64" / "实施"
出 = M71 / "产物"
SR = 22050
D2目录 = M67 / "数据" / "子集_D2_le30min" / "样本"
音频根 = Path(r"<path>\SpeakerA7代模型训练数据")
参数 = dict(OQ=0.62, OQ1=0.42, 噪声尺度=0.40, 去加重=0.97)
插值帧 = 8          # 40 ms（5 ms 控制帧 × 8）
库缓存 = 出 / "库_D2.pkl"
句表 = {
    "M71主句": "今天天气很好我听见了自己的声音我想再说一遍",
    "M71新句1": "今天的风很轻我慢慢说给你听",
    "M71新句2": "窗外的雨停了我该休息一会",
    "M71新句3": "你先别走我还想再说两句",
}
声母表 = ["∅", "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s"]
韵母表 = ["a", "o", "e", "i", "u", "v", "er", "ii", "iii", "ai", "ei", "ao", "ou", "ia", "ie", "iao", "iou", "ua", "uo",
        "uai", "uei", "ve", "an", "en", "in", "uen", "vn", "ang", "eng", "ing", "ong", "iong", "ian", "uan", "uang",
        "van", "", "iang", "io"]


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


def 建库():
    meta = json.loads((M66 / "产物" / "数据_对齐asr" / "meta.json").read_text(encoding="utf-8"))
    样本 = {r["id"]: r for r in meta["样本"]}
    库 = []
    for f in sorted(D2目录.glob("*.npz")):
        sid = int(f.stem)
        z = np.load(f, allow_pickle=True)
        x = 器.耳.读wav(音频根 / 样本[sid]["wav"])
        c = 器.分析控制(x)
        声, 韵, 调id, 帧数 = z["声母"], z["韵母"], z["调"], z["音节帧数"]
        chs = [ch for ch in 样本[sid]["文"] if "\u4e00" <= ch <= "\u9fff"]
        cum = 0
        for i in range(len(声)):
            dur = int(帧数[i])
            a = cum * 2; b = (cum + dur) * 2
            cum += dur
            if b > c["T"] or b - a < 8:
                continue
            库.append(dict(sid=sid, sm=int(声[i]), ym=int(韵[i]), tone=int(调id[i]),
                           ch=chs[i] if i < len(chs) else "",
                           f0=c["f0"][a:b].astype(np.float32), F=c["F"][a:b].astype(np.float32),
                           B=c["B"][a:b].astype(np.float32), E=c["E"][a:b].astype(np.float32),
                           NR=c["NR"][a:b].astype(np.float32), A=c["A"][a:b].astype(np.float32)))
    return 库


def 载库():
    出.mkdir(parents=True, exist_ok=True)
    if 库缓存.exists():
        with open(库缓存, "rb") as f:
            return pickle.load(f)
    库 = 建库()
    with open(库缓存, "wb") as f:
        pickle.dump(库, f, protocol=4)
    return 库


def 建字表(库):
    from collections import Counter
    表 = {}
    for u in 库:
        ch = u["ch"]
        if ch:
            表.setdefault(ch, []).append((u["sm"], u["ym"], u["tone"]))
    return {ch: Counter(v).most_common(1)[0][0] for ch, v in 表.items()}


def 选单元(库, 键, ch):
    sm, ym, tone = 键
    for 条件 in ((lambda u: u["sm"] == sm and u["ym"] == ym and u["tone"] == tone and u["ch"] == ch),
                 (lambda u: u["sm"] == sm and u["ym"] == ym and u["tone"] == tone),
                 (lambda u: u["sm"] == sm and u["ym"] == ym)):
        候 = [u for u in 库 if 条件(u)]
        if 候:
            候.sort(key=lambda u: (abs(len(u["f0"]) - 48), u["sid"]))
            return 候[0]
    return None


def 键_of(文本, 字表):
    from pypinyin import lazy_pinyin, Style
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


def 插接(a, b, k):
    """在控制向量 a（末帧）与 b（首帧）之间插 k 帧线性过渡（保持原维度）。"""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    ash = a.shape
    av = a.reshape(-1); bv = b.reshape(-1)
    if av.size != bv.size:
        k = min(av.size, bv.size)
        av, bv = av[:k], bv[:k]
    w = np.linspace(0, 1, k + 2)[1:-1].reshape(-1, 1)
    out = (1 - w) * av[None, :] + w * bv[None, :]
    if ash == ():
        return out.reshape(k)
    return out.reshape((k,) + ash)


def 合成句(文本, 库, 字表, 基F0=300.0):
    keys = 键_of(文本, 字表)
    单元 = []
    状态 = []
    for ch, 键, 来源 in keys:
        u = 选单元(库, 键, ch)
        if u is None:
            状态.append((ch, "缺")); continue
        状态.append((ch, 来源))
        单元.append(u)
    if not 单元:
        return None, 状态
    # F0 归一：每个单元乘到基频附近，再句末下降
    n = len(单元)
    for j, u in enumerate(单元):
        v = u["f0"][u["f0"] > 0]
        med = float(np.median(v)) if len(v) else 基F0
        sc = 基F0 / max(med, 50.0)
        u["f0"] = np.where(u["f0"] > 0, u["f0"] * sc, 0.0)
        pos = j / max(n - 1, 1)
        u["f0"] = u["f0"] * (2.0 ** ((-1.5 * pos) / 12.0))
        u["E"] = u["E"] * (1.0 - 0.35 * pos)
    # 拼接（40 ms 插值）
    fields = ("f0", "F", "B", "E", "NR", "A")
    parts = {k: [] for k in fields}
    prev = None
    for u in 单元:
        if prev is not None:
            for k in fields:
                parts[k].append(插接(prev[k][-1], u[k][0], 插值帧))
        for k in fields:
            parts[k].append(u[k])
        prev = u
    c = {}
    for k in fields:
        v = np.concatenate(parts[k], axis=0)
        c[k] = v.astype(np.float64)
    T = len(c["f0"])
    c["T"] = T
    c["浊"] = (c["f0"] > 0) & (c["E"] > 1e-4)
    c["f0"] = np.where(c["浊"], c["f0"], 0.0)
    c["E"] = np.maximum(c["E"], 1e-5)
    # 轻微平滑（3 点）防抖
    for k in ("f0", "E", "NR"):
        c[k] = np.convolve(c[k], np.ones(3) / 3, mode="same")
    return c, 状态


def 主(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    出.mkdir(parents=True, exist_ok=True)
    print("载库…", flush=True)
    库 = 载库()
    字表 = 建字表(库)
    print("库音节数", len(库), "字表", len(字表), "耗时", round(time.time() - t0, 1), flush=True)
    引 = None
    报 = dict(工具="音节控制移植_v1.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              库音节数=len(库), 字表字数=len(字表), 插值帧=插值帧, 项=[])
    for 名, 文 in 句表.items():
        c, 状态 = 合成句(文, 库, 字表)
        if c is None:
            报["项"].append(dict(样本=名, 失败="无单元")); continue
        y = 器.合成(c, 模式="全极点", **参数)
        p = 出 / f"{名}.wav"; 写(p, y)
        sha = hashlib.sha256((y * 32767).astype(np.int16).tobytes()).hexdigest()[:16]
        缺 = sum(1 for _, s in 状态 if s == "缺")
        报["项"].append(dict(样本=名, 文本=文, 路径=str(p), 秒=round(len(y) / SR, 3),
                            帧=c["T"], 缺=缺, 有效字=len(状态) - 缺, sha=sha))
        print("|", 名, "秒", round(len(y) / SR, 2), "帧", c["T"], "缺", 缺, flush=True)
    # 读数
    if 引 is None:
        引 = 量.载引擎()
    for r in 报["项"]:
        if "文本" not in r:
            continue
        q = 引["转写"](r["路径"])
        r["转写"] = q
        r["CER"] = round(量.CER(r["文本"], q), 4)
        print("|", r["样本"], "CER=%.3f" % r["CER"], "|", q[:28], flush=True)
    r0 = [r for r in 报["项"] if "CER" in r]
    报["均值CER"] = round(float(np.mean([r["CER"] for r in r0])), 4) if r0 else None
    (M71 / "产物" / "M71_读数.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("均值", 报["均值CER"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(主())
