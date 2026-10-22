# -*- coding: utf-8 -*-
"""meeting70/实施/换词_H4.py —— H4 换词探针：底句真实控制轨迹里换 1 个词（2 音节）。

判据见 07_预注册_H4_换词.md；零训练；参数固定 04 最优臂 E。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M70 = HERE.parent
M66 = M70.parent / "meeting66"
M67 = M70.parent / "meeting67"
M64 = M70.parent / "meeting64" / "实施"
出 = M70 / "产物" / "H4换词"
SR = 22050
参数 = dict(OQ=0.62, OQ1=0.42, 噪声尺度=0.40, 去加重=0.97)
音频根 = Path(r"<path>\SpeakerA7代模型训练数据")


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


器 = 载("器官v2", HERE / "器官v2.py")
量 = 载("量具_v9", M64 / "量具_v9.py")


def 写(p, a):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def 元数据():
    meta = json.loads((M66 / "产物" / "数据_对齐asr" / "meta.json").read_text(encoding="utf-8"))
    样本 = {r["id"]: r for r in meta["样本"]}
    对齐 = json.loads((M67 / "产物" / "P2" / "对齐证据.json").read_text(encoding="utf-8"))
    偏移 = {r["id"]: int(r["偏移帧"]) for r in 对齐["per_sample"]}
    return 样本, 偏移


def 解析文本(文):
    return [c for c in 文 if "\u4e00" <= c <= "\u9fff"]


def 找词(样本, 词):
    for sid, r in 样本.items():
        chs = 解析文本(r["文"])
        idx = "".join(chs).find(词)
        if idx >= 0:
            return sid, idx
    return None, None


def 取窗控制(sid, 起始帧, 窗帧数, 样本, 偏移, 缓存):
    """分析 wav 上 [起始帧, 起始帧+窗帧数)（10 ms 帧、含偏移）的器官控制。"""
    x = 器.耳.读wav(音频根 / 样本[sid]["wav"])
    a = max(0, int((偏移[sid] + 起始帧) * 220))
    b = min(len(x), int((偏移[sid] + 起始帧 + 窗帧数) * 220))
    片 = x[a:b]
    key = (sid, a, b)
    if key not in 缓存:
        缓存[key] = 器.分析控制(片)
    return 缓存[key], a


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    出.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    样本, 偏移 = 元数据()
    引 = 量.载引擎()
    缓存 = {}
    # 底句：样本 57，切片 10–15 s（帧 1000–1500）
    base_sid, b0, bwin = 57, 1000, 500
    z = np.load(M67 / "数据" / "子集_D0_le5min" / "样本" / f"{base_sid:04d}.npz", allow_pickle=True) \
        if (M67 / "数据" / "子集_D0_le5min" / "样本" / f"{base_sid:04d}.npz").exists() else None
    # 用全量对齐数据里的帧数（样本 57 在 D4，不在 D0）
    z = np.load(M66 / "产物" / "数据_对齐asr" / "样本" / f"{base_sid:04d}.npz", allow_pickle=True)
    帧数 = z["音节帧数"]; 起 = np.concatenate([[0], np.cumsum(帧数)[:-1]])
    chs = 解析文本(样本[base_sid]["文"])
    # 完全落进切片的音节
    inside = [(i, chs[i], int(起[i]), int(起[i] + 帧数[i])) for i in range(min(len(chs), len(帧数)))
              if 起[i] >= b0 and 起[i] + 帧数[i] <= b0 + bwin]
    底文本 = "".join(c for _, c, _, _ in inside)
    报 = dict(工具="换词_H4.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 底样本=base_sid,
              底文本=底文本, 项=[])
    print("底文本:", 底文本, flush=True)
    if len(inside) < 5:
        print("底句音节不足"); return 1
    # 试例：在底文本里选第 1 个词（2 音节）替换成语料里的候选词
    试例 = []
    for 候选 in ("很好", "不错", "可以", "今天", "我们"):
        rsid, ridx = 找词(样本, 候选)
        if rsid is None or rsid == base_sid:
            continue
        # 底句替换位置：取中间一个 2 音节（第 2–3 个），避免句首句尾
        if len(inside) < 4:
            continue
        p = 1 if len(inside) >= 5 else 0
        span = inside[p:p + 2]
        if "".join(c for _, c, _, _ in span) == 候选:
            p += 2
            span = inside[p:p + 2]
        试例.append((候选, rsid, ridx, span))
        if len(试例) >= 2:
            break
    if not 试例:
        print("找不到替换词"); return 1
    for k, (候选, rsid, ridx, span) in enumerate(试例):
        # 底句控制（10 ms 帧 → 5 ms 控制帧）
        cbase, a0 = 取窗控制(base_sid, b0, bwin, 样本, 偏移, 缓存)
        F0 = cbase["f0"].copy(); 浊 = cbase["浊"].copy(); F = cbase["F"].copy(); B = cbase["B"].copy()
        E = cbase["E"].copy(); NR = cbase["NR"].copy(); A = cbase["A"].copy()
        # 替换词控制
        zr = np.load(M66 / "产物" / "数据_对齐asr" / "样本" / f"{rsid:04d}.npz", allow_pickle=True)
        帧r = zr["音节帧数"]; 起r = np.concatenate([[0], np.cumsum(帧r)[:-1]])
        wr0 = int(起r[ridx]); wr1 = int(起r[ridx] + 帧r[ridx] + (帧r[ridx + 1] if ridx + 1 < len(帧r) else 0))
        cr, ar0 = 取窗控制(rsid, max(0, wr0 - 40), (wr1 - wr0) + 80, 样本, 偏移, 缓存)
        # 替换词在窗内的控制帧：留 40 帧余量
        l0 = 40 * 2; l1 = l0 + (wr1 - wr0) * 2
        l1 = min(l1, cr["T"])
        rf0 = cr["f0"][l0:l1].copy(); r浊 = cr["浊"][l0:l1].copy()
        rF = cr["F"][l0:l1].copy(); rB = cr["B"][l0:l1].copy()
        rE = cr["E"][l0:l1].copy(); rNR = cr["NR"][l0:l1].copy(); rA = cr["A"][l0:l1].copy()
        # 底句切口（10 ms 帧 → 5 ms）
        a = int((span[0][2] - b0) * 2); b = int((span[-1][3] - b0) * 2)
        a = max(0, min(a, len(F0))); b = max(a + 2, min(b, len(F0)))
        # F0 平移到底句局部中位
        loc = F0[max(0, a - 40):min(len(F0), b + 40)]
        locv = loc[loc > 0]
        med_base = float(np.median(locv)) if len(locv) else 300.0
        rv = rf0[rf0 > 0]
        if len(rv):
            med_r = float(np.median(rv))
            rf0 = np.where(rf0 > 0, rf0 * (med_base / max(med_r, 60.0)), 0.0)
        # 能量缩放到局部 RMS
        le = E[max(0, a - 40):min(len(E), b + 40)]
        scale = float(np.median(le) / max(float(np.median(rE)), 1e-6))
        rE = rE * np.clip(scale, 0.3, 3.0)
        # 拼接 + 20 ms 交叉淡化（4 个 5 ms 帧）
        n = 4
        w = np.linspace(0, 1, n)[:, None]
        def 融(x1, x2):
            x1 = np.asarray(x1); x2 = np.asarray(x2)
            n2 = min(n, len(x1), len(x2))
            if n2 < 1:
                return np.concatenate([x1, x2])
            ww = w[:n2].reshape((n2,) + (1,) * (x1.ndim - 1))
            return np.concatenate([x1[:-n2], x1[-n2:] * (1 - ww) + x2[:n2] * ww, x2[n2:]])
        new = {}
        for k2, (left, mid, right) in {
            "f0": (F0[:a], rf0, F0[b:]), "浊": (浊[:a], r浊, 浊[b:]),
            "F": (F[:a], rF, F[b:]), "B": (B[:a], rB, B[b:]),
            "E": (E[:a], rE, E[b:]), "NR": (NR[:a], rNR, NR[b:]), "A": (A[:a], rA, A[b:]),
        }.items():
            new[k2] = 融(融(left, mid), right)
        # 浊音布尔对齐 f0 长度
        T = len(new["f0"])
        for k2 in ("浊", "F", "B", "E", "NR", "A"):
            if len(new[k2]) != T:
                new[k2] = new[k2][:T] if len(new[k2]) > T else np.concatenate(
                    [new[k2], np.repeat(new[k2][-1:], T - len(new[k2]), axis=0)])
        目标文本 = 底文本[:p] + 候选 + 底文本[p + 2:]
        c = dict(T=T, f0=new["f0"], 浊=new["浊"].astype(bool), F=new["F"], B=new["B"],
                 E=np.maximum(new["E"], 1e-5), NR=new["NR"], A=new["A"])
        y = 器.合成(c, 模式="全极点", **参数)
        p器 = 出 / f"H4_{k}_器官.wav"; 写(p器, y)
        sha1 = hashlib.sha256((y * 32767).astype(np.int16).tobytes()).hexdigest()
        y2 = 器.合成(c, 模式="全极点", **参数)
        sha2 = hashlib.sha256((y2 * 32767).astype(np.int16).tobytes()).hexdigest()
        r = 引["转写"](str(p器))
        cer = 量.CER(目标文本, r)
        报["项"].append(dict(试例=k, 替换=候选, 来源样本=rsid, 底文本=底文本, 目标文本=目标文本,
                           转写=r, CER=round(cer, 4), 词出现=(候选 in r), 两趟同=(sha1 == sha2)))
        print("|", k, 候选, "CER=%.3f" % cer, "词出现", 候选 in r, "|", r[:30], flush=True)
    报["耗时秒"] = round(time.time() - t0, 1)
    (M70 / "产物" / "H4_读数.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(主())
