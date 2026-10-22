# -*- coding: utf-8 -*-
"""meeting71/实施/声学单元移植_v4.py —— 自相关域插值 + 器官全极点发声（修 v1–v3 失稳）。

v1：LPC 系数直接插值 → 全极点失稳；v2/v3：对数包络 + STFT 相位迭代 → 仍不可懂。
v4：把每帧 LPC 换成**自相关向量**（|1/A|² 的 IFFT），在自相关域插值（正定、稳定），
再 Levinson 回 LPC，用器官 v2.1 全极点模式发声（H3 已验证该模式听审「真人自然」）。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M71 = HERE.parent
M70 = M71.parent / "meeting70"
M64 = M71.parent / "meeting64" / "实施"
出 = M71 / "产物"
句表 = {
    "V4主句": "今天天气很好我听见了自己的声音我想再说一遍",
    "V4新句1": "今天的风很轻我慢慢说给你听",
    "V4新句2": "窗外的雨停了我该休息一会",
    "V4新句3": "你先别走我还想再说两句",
}


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


V3 = 载("v3", HERE / "声学单元移植_v3.py")
器 = V3.器
量 = V3.量
SR = V3.SR
写 = V3.写


def 自相关_由LPC(A, 阶=24, n_fft=1024):
    """A:(T,25) → r:(T,阶+1)（归一化 r[0]=1）。"""
    out = np.empty((len(A), 阶 + 1), np.float64)
    idx = np.arange(n_fft // 2 + 1)
    for i, a in enumerate(A):
        H = np.fft.rfft(a, n=n_fft)
        P = 1.0 / (np.abs(H) ** 2 + 1e-12)
        r = np.fft.irfft(P, n=n_fft)[: 阶 + 1]
        out[i] = r / (r[0] + 1e-12)
    return out


def levinson(r, p):
    a = np.zeros(p + 1); a[0] = 1.0
    e = float(r[0]) + 1e-12
    for i in range(1, p + 1):
        acc = float(r[i]) + (float(np.dot(a[1:i], r[i - 1:0:-1])) if i > 1 else 0.0)
        k = -acc / e
        if i > 1:
            a[1:i] = a[1:i] + k * a[i - 1:0:-1]
        a[i] = k
        e *= (1 - k * k)
        if e <= 1e-12:
            break
    return a


def 合成序列(文本, 库, 字表, 基F0=300.0):
    keys = V3.键_of(文本, 字表)
    单元 = []; 状态 = []
    for ch, 键, 来源 in keys:
        u = V3.选单元(库, 键, ch)
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
        u["r"] = 自相关_由LPC(u["A"])
    seq_f0 = []; seq_E = []; seq_NR = []; seq_r = []
    prev = None
    for u in 单元:
        if prev is not None:
            w = np.linspace(0, 1, V3.插值帧 + 2)[1:-1].reshape(-1, 1)
            seq_f0.append((1 - w).ravel() * prev["f0"][-1] + w.ravel() * u["f0"][0])
            seq_E.append((1 - w).ravel() * prev["E"][-1] + w.ravel() * u["E"][0])
            seq_NR.append((1 - w).ravel() * prev["NR"][-1] + w.ravel() * u["NR"][0])
            seq_r.append((1 - w) * prev["r"][-1][None, :] + w * u["r"][0][None, :])
        seq_f0.append(u["f0"]); seq_E.append(u["E"]); seq_NR.append(u["NR"]); seq_r.append(u["r"])
        prev = u
    f0 = np.concatenate(seq_f0); E = np.concatenate(seq_E); NR = np.concatenate(seq_NR)
    r = np.concatenate(seq_r, axis=0)
    A = np.stack([levinson(rr, 24) for rr in r])
    return dict(f0=f0, E=np.maximum(E, 1e-5), NR=NR, A=A, T=len(f0)), 状态


def 主(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    库 = V3.载库()
    字表 = V3.建字表(库)
    print("库音节数", len(库), "字表", len(字表), flush=True)
    引 = 量.载引擎()
    报 = dict(工具="声学单元移植_v4.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              库音节数=len(库), 字表字数=len(字表), 项=[])
    for 名, 文 in 句表.items():
        序列, 状态 = 合成序列(文, 库, 字表)
        if 序列 is None:
            报["项"].append(dict(样本=名, 失败="无单元")); continue
        c = dict(T=序列["T"], f0=序列["f0"], 浊=(序列["f0"] > 0) & (序列["E"] > 1e-4),
                 F=np.zeros((序列["T"], 4)), B=np.zeros((序列["T"], 4)),
                 E=序列["E"], NR=序列["NR"], A=序列["A"])
        y = 器.合成(c, 模式="全极点", OQ=0.62, OQ1=0.42, 噪声尺度=0.40, 去加重=0.97)
        p = 出 / f"{名}.wav"; 写(p, y)
        sha = hashlib.sha256((y * 32767).astype(np.int16).tobytes()).hexdigest()[:16]
        q = 引["转写"](str(p))
        缺 = sum(1 for _, s in 状态 if s == "缺")
        报["项"].append(dict(样本=名, 文本=文, 路径=str(p), 秒=round(len(y) / SR, 3),
                            帧=序列["T"], 缺=缺, sha=sha, 转写=q, CER=round(量.CER(文, q), 4)))
        print("|", 名, "CER=%.3f" % 报["项"][-1]["CER"], "缺", 缺, "|", q[:30], flush=True)
    r0 = [r for r in 报["项"] if "CER" in r]
    报["均值CER"] = round(float(np.mean([r["CER"] for r in r0])), 4) if r0 else None
    (出 / "V4_读数.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("均值", 报["均值CER"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(主())
