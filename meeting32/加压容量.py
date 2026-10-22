# -*- coding: utf-8 -*-
"""meeting32/加压容量.py —— M3 判据天花板补测：重位翻下 1024 vs 10240 硬对照。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "meeting28"))


def 测(D: int, ps=(0.20, 0.30, 0.40, 0.45), K: int = 2000, 问数: int = 500) -> dict:
    import lang_codec
    lang_codec.D = D
    from lang_codec import 语言编解码器
    sys.path.insert(0, os.path.join(HERE))
    from 维度十倍 import 取语料
    语料 = 取语料()[:K]
    codec = 语言编解码器(seed=216)
    for s in 语料:
        codec.注册段(s)
    M = np.stack(codec._段向量)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    out = dict(D=D)
    for p in ps:
        rng = np.random.default_rng(500 + int(p * 100))
        对 = rng.integers(0, len(语料), 问数)
        翻 = 1.0 - 2.0 * (rng.random((问数, D)) < p)
        Q = M[对] * 翻
        Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
        检 = np.argmax(Q @ Mn.T, axis=1)
        out["p%.2f" % p] = round(float(np.mean(检 == 对)), 4)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    结果 = {}
    for D in (1024, 2560, 10240):
        r = 测(D)
        结果[str(D)] = r
        print("D=%d: %s" % (D, r), flush=True)
    b, x = 结果["1024"], 结果["10240"]
    文 = ["# 加压容量对照（K=2000，问500）\n"]
    for p in ("p0.20", "p0.30", "p0.40", "p0.45"):
        判 = x[p] >= b[p]
        文.append("- %s: D=1024 → %s, D=10240 → %s [%s]\n"
                  % (p, b[p], x[p], "PASS" if 判 else "FAIL"))
    print("".join(文))
    with open(os.path.join(HERE, "加压容量_判读.md"), "w", encoding="utf-8") as f:
        f.writelines(文)
    with open(os.path.join(HERE, "加压容量.json"), "w", encoding="utf-8") as f:
        json.dump(结果, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
