# -*- coding: utf-8 -*-
"""meeting71/实施/准备_P6数据.py —— 控制生成器训练数据（10 ms 帧网格）。

每样本：tokens(N,3) 声母/韵母/调、durs(N,) 10 ms 帧、目标 r(T,25)/f0(T)/logE(T)/NR(T)、文；另存字表。
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M71 = HERE.parent
M70 = M71.parent / "meeting70"
M67 = M71.parent / "meeting67"
M66 = M71.parent / "meeting66"
产 = M71 / "产物" / "P6"
SR = 22050
D2目录 = M67 / "数据" / "子集_D2_le30min" / "样本"
音频根 = Path(r"<path>\SpeakerA7代模型训练数据")


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


器 = 载("器官v2", M70 / "实施" / "器官v2.py")


def 自相关_由LPC(A, 阶=24, n_fft=1024):
    out = np.empty((len(A), 阶 + 1), np.float64)
    for i, a in enumerate(A):
        H = np.fft.rfft(a, n=n_fft)
        P = 1.0 / (np.abs(H) ** 2 + 1e-12)
        r = np.fft.irfft(P, n=n_fft)[: 阶 + 1]
        out[i] = r / (r[0] + 1e-12)
    return out


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    产.mkdir(parents=True, exist_ok=True)
    meta = json.loads((M66 / "产物" / "数据_对齐asr" / "meta.json").read_text(encoding="utf-8"))
    样本 = {r["id"]: r for r in meta["样本"]}
    件 = []
    from collections import Counter
    表 = {}
    for f in sorted(D2目录.glob("*.npz")):
        sid = int(f.stem)
        z = np.load(f, allow_pickle=True)
        x = 器.耳.读wav(音频根 / 样本[sid]["wav"])
        c = 器.分析控制(x)
        r = 自相关_由LPC(c["A"][::2])
        f0 = c["f0"][::2].astype(np.float32)
        logE = np.log(np.maximum(c["E"][::2], 1e-6)).astype(np.float32)
        NR = c["NR"][::2].astype(np.float32)
        T = min(len(r), len(f0), len(logE), len(NR))
        声, 韵, 调id, 帧数 = z["声母"], z["韵母"], z["调"], z["音节帧数"]
        n = min(len(声), len(帧数))
        chs = [ch for ch in 样本[sid]["文"] if "\u4e00" <= ch <= "\u9fff"]
        toks = np.stack([声[:n], 韵[:n], 调id[:n]], 1).astype(np.int64)
        durs = 帧数[:n].astype(np.int64)
        # 保证帧数之和与 T 一致
        if int(durs.sum()) > T:
            acc = 0
            for i in range(len(durs)):
                if acc + int(durs[i]) > T:
                    durs[i] = max(1, T - acc)
                    durs[i + 1:] = 0
                    break
                acc += int(durs[i])
        for i in range(min(n, len(chs))):
            表.setdefault(chs[i], []).append((int(声[i]), int(韵[i]), int(调id[i])))
        件.append(dict(sid=sid, toks=toks, durs=durs, r=r[:T].astype(np.float32),
                     f0=f0[:T], logE=logE[:T], NR=NR[:T]))
        print("|", sid, "音节", n, "帧", T, flush=True)
    字表 = {ch: Counter(v).most_common(1)[0][0] for ch, v in 表.items()}
    with open(产 / "数据.pkl", "wb") as fp:
        pickle.dump(dict(件=件, 字表=字表), fp, protocol=4)
    print("样本", len(件), "字表", len(字表), "总帧", sum(len(x["f0"]) for x in 件), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(主())
