# -*- coding: utf-8 -*-
"""meeting71/实施/准备_P6v5数据.py —— 路线 1：把新扩窗口并入训练集。

P6v4_数据.pkl（346 条 / 3.9 h）+ 若干新 npz 目录（meta_扩容.json）→ P6v5_数据.pkl。
新样本：mel 用 npz 自带的；F0/E/NR 在 wav 上算（与 P6v4 同口径）。
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
E = Path(r"<path>\AHI_m71")
SR = 22050


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


耳 = 载("耳蜗前端_S0_2", M71.parent / "meeting69" / "实施" / "耳蜗前端_S0_2.py")


def 找wav(名):
    for root in (E / "切_8h" / "片段", E / "切_9h" / "片段", E / "切_10h" / "片段",
                 E / "切_11h" / "片段", E / "切_12h" / "片段", E / "切_13h" / "片段"):
        p = root / 名
        if p.exists():
            return p
    return None


def 处理(npz目录, meta路, 前缀):
    meta = json.loads(Path(meta路).read_text(encoding="utf-8"))
    记录 = {r["id"]: r for r in meta["记录"]}
    件 = []
    for f in sorted(npz目录.glob("*.npz")):
        sid = int(f.stem)
        if sid not in 记录:
            continue
        wav = Path(记录[sid]["wav"])
        if not wav.exists():
            wav = 找wav(Path(记录[sid]["wav"]).name)
        if wav is None or not Path(wav).exists():
            continue
        z = np.load(f, allow_pickle=True)
        ml = np.asarray(z["mel"], np.float32)
        Tm = len(ml)
        x = 耳.读wav(wav)
        t, f0 = 耳.F0轨迹(x)
        grid = (np.arange(Tm) + 0.5) * 0.01
        f0g = np.interp(grid, t, np.asarray(f0, np.float64)).astype(np.float32)
        Ee = np.array([np.sqrt(np.mean(np.asarray(x[i * 220:(i + 1) * 220], np.float64) ** 2))
                       for i in range(Tm)]).astype(np.float32)
        mp = np.exp(np.clip(ml, -40, 40))
        flat = np.exp(np.mean(np.log(mp + 1e-10), 1)) / (np.mean(mp, 1) + 1e-10)
        NR = np.clip((flat - 0.10) / 0.60, 0.0, 1.0).astype(np.float32)
        声, 韵, 调id, 帧数 = z["声母"], z["韵母"], z["调"], z["音节帧数"]
        n = min(len(声), len(帧数))
        toks = np.stack([声[:n], 韵[:n], 调id[:n]], 1).astype(np.int64)
        durs = 帧数[:n].astype(np.int64)
        T = min(int(durs.sum()), Tm, len(f0g))
        件.append(dict(sid=前缀 + sid, toks=toks, durs=durs, mel=ml[:T], f0=f0g[:T], E=Ee[:T], NR=NR[:T]))
    return 件


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    with open(E / "P6v4_数据.pkl", "rb") as f:
        d = pickle.load(f)
    件 = d["件"]; 字表 = d["字表"]
    print("旧", len(件), "条", flush=True)
    for 名, 前缀 in (("数据npz_8_14", 20000),):
        目录 = E / 名 / "样本"; meta = E / 名 / "meta_扩容.json"
        if 目录.exists() and meta.exists():
            新 = 处理(目录, meta, 前缀)
            print(名, len(新), "条", flush=True)
            件 += 新
    from collections import defaultdict
    时长 = defaultdict(list)
    for x in 件:
        for t, dn in zip(x["toks"], x["durs"]):
            时长[(int(t[0]), int(t[1]), int(t[2]))].append(int(dn))
    时长表 = {k: int(np.median(v)) for k, v in 时长.items()}
    全中位 = int(np.median([dn for v in 时长.values() for dn in v]))
    with open(E / "P6v5_数据.pkl", "wb") as f:
        pickle.dump(dict(件=件, 字表=字表, 时长表=时长表, 全中位=全中位), f, protocol=4)
    print("合并", len(件), "条 总帧", sum(len(x["f0"]) for x in 件),
          "≈小时", round(sum(len(x["f0"]) for x in 件) / 360000, 2),
          "耗时", round(time.time() - t0, 1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(主())
