# -*- coding: utf-8 -*-
"""meeting71/实施/准备_P6v2数据.py —— P6v2（mel 目标）训练数据，D4 全量 167 条。

每样本：tokens(N,3)/durs(N,)/mel(T,80)/f0(T)/E(T)/NR(T)/文；另存字表。
只算 YIN F0 + mel + 能量 + 谱平坦度（不做 LPC），约 2–3 min。
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import librosa

HERE = Path(__file__).resolve().parent
M71 = HERE.parent
M69 = M71.parent / "meeting69"
M66 = M71.parent / "meeting66"
产 = M71 / "产物" / "P6v2"
SR = 22050
npz目录 = M66 / "产物" / "数据_对齐asr" / "样本"
音频根 = Path(r"<path>\SpeakerA7代模型训练数据")


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


耳 = 载("耳蜗前端_S0_2", M69 / "实施" / "耳蜗前端_S0_2.py")


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
    t0 = time.time()
    for f in sorted(npz目录.glob("*.npz")):
        sid = int(f.stem)
        z = np.load(f, allow_pickle=True)
        x = 耳.读wav(音频根 / 样本[sid]["wav"])
        # mel 10 ms
        S = np.abs(librosa.stft(np.asarray(x, np.float64), n_fft=1024, hop_length=220,
                                win_length=551, window="hann", center=True)) ** 2
        M = librosa.feature.melspectrogram(S=S, sr=SR, n_fft=1024, n_mels=80, fmin=0.0, fmax=SR / 2.0)
        ml = np.log(np.maximum(M, 1e-10)).T.astype(np.float32)          # (Tm,80)
        Tm = len(ml)
        # F0：整条 YIN → 10 ms 网格
        t, f0 = 耳.F0轨迹(x)
        grid = (np.arange(Tm) + 0.5) * 0.01
        f0g = np.interp(grid, t, np.asarray(f0, np.float64)).astype(np.float32)
        # E（10 ms RMS）
        E = np.array([np.sqrt(np.mean(np.asarray(x[i * 220:(i + 1) * 220], np.float64) ** 2))
                      for i in range(Tm)]).astype(np.float32)
        # NR：mel 谱平坦度
        mp = np.exp(np.clip(ml, -40, 40))
        flat = np.exp(np.mean(np.log(mp + 1e-10), 1)) / (np.mean(mp, 1) + 1e-10)
        NR = np.clip((flat - 0.10) / 0.60, 0.0, 1.0).astype(np.float32)
        声, 韵, 调id, 帧数 = z["声母"], z["韵母"], z["调"], z["音节帧数"]
        n = min(len(声), len(帧数))
        chs = [ch for ch in 样本[sid]["文"] if "\u4e00" <= ch <= "\u9fff"]
        toks = np.stack([声[:n], 韵[:n], 调id[:n]], 1).astype(np.int64)
        durs = 帧数[:n].astype(np.int64)
        T = min(int(durs.sum()), Tm, len(f0g))
        for i in range(min(n, len(chs))):
            表.setdefault(chs[i], []).append((int(声[i]), int(韵[i]), int(调id[i])))
        件.append(dict(sid=sid, toks=toks, durs=durs, mel=ml[:T], f0=f0g[:T], E=E[:T], NR=NR[:T]))
        if len(件) % 20 == 0:
            print("|", len(件), "样本", round(time.time() - t0, 1), "s", flush=True)
    字表 = {ch: Counter(v).most_common(1)[0][0] for ch, v in 表.items()}
    with open(产 / "数据.pkl", "wb") as fp:
        pickle.dump(dict(件=件, 字表=字表), fp, protocol=4)
    print("样本", len(件), "字表", len(字表), "总帧", sum(len(x["f0"]) for x in 件),
          "耗时", round(time.time() - t0, 1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(主())
