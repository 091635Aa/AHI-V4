# -*- coding: utf-8 -*-
"""meeting69/实施/跑_精修v1.py —— 预注册 `26` 的臂生成 + 客观读数（先写后跑）。

产出：`产物/精修v1/*.wav`、`产物/精修v1/读数_精修v1.json`
指标：电音比（90–110/190–210 Hz 调制能量占比）、浊帧 HNR 中位、F0 中位、时长、前后半段电音比。
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M69 = HERE.parent
出 = M69 / "产物" / "精修v1"
SR = 22050

import importlib.util


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


精 = 载("音频线_v1_精修", HERE / "音频线_v1_精修.py")
耳 = 精.耳
声 = 精.声
协 = 载("合成_S0刺激", HERE / "合成_S0刺激.py")

句子 = [
    "今天天气很好我听见了自己的声音我想再说一遍",
    "早上起来我先看了看窗外的光",
    "我不太确定这句话你听清了没有",
    "如果听不清也没关系我再说一遍",
]


def 电音比(x):
    """帧率调制比：90–110 + 190–210 Hz 占 1–400 Hz 的调制能量比例。"""
    import librosa
    S = np.abs(librosa.stft(np.asarray(x, np.float32), n_fft=512, hop_length=128))
    L = np.log(S + 1e-6)
    L = L - L.mean(axis=1, keepdims=True)
    w = np.hanning(L.shape[1])[None, :]
    F = np.abs(np.fft.rfft(L * w, axis=1)) ** 2
    fr = np.fft.rfftfreq(L.shape[1], d=128.0 / SR)

    def E(a, b):
        return float(F[:, (fr >= a) & (fr < b)].sum())

    return (E(90, 110) + E(190, 210)) / max(E(1, 400), 1e-12)


def 电音比_半(x):
    n = len(x) // 2
    return dict(前半=round(电音比(x[:n]), 6), 后半=round(电音比(x[n:]), 6))


def HNR(x):
    t, f0 = 耳.F0轨迹(np.asarray(x, np.float32))
    W = int(0.040 * SR)
    vals = []
    for ti, fi in zip(t, f0):
        if fi <= 0:
            continue
        c = int(round(float(ti) * SR))
        a = c - W // 2
        if a < 0 or a + W > len(x):
            continue
        w = np.asarray(x[a:a + W], np.float64)
        w = w - w.mean()
        能 = float(np.dot(w, w)) + 1e-12
        lag = int(round(SR / float(fi)))
        best = 0.0
        for d in range(max(1, lag - 2), lag + 3):
            if d >= len(w):
                break
            r = float(np.dot(w[:-d], w[d:]) / 能)
            if r > best:
                best = r
        r = min(max(best, 1e-6), 0.999999)
        vals.append(10.0 * math.log10(r / (1.0 - r)))
    return float(np.median(vals)) if vals else float("nan")


def F0中位(x):
    t, f0 = 耳.F0轨迹(np.asarray(x, np.float32))
    v = [float(f) for f in f0 if f > 0]
    return float(np.median(v)) if v else float("nan")


def 记一分(x):
    return dict(时长秒=round(len(x) / SR, 3), 电音比=round(电音比(x), 6),
                电音比_前后=电音比_半(x), HNR_dB=round(HNR(x), 3), F0中位Hz=round(F0中位(x), 1))


def 写(名, x):
    路 = 出 / 名
    协.写wav(路, x)
    return str(路), hashlib.sha256(路.read_bytes()).hexdigest()


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    出.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    报 = dict(工具="跑_精修v1.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              预注册="meeting69/26_预注册_音频线精修v1.md", 句子=句子, 项=[], 移调=[])

    # ── 平静：A0/A1/A2/A3 × 4 句 ──
    for ai, 文 in enumerate(句子):
        for 臂 in ("A0", "A1", "A2", "A3"):
            r = 精.合成_精修(文, "平静", "正常", 臂=臂)
            路, sha = 写(f"句{ai}_{臂}.wav", r["波形"])
            项 = dict(句=ai, 臂=臂, 路径=路, sha256=sha)
            项.update(记一分(r["波形"]))
            报["项"].append(项)
            print("句%d %s 电音比=%.6f HNR=%.2f F0=%.1f 时长=%.2f" %
                  (ai, 臂, 项["电音比"], 项["HNR_dB"], 项["F0中位Hz"], 项["时长秒"]), flush=True)

    # ── 移调臂（高兴/低落，句0）：A0-P0（v0 原样） / A2-P1（纯重采样） / A2-P2（不移调） ──
    for 情 in ("高兴", "低落"):
        for 臂, 移 in (("A0", "P0"), ("A2", "P1"), ("A2", "P2")):
            r = 精.合成_精修(句子[0], 情, "正常", 臂=臂, 移调=移)
            路, sha = 写(f"{情}_{臂}_{移}.wav", r["波形"])
            项 = dict(情绪=情, 臂=臂, 移调=移, 路径=路, sha256=sha)
            项.update(记一分(r["波形"]))
            报["移调"].append(项)
            print("%s %s-%s 电音比=%.6f HNR=%.2f F0=%.1f 时长=%.2f" %
                  (情, 臂, 移, 项["电音比"], 项["HNR_dB"], 项["F0中位Hz"], 项["时长秒"]), flush=True)

    # ── F5：A0 是否逐位复现 v0（句0 平静） ──
    v0 = M69 / "产物" / "S2A" / "自声" / "平静.wav"
    a0 = [x for x in 报["项"] if x["句"] == 0 and x["臂"] == "A0"][0]
    报["F5_A0复现v0"] = dict(v0_sha=hashlib.sha256(v0.read_bytes()).hexdigest(),
                            A0_sha=a0["sha256"], 同=bool(hashlib.sha256(v0.read_bytes()).hexdigest() == a0["sha256"]))
    报["耗时秒"] = round(time.time() - t0, 1)
    (出 / "读数_精修v1.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("F5_A0复现v0:", 报["F5_A0复现v0"]["同"], "耗时", 报["耗时秒"], "s")
    return 0


if __name__ == "__main__":
    sys.exit(主())
