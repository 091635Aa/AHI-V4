# -*- coding: utf-8 -*-
"""meeting71/实施/帧检索移植_v5.py —— 帧级「检索特征点→复制」（所有者思路，不训练）。

不做单元拼接：v13 mel 给「内容/时间」，逐帧（10 ms）在 D2 真人库里找**最像的真实帧**，
复制它的 24 阶 LPC 包络/F0/能量/噪声，在自相关域平滑后驱动器官发声。
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
D2目录 = M67 / "数据" / "子集_D2_le30min" / "样本"
音频根 = Path(r"<path>\SpeakerA7代模型训练数据")
库缓存 = 出 / "库_帧.pkl"
句表 = {
    "V5主句": "今天天气很好我听见了自己的声音我想再说一遍",
    "V5新句1": "今天的风很轻我慢慢说给你听",
    "V5新句2": "窗外的雨停了我该休息一会",
    "V5新句3": "你先别走我还想再说两句",
}
参数 = dict(OQ=0.62, OQ1=0.42, 噪声尺度=0.40, 去加重=0.97)


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


器 = 载("器官v2", M70 / "实施" / "器官v2.py")
精 = 载("音频线_v1_精修", M71.parent / "meeting69" / "实施" / "音频线_v1_精修.py")
量 = 载("量具_v9", M64 / "量具_v9.py")


def 写(p, a):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def logmel(x):
    S = np.abs(librosa.stft(np.asarray(x, np.float64), n_fft=1024, hop_length=220,
                            win_length=551, window="hann", center=True)) ** 2
    M = librosa.feature.melspectrogram(S=S, sr=SR, n_fft=1024, n_mels=80, fmin=0.0, fmax=SR / 2.0)
    return np.log(np.maximum(M, 1e-10)).T                # (T,80)


def 建库():
    meta = json.loads((M66 / "产物" / "数据_对齐asr" / "meta.json").read_text(encoding="utf-8"))
    样本 = {r["id"]: r for r in meta["样本"]}
    mfccs = []; As = []; f0s = []; Es = []; NRs = []
    for f in sorted(D2目录.glob("*.npz")):
        sid = int(f.stem)
        x = 器.耳.读wav(音频根 / 样本[sid]["wav"])
        c = 器.分析控制(x)                                 # 5 ms 帧
        ml = logmel(x)                                     # 10 ms 帧
        mf = librosa.feature.mfcc(S=ml.T, n_mfcc=13).T      # (T,13)
        T = min(len(mf), c["T"] // 2)
        mfccs.append(mf[:T])
        As.append(c["A"][: 2 * T: 2])                      # 取每 10 ms 的第一帧
        f0s.append(c["f0"][: 2 * T: 2])
        Es.append(c["E"][: 2 * T: 2])
        NRs.append(c["NR"][: 2 * T: 2])
    db = dict(mfcc=np.concatenate(mfccs, 0), A=np.concatenate(As, 0),
              f0=np.concatenate(f0s, 0), E=np.concatenate(Es, 0), NR=np.concatenate(NRs, 0))
    return db


def 载库():
    出.mkdir(parents=True, exist_ok=True)
    if 库缓存.exists():
        with open(库缓存, "rb") as f:
            return pickle.load(f)
    db = 建库()
    with open(库缓存, "wb") as f:
        pickle.dump(db, f, protocol=4)
    return db


def 自相关_由LPC(A, 阶=24, n_fft=1024):
    out = np.empty((len(A), 阶 + 1), np.float64)
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


def 主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    print("载库…", flush=True)
    db = 载库()
    print("帧库", db["mfcc"].shape, flush=True)
    # 归一化
    mu = db["mfcc"].mean(0); sd = db["mfcc"].std(0) + 1e-6
    dbn = (db["mfcc"] - mu) / sd
    引 = 量.载引擎()
    报 = dict(工具="帧检索移植_v5.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"),
              帧库=int(db["mfcc"].shape[0]), 项=[])
    for 名, 文 in 句表.items():
        mel, _, _ = 精.推理mel(文, 24)
        ml = np.asarray(mel, np.float64)                    # (T,80) log-mel
        mf = librosa.feature.mfcc(S=ml.T, n_mfcc=13).T
        mfn = (mf - mu) / sd
        # 逐帧最近邻（余弦/欧氏），分块
        K = 5
        idx = np.zeros((len(mfn), K), np.int64)
        for s in range(0, len(mfn), 256):
            blk = mfn[s:s + 256]
            d = ((blk[:, None, :] - dbn[None, :, :]) ** 2).sum(-1)   # (b, N)
            idx[s:s + 256] = np.argsort(d, axis=1)[:, :K]
        # 简化选择：取最近（多候选留给下一版 Viterbi）
        sel = idx[:, 0]
        # 控制：10 ms 帧 → 两个 5 ms 帧
        f0 = np.repeat(db["f0"][sel], 2)
        E = np.repeat(db["E"][sel], 2)
        NR = np.repeat(db["NR"][sel], 2)
        r = 自相关_由LPC(db["A"][sel])
        r5 = np.repeat(r, 2, axis=0)
        # 时间平滑：EMA on r / f0 median / E
        for i in range(1, len(r5)):
            r5[i] = 0.55 * r5[i] + 0.45 * r5[i - 1]
        f0s = np.copy(f0)
        for i in range(len(f0s)):
            lo = max(0, i - 2); hi = min(len(f0s), i + 3)
            v = f0[lo:hi][f0[lo:hi] > 0]
            f0s[i] = float(np.median(v)) if len(v) else 0.0
        E = np.convolve(E, np.ones(3) / 3, mode="same")
        A = np.stack([levinson(rr, 24) for rr in r5])
        c = dict(T=len(f0s), f0=f0s, 浊=f0s > 0, F=np.zeros((len(f0s), 4)),
                 B=np.zeros((len(f0s), 4)), E=np.maximum(E, 1e-5), NR=NR, A=A)
        y = 器.合成(c, 模式="全极点", **参数)
        p = 出 / f"{名}.wav"; 写(p, y)
        sha = hashlib.sha256((y * 32767).astype(np.int16).tobytes()).hexdigest()[:16]
        q = 引["转写"](str(p))
        报["项"].append(dict(样本=名, 文本=文, 路径=str(p), 秒=round(len(y) / SR, 3),
                            帧=c["T"], sha=sha, 转写=q, CER=round(量.CER(文, q), 4)))
        print("|", 名, "CER=%.3f" % 报["项"][-1]["CER"], "|", q[:30], flush=True)
    r0 = [x for x in 报["项"] if "CER" in x]
    报["均值CER"] = round(float(np.mean([x["CER"] for x in r0])), 4) if r0 else None
    报["耗时秒"] = round(time.time() - t0, 1)
    (出 / "V5_读数.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("均值", 报["均值CER"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(主())
