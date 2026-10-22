# -*- coding: utf-8 -*-
"""meeting67/实施/过滤_变长片段.py —— 变长片段质量门 + 去重（离线；确定性）。

指标/门槛沿用 `<path>\\SpeakerA7代模型训练数据\\audio_filter.py`（只读参考公式，不调用、不改）：
  SNR≥15 dB / 谱平坦度≤0.30 / 削波≤0.0005 / 爆音≤5 / 静音≤0.60
去重：log-mel 统计指纹 cos≥0.995 且包络相关≥0.92（对既有 167 条 + 本批内部）。

输入：切分目录（含 manifest.jsonl；可为多个 --目录）
输出：通过/manifest_通过.jsonl、淘汰.jsonl、质量报告.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

SR = 16000
SNR_MIN_DB = 15.0
FLAT_MAX = 0.30
CLIP_MAX = 0.0005
POP_MAX = 5
SILENCE_MAX = 0.60
DUP_COS = 0.995
DUP_ENV = 0.92
既有根 = Path(r"<path>\SpeakerA7代模型训练数据")


def 帧rms(y, frame=320, hop=160):
    n = 1 + max(0, (len(y) - frame) // hop)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt((y[idx] ** 2).mean(axis=1))


def 分析(y: np.ndarray) -> dict:
    rms = 帧rms(y)
    p = rms ** 2 + 1e-12
    floor = np.percentile(p, 15)
    act = p[p > floor * 10]
    if len(act) < 10:
        act = p[p >= np.median(p)]
    snr = 10 * np.log10(act.mean() / max(floor, 1e-12))
    silence = float((p < floor * 4).mean())
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=1024)) ** 2
    flat = float(librosa.feature.spectral_flatness(S=S + 1e-12).mean())
    clip = float((np.abs(y) >= 0.999).mean())
    pops = int((np.abs(np.diff(y)) > 0.5).sum())
    return dict(snr=float(snr), flat=flat, clip=clip, pops=pops, silence=silence,
                rms=float(np.sqrt((y ** 2).mean())))


def 指纹(y: np.ndarray):
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64, hop_length=1024, n_fft=1024)
    logmel = np.log(mel + 1e-10)
    fp = np.concatenate([logmel.mean(axis=1), logmel.std(axis=1)])
    fp = fp / (np.linalg.norm(fp) + 1e-12)
    env = 帧rms(y, 1024, 1024)
    env = np.interp(np.linspace(0, 1, 200), np.linspace(0, 1, len(env)), env)
    env = (env - env.mean()) / (env.std() + 1e-12)
    return fp, env


def 读单(path: Path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = x.mean(axis=1)
    if sr != SR:
        import math
        import scipy.signal as sps
        g = math.gcd(int(sr), SR)
        mono = sps.resample_poly(mono, SR // g, int(sr) // g).astype(np.float32)
    return mono


def 质量分(m):
    s_snr = np.clip((m["snr"] - SNR_MIN_DB) / 40.0, 0, 1)
    s_flat = np.clip((FLAT_MAX - m["flat"]) / FLAT_MAX, 0, 1)
    s_clip = 1.0 if m["clip"] <= CLIP_MAX else np.clip(1 - m["clip"] / 0.01, 0, 1)
    s_sil = np.clip(1 - m["silence"] / SILENCE_MAX, 0, 1)
    s_loud = np.clip(1 - abs(np.log10(m["rms"] + 1e-9) - np.log10(0.05)) / 1.5, 0, 1)
    return 100 * (0.40 * s_snr + 0.25 * s_flat + 0.15 * s_clip + 0.10 * s_sil + 0.10 * s_loud)


def 主(argv=None):
    ap = argparse.ArgumentParser(description="变长片段质量门 + 去重")
    ap.add_argument("--目录", nargs="+", required=True, help="切分目录（含 manifest.jsonl）")
    ap.add_argument("--出", required=True)
    ap.add_argument("--既有参照", action="store_true", default=True)
    ap.add_argument("--SNR", type=float, default=SNR_MIN_DB)
    ap.add_argument("--pops", type=int, default=POP_MAX)
    ap.add_argument("--clip", type=float, default=CLIP_MAX)
    ap.add_argument("--flat", type=float, default=FLAT_MAX)
    ap.add_argument("--silence", type=float, default=SILENCE_MAX)
    参 = ap.parse_args(argv)
    出 = Path(参.出); 出.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # 既有 167 指纹（缓存）
    缓存 = 出 / "既有167指纹.npz"
    if 缓存.exists():
        z = np.load(缓存)
        既fps, 既envs = z["fps"], z["envs"]
    else:
        fps, envs = [], []
        for w in sorted(既有根.glob("*.wav")):
            try:
                y = 读单(w)
                fp, env = 指纹(y)
                fps.append(fp); envs.append(env)
            except Exception:
                pass
        既fps = np.array(fps); 既envs = np.array(envs)
        np.savez(缓存, fps=既fps, envs=既envs)
    print(f"[既有] 指纹 {len(既fps)} 条，用时 {time.time()-t0:.1f}s")
    通过, 淘汰 = [], []
    见过 = set()
    保留fps, 保留envs = list(既fps), list(既envs)
    全部 = 0
    for 目 in 参.目录:
        mf = Path(目) / "manifest.jsonl"
        if not mf.exists():
            continue
        for 行 in mf.read_text(encoding="utf-8").splitlines():
            if not 行.strip():
                continue
            r = json.loads(行)
            全部 += 1
            p = Path(r["文件"])
            try:
                y = 读单(p)
                m = 分析(y)
                fp, env = 指纹(y)
                分 = 质量分(m)
                原因s = []
                if m["snr"] < 参.SNR: 原因s.append(f"SNR<{参.SNR}")
                if m["flat"] > 参.flat: 原因s.append(f"谱平坦度>{参.flat}")
                if m["clip"] > 参.clip: 原因s.append(f"削波>{参.clip}")
                if m["pops"] > 参.pops: 原因s.append(f"爆音>{参.pops}")
                if m["silence"] > 参.silence: 原因s.append(f"静音>{参.silence}")
                if 原因s:
                    淘汰.append({**r, "原因": "+".join(原因s), "质量分": round(分, 1), **m})
                    continue
                # 去重：既有 + 已保留
                dup = None
                for k in range(len(保留fps)):
                    cos = float(fp @ 保留fps[k])
                    if cos >= DUP_COS:
                        ec = float(np.corrcoef(env, 保留envs[k])[0, 1])
                        if ec >= DUP_ENV:
                            dup = k; break
                if dup is not None:
                    淘汰.append({**r, "原因": "近重复", "质量分": round(分, 1), **m})
                    continue
                保留fps.append(fp); 保留envs.append(env)
                通过.append({**r, "质量分": round(分, 1), **m})
            except Exception as e:  # noqa: BLE001
                淘汰.append({**r, "原因": f"{type(e).__name__}: {e}"})
    总秒 = sum(x["时长秒"] for x in 通过)
    报 = {"工具": "过滤_变长片段.py", "时间": time.strftime("%Y-%m-%d %H:%M:%S"),
          "总片段": 全部, "通过": len(通过), "淘汰": len(淘汰),
          "通过率": round(len(通过) / max(1, 全部), 3),
          "通过语音秒": round(总秒, 1), "通过语音小时": round(总秒 / 3600, 3),
          "耗时秒": round(time.time() - t0, 1),
          "门槛": dict(SNR_MIN_DB=参.SNR, FLAT_MAX=参.flat, CLIP_MAX=参.clip,
                     POP_MAX=参.pops, SILENCE_MAX=参.silence, DUP_COS=DUP_COS, DUP_ENV=DUP_ENV)}
    (出 / "manifest_通过.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in 通过) + ("\n" if 通过 else ""),
        encoding="utf-8")
    (出 / "淘汰.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in 淘汰) + ("\n" if 淘汰 else ""),
        encoding="utf-8")
    (出 / "质量报告.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(报, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(主())
