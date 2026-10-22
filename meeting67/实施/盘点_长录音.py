# -*- coding: utf-8 -*-
"""meeting67/实施/盘点_长录音.py —— 长录音数据集盘点（离线仪器；不训练、不改既有件）。

目的：所有者 2026-10-21 13:27「录音自己找在数据集」→ 对 `<data-root>/\格式工厂音频合并 EmoCompanion数据集(...).mp3`
（ffprobe: 38.35 h / 48 kHz / 2ch / 320 kbps）做：
  1) 7 个等距 12 s 采样 → 离线 ASR 转写（判断内容域/说话人）；
  2) 采样与 167 条既有语料的 F0 统计对比（判断是否同一说话人）；
  3) 格式/总时长/与既有 2.77 h 的关系登记 → 产物 JSON。

口径：ASR 只作离线分析（C10/C9）；F0 用 librosa.pyin 固定参数（确定性）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

根 = Path(r"<workspace>/03_AHI\AHI4\meeting67")
出 = 根 / "产物" / "数据盘点"
出.mkdir(parents=True, exist_ok=True)
长录音 = Path(r"<data-root>/格式工厂音频合并 EmoCompanion数据集 (1+EmoCompanion数据集 (2+EmoCompanion数据集 (3+EmoCompanion数据集 (4+EmoCompanion数据集 (5+EmoCompanion数据集 (6.mp3")
FFMPEG = Path(r"<path>\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe")
数据根 = Path(r"<path>\SpeakerA7代模型训练数据")


def 提取(路: Path, 起: float, 长: float, 出wav: Path, sr=16000):
    出wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(FFMPEG), "-v", "error", "-y", "-ss", f"{起:.3f}", "-t", f"{长:.3f}",
           "-i", str(路), "-ac", "1", "-ar", str(sr), "-f", "wav", str(出wav)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    import wave
    with wave.open(str(出wav), "rb") as w:
        n = w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
        return x, w.getframerate()


def F0统计(x: np.ndarray, sr: int) -> dict:
    import librosa
    try:
        f0, 浊, _ = librosa.pyin(x, fmin=60, fmax=500, sr=sr, frame_length=1024, hop_length=160)
        浊 = np.asarray(浊, dtype=bool)
        v = np.asarray(f0)[浊]
        v = v[np.isfinite(v)]
        if v.size < 10:
            return {"浊帧": int(浊.sum()), "中位F0": None}
        return {"浊帧": int(浊.sum()), "浊帧占比": float(浊.mean()),
                "中位F0": float(np.median(v)),
                "F0_10分位": float(np.percentile(v, 10)),
                "F0_90分位": float(np.percentile(v, 90))}
    except Exception as e:  # noqa: BLE001
        return {"错误": f"{type(e).__name__}: {e}"}


def 主():
    import importlib.util as u
    量 = u.spec_from_file_location("量具", r"<workspace>/03_AHI\AHI4\meeting64\实施\量具_v9.py")
    m量 = u.module_from_spec(量)
    量.loader.exec_module(m量)
    引 = m量.载引擎()
    报 = {"工具": "盘点_长录音.py", "时间": time.strftime("%Y-%m-%d %H:%M:%S"),
           "长录音": {"路径": str(长录音), "大小MB": round(长录音.stat().st_size / 1e6, 1)},
           "ASR引擎": 引["名"], "采样": []}
    # 7 个采样点：按 38.35h 等距（覆盖 6 个拼接段）
    总h = 38.35
    点 = [0.5, 6.8, 13.2, 19.5, 25.9, 32.3, 37.6]
    临时 = Path(tempfile.mkdtemp(prefix="盘点_"))
    for i, h in enumerate(点):
        w = 临时 / f"长_{i}.wav"
        try:
            x, sr = 提取(长录音, h * 3600, 12.0, w)
            t0 = time.time()
            文 = 引["转写"](str(w))
            f0 = F0统计(x, sr)
            报["采样"].append({"序": i, "位置h": h, "秒数": round(len(x) / sr, 2),
                              "转写": 文, "转写耗时s": round(time.time() - t0, 2), **f0})
            print(f"[长录音 {i}] {h}h 转写：{文[:60]} | F0中位 {f0.get('中位F0')}")
        except Exception as e:  # noqa: BLE001
            报["采样"].append({"序": i, "位置h": h, "错误": f"{type(e).__name__}: {e}"})
            print(f"[长录音 {i}] 失败 {type(e).__name__}: {e}")

    # 既有 167 条中随机 12 条（固定种子）60 s 的 F0 统计
    import random
    random.seed(300)
    wavs = sorted(数据根.glob("*.wav"))
    抽 = random.sample(wavs, min(12, len(wavs)))
    报["既有语料F0"] = []
    for w in 抽:
        try:
            import wave
            with wave.open(str(w), "rb") as f:
                n = f.getnframes(); sr = f.getframerate(); ch = f.getnchannels()
                raw = f.readframes(n)
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if ch == 2:
                x = x.reshape(-1, 2).mean(axis=1)
            if sr != 16000:
                import librosa
                x = librosa.resample(x, orig_sr=sr, target_sr=16000)
                sr = 16000
            报["既有语料F0"].append({"文件": w.name, "秒": round(len(x) / sr, 1),
                                   **F0统计(x, sr)})
        except Exception as e:  # noqa: BLE001
            报["既有语料F0"].append({"文件": w.name, "错误": f"{type(e).__name__}: {e}"})

    (出 / "盘点_长录音.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已写", 出 / "盘点_长录音.json")
    # 汇总 F0 中位
    f0长 = [s.get("中位F0") for s in 报["采样"] if s.get("中位F0")]
    f0既 = [s.get("中位F0") for s in 报["既有语料F0"] if s.get("中位F0")]
    print("长录音 F0 中位:", [round(v, 1) for v in f0长], "均值 %.1f" % (np.mean(f0长) if f0长 else -1))
    print("既有语料 F0 中位:", [round(v, 1) for v in f0既], "均值 %.1f" % (np.mean(f0既) if f0既 else -1))
    return 0


if __name__ == "__main__":
    sys.exit(主())
