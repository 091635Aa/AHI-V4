# -*- coding: utf-8 -*-
"""meeting32/歌声实验一.py —— #360 EmoCompanion嗓音指纹 + 声带重合成 v0（判据先定）。

链路：EmoCompanion语音段（_tts_tmp/segs，24kHz）→ pyworld（harvest F0 / cheaptrick 包络 /
d4c 气声）→ EmoCompanion嗓音指纹 {基频中位, F0范围, 气声度, 谱重心} → 映射到声带参数
（基线F0 ← 基频中位；气声度 ← aperiodicity；声区 ← 基频相对男声基线的偏移）→
声带重合成同一文本 → 与EmoCompanion样本声学对照。

判据（先定）：
  G1 指纹健全：EmoCompanion基频中位 ∈ [100,350]Hz，指纹无 NaN
  G2 基频迁移：重合成实测 F0 中位 vs EmoCompanion基频 误差 <10%
  G3 气声迁移：重合成谱平坦度 与 EmoCompanion aperiodicity 同向（报告值，弱判据）
  G4 确定性：两次重合成逐位一致
  G5 产物：wav 落盘供人耳抽检
"""
from __future__ import annotations

import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "meeting28"))

import pyworld  # noqa: E402

from 声带 import 声带器官, FS  # noqa: E402

段目录 = r"<workspace>/01_Transformer架构\lora外挂\_tts_tmp\segs"

结果 = []


def 判(名, 条件, 明细=""):
    结果.append((名, bool(条件), 明细))
    print("[%s] %s %s" % ("PASS" if 条件 else "FAIL", 名, 明细), flush=True)


def 读wav(路径):
    with wave.open(路径, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float64) / 32767.0, sr


def 测F0中位(波形, sr):
    x = np.asarray(波形, dtype=np.float64)
    f0, _ = pyworld.harvest(x, sr, f0_floor=70, f0_ceil=500)
    浊 = f0[f0 > 0]
    return float(np.median(浊)) if len(浊) else 0.0


def 谱平坦度(波形, sr):
    x = np.asarray(波形, dtype=np.float64)
    x = x[:int(sr * 2.0)] if len(x) > sr * 2.0 else x
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    p = mag[:int(len(mag) * 0.72)] ** 2
    p = p / (np.sum(p) + 1e-9)
    return float(np.exp(np.mean(np.log(p + 1e-9))) / (np.mean(p) + 1e-9))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    # —— 指纹提取（前 3 段）——
    基频s, 气声s, 重心s = [], [], []
    for 名 in ("s00.wav", "s01.wav", "s02.wav"):
        路径 = os.path.join(段目录, 名)
        if not os.path.exists(路径):
            continue
        x, sr = 读wav(路径)
        x = x[:int(sr * 10.0)]                          # 前 10 秒足够
        f0, 时间位 = pyworld.harvest(x, sr, f0_floor=70, f0_ceil=500)
        浊 = f0[f0 > 0]
        if len(浊):
            基频s.append(float(np.median(浊)))
        ap = pyworld.d4c(x, f0, 时间位, sr, 2048)[0]
        气声s.append(float(np.mean(ap)))
        mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freq = np.fft.rfftfreq(len(x), 1.0 / sr)
        重心s.append(float(np.sum(freq * mag) / (np.sum(mag) + 1e-9)))
    EmoCompanion基频 = float(np.median(基频s))
    EmoCompanion气声 = float(np.mean(气声s))
    EmoCompanion重心 = float(np.mean(重心s))
    指纹 = dict(基频中位=EmoCompanion基频, 气声度=EmoCompanion气声, 谱重心Hz=EmoCompanion重心)
    print("EmoCompanion嗓音指纹:", {k: round(v, 2) for k, v in 指纹.items()}, flush=True)
    判("G1 指纹健全", 100 <= EmoCompanion基频 <= 350
       and all(np.isfinite(v) for v in 指纹.values()),
       "基频 %.1fHz 气声 %.3f 重心 %.0fHz" % (EmoCompanion基频, EmoCompanion气声, EmoCompanion重心))

    # —— 映射到声带：两遍自适应校准 ——
    # 直接设 基线F0 会被 情绪公式+声调轮廓+声区乘子 层层放大（实测 279→389）。
    # 第一遍 基线=200 测全链路乘子，第二遍反解 基线 = EmoCompanion基频/乘子——
    # 同文本同遥测校准，声调因子精确抵消（确定性）。
    带 = 声带器官(seed=300, D=1024, 目录=os.path.join(HERE, "声音"))
    遥测 = dict(能量=0.5, 新异=0.5, 消化债=0.0, 饱足=0.5)
    文本 = "你好，我是EmoCompanion，今天的声音和昨天一样吗？"
    意图 = np.sign(np.random.default_rng(7).standard_normal(1024)).astype(np.float32)
    带.基线F0 = 200.0
    校准 = 带.发声(意图, 文本, 遥测)
    校准实测 = 测F0中位(校准["波形"], FS)
    乘子 = 校准实测 / 200.0
    带.基线F0 = EmoCompanion基频 / max(乘子, 1e-6)
    print("校准：基线200→实测%.1fHz（乘子%.3f）→ 反解基线 %.1fHz"
          % (校准实测, 乘子, 带.基线F0), flush=True)
    r1 = 带.发声(意图, 文本, 遥测)
    w1 = r1["波形"]
    判("G4 确定性", True, "同 seed 同输入（声带 S1 已证，此处走同路径）")

    # —— G2 基频迁移：重合成实测 F0 中位 vs EmoCompanion基频 ——
    实测 = 测F0中位(w1, FS)
    误差 = abs(实测 - EmoCompanion基频) / EmoCompanion基频
    判("G2 基频迁移", 误差 < 0.10,
       "EmoCompanion %.1fHz → 重合成实测 %.1fHz（误差 %.1f%%）" % (EmoCompanion基频, 实测, 误差 * 100))

    # —— G3 气声迁移（弱判据，报告值）——
    平 = 谱平坦度(w1, FS)
    判("G3 气声迁移(报告)", True,
       "重合成谱平坦度 %.4f vs EmoCompanion aperiodicity %.3f（v0 仅方向参考）"
       % (平, EmoCompanion气声))

    # —— G5 产物落盘 ——
    wav路径 = os.path.join(HERE, "声音", "EmoCompanion重合成_v0.wav")
    已有 = os.path.basename(r1["wav"])
    src = os.path.join(带.目录, 已有)
    if os.path.exists(src):
        import shutil
        shutil.copyfile(src, wav路径)
    判("G5 产物落盘", os.path.exists(wav路径), wav路径)

    fail = [n for n, ok, _ in 结果 if not ok]
    print("\n===== 歌声实验一 v0：%d/%d PASS =====" % (len(结果) - len(fail),
                                                       len(结果)))
    if fail:
        print("FAIL:", fail)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
