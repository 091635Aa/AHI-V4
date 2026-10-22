# -*- coding: utf-8 -*-
"""meeting69/实施/合成_S1刺激.py —— S1 合成刺激（09 §3；确定性；无外部模型）。

复用 `合成_S0刺激.py` 的合成器（元音/噪声段/爆破/静音/拼接/写wav），生成：
  稳态元音 4 s（A-2 适应）；突变 ×4（A-2 突变）；重复音节 ×3 音高（A-5 虽然延后，仍作信息臂素材）；
  双流交错 1 kHz/4 kHz（H6 信息臂）；单流 1 kHz（H6 对照）；静音、白噪（sham）。
真实语音清单：L: 下按文件名排序取 [0,8,…,152] 共 20 条 × 5 s（偏移 10.0 s），只读。
落盘：meeting69/产物/S1/刺激/ + manifest.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M69 = HERE.parent
出 = M69 / "产物" / "S1" / "刺激"
SR = 22050
真实根 = Path(r"<path>\SpeakerA7代模型训练数据")


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


合 = 载("合成_S0刺激", HERE / "合成_S0刺激.py")


def 主():
    出.mkdir(parents=True, exist_ok=True)
    真 = []

    def 存(名, x, gt):
        路 = 出 / 名
        合.写wav(路, x)
        真.append(dict(文件=名, 路径=str(路), **gt))

    # ① 稳态元音 4 s（A-2 适应）
    存("S1_稳态元音.wav", 合.元音(120.0, 4.0), dict(类="稳态", 起秒=0.0, 时长秒=4.0))

    # ② 突变 ×4（A-2 突变）：1 s 静音 + 爆破 + 送气 20 ms + 1.5 s 元音
    for k, (中心, f0) in enumerate(((2500.0, 120.0), (3500.0, 130.0), (4500.0, 140.0), (5500.0, 150.0))):
        爆 = 合.爆破(中心=中心, 带宽=2200.0, 种子=20 + k)
        x = 合.拼接(合.静默(1.0), 爆, 合.送气(0.020, 种子=30 + k), 合.元音(f0, 1.5))
        存(f"S1_突变{k}.wav", x, dict(类="突变", burst秒=1.0, burst中心Hz=中心, 子=k))

    # ③ 重复音节 ×3 音高（每个 5 次；A-5 延后为信息臂）
    for k, f0 in enumerate((100.0, 130.0, 160.0)):
        段 = [合.静默(0.5)]
        for r in range(5):
            段 += [合.爆破(中心=3500.0, 带宽=2000.0, 种子=40 + r), 合.送气(0.020, 种子=50 + r),
                   合.元音(f0, 0.15), 合.静默(0.25)]
        x = 合.拼接(*段)
        存(f"S1_重复{k}.wav", x, dict(类="重复", f0=f0, 周期秒=0.432, 首起秒=0.5, 次数=5, 子=k))

    # ④ 双流交错：1 kHz 与 4 kHz 噪声短爆，各 200 ms 周期、错开 100 ms，4 s
    n = int(4.0 * SR)
    A = np.zeros(n, np.float32); B = np.zeros(n, np.float32)
    短 = 合.噪声段(0.06, 中心=1000.0, 带宽=700.0, 振幅=0.6, 种子=60)
    for t0 in np.arange(0.0, 3.9, 0.2):
        i = int(t0 * SR); A[i:i + len(短)] += 短[: max(0, min(len(短), n - i))]
    短2 = 合.噪声段(0.06, 中心=4000.0, 带宽=1500.0, 振幅=0.6, 种子=61)
    for t0 in np.arange(0.1, 3.9, 0.2):
        i = int(t0 * SR); B[i:i + len(短2)] += 短2[: max(0, min(len(短2), n - i))]
    存("S1_双流.wav", (A + B).astype(np.float32), dict(类="双流", 流A=[0, 200, 400], 流B=[100, 300, 500]))
    存("S1_单流.wav", A.astype(np.float32), dict(类="单流", 流A=[0, 200, 400]))

    # ⑤ sham
    存("S1_静音.wav", 合.静默(4.0), dict(类="静音"))
    存("S1_白噪.wav", 合.噪声段(4.0, 振幅=0.3, 种子=70), dict(类="白噪"))

    # ⑥ 真实语音清单（只读；确定性选 20 条 × 5 s @10 s）
    真实 = []
    if 真实根.exists():
        文件们 = sorted(p for p in 真实根.iterdir() if p.suffix.lower() == ".wav")
        取 = [文件们[i] for i in range(0, len(文件们), 8)][:20]
        for p in 取:
            真实.append(dict(文件=p.name, 路径=str(p), 起始秒=10.0, 时长秒=5.0))
    (M69 / "产物" / "S1" / "manifest.json").write_text(
        json.dumps(dict(工具="合成_S1刺激.py", 采样率=SR, 刺激数=len(真),
                        真实=真实, 说明="09 §3；真实件只读、确定性选片", 真值=真),
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"合成刺激 {len(真)} 条 → {出}；真实清单 {len(真实)} 条")
    return 0


if __name__ == "__main__":
    sys.exit(主())
