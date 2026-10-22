# -*- coding: utf-8 -*-
"""meeting69/实施/合成_泛化声调_S0.1.py —— J4-G 泛化声调（8 条；06 §6 先写死轮廓）。

复用 `合成_S0刺激.py` 的同一合成器（`元音/静默/拼接/写wav`），只换 F0 轮廓；
轮廓在 `06` §6 表里先于读数写死；真值随 wav 落 manifest。
产物：meeting69/产物/S0.1/刺激_泛化/
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M69 = HERE.parent
出 = M69 / "产物" / "S0.1" / "刺激_泛化"


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


合 = 载("合成_S0刺激", HERE / "合成_S0刺激.py")


def 轮廓集():
    t = np.arange(50) / 50.0          # 0..0.49 秒（50 点 @10 ms）
    return {
        "H1_平200": np.full(50, 200.0),
        "H2_平100": np.full(50, 100.0),
        "H3_升90_180": np.linspace(90.0, 180.0, 50),
        "H4_降180_90": np.linspace(180.0, 90.0, 50),
        "H5_V形150_95_150": np.concatenate([np.linspace(150.0, 95.0, 25), np.linspace(95.0, 150.0, 25)]),
        "H6_倒V95_150_95": np.concatenate([np.linspace(95.0, 150.0, 25), np.linspace(150.0, 95.0, 25)]),
        "H7_台阶120_160": np.concatenate([np.full(20, 120.0), np.full(30, 160.0)]),
        "H8_抖动140_8": 140.0 + 8.0 * np.sin(2 * np.pi * 5.0 * t),
    }


def 主():
    出.mkdir(parents=True, exist_ok=True)
    真 = []
    for 名, 轮廓 in 轮廓集().items():
        f0 = np.repeat(np.asarray(轮廓, np.float64), 合.SR // 100)   # 50×10 ms = 0.5 s
        vow = 合.元音(f0, 0.5)
        x = 合.拼接(合.静默(0.3), vow)
        路 = 出 / f"{名}.wav"
        合.写wav(路, x)
        真.append(dict(文件=f"{名}.wav", 路径=str(路), 类="声调_泛化", 标签=名,
                       f0轮廓=[round(float(v), 2) for v in 轮廓]))
    (M69 / "产物" / "S0.1" / "manifest_泛化.json").write_text(
        json.dumps(dict(工具="合成_泛化声调_S0.1.py", 采样率=合.SR, 刺激数=len(真),
                        说明="J4-G 泛化臂（信息臂；轮廓先于读数见 06 §6）", 真值=真),
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已生成 {len(真)} 条泛化声调 → {出}")
    return 0


if __name__ == "__main__":
    sys.exit(主())
