# -*- coding: utf-8 -*-
"""meeting69/实施/跑_谐波v1.py —— 候选 H（谐波重合成）生成 + 不纯度读数。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M69 = HERE.parent
出 = M69 / "产物" / "精修v1"
SR = 22050


def 载(名, 路):
    s = importlib.util.spec_from_file_location(名, 路)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


精 = 载("音频线_v1_精修", HERE / "音频线_v1_精修.py")
谐 = 载("谐波重合成_v1", HERE / "谐波重合成_v1.py")
协 = 载("合成_S0刺激", HERE / "合成_S0刺激.py")

句子 = [
    "今天天气很好我听见了自己的声音我想再说一遍",
    "早上起来我先看了看窗外的光",
    "我不太确定这句话你听清了没有",
    "如果听不清也没关系我再说一遍",
]


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
    报 = dict(工具="跑_谐波v1.py", 时间=time.strftime("%Y-%m-%d %H:%M:%S"), 项=[])
    # 平静 4 句：H
    for i, 文 in enumerate(句子):
        mel, 有效, _ = 精.推理mel(文, 24)
        x, meta = 谐.谐波合成(mel)
        x = 精.收尾(x)
        路, sha = 写(f"句{i}_H.wav", x)
        不 = 谐.不纯度(x)
        报["项"].append(dict(句=i, 臂="H", 路径=路, sha256=sha, 时长秒=round(len(x) / SR, 3), **meta, **不))
        print("句%d H F0=%s 浊帧=%.2f jitter=%.5f shimmer=%.4f" %
              (i, meta["F0中位"], meta["浊帧比"], 不["jitter"], 不["shimmer"]), flush=True)
    # 高兴/低落：H + P1 重采样移调
    for 情, 半音 in (("高兴", 3.0), ("低落", -3.0)):
        p = 精.参数(情, "正常")
        mel, 有效, _ = 精.推理mel(句子[0], p["每音节帧数"])
        if abs(float(p.get("倾斜", 0.0))) > 1e-9:
            tilt = np.linspace(-1.0, 1.0, mel.shape[1], dtype=np.float32) * float(p["倾斜"])
            mel = (mel + tilt[None, :]).astype(np.float32)
        x, meta = 谐.谐波合成(mel)
        x = 精.移调_重采样(x, 半音)
        if abs(float(p.get("增益", 0.0))) > 1e-9:
            x = x * (10.0 ** (float(p["增益"]) / 20.0))
        x = 精.收尾(x)
        路, sha = 写(f"{情}_H_P1.wav", x)
        不 = 谐.不纯度(x)
        报["项"].append(dict(情绪=情, 臂="H", 移调="P1", 路径=路, sha256=sha,
                            时长秒=round(len(x) / SR, 3), **meta, **不))
        print("%s H-P1 F0=%s jitter=%.5f shimmer=%.4f" % (情, meta["F0中位"], 不["jitter"], 不["shimmer"]), flush=True)
    报["耗时秒"] = round(time.time() - t0, 1)
    (出 / "读数_谐波v1.json").write_text(json.dumps(报, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(主())
