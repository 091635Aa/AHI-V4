# -*- coding: utf-8 -*-
"""meeting66 / 诊断_训练句.py —— 记忆化诊断：对**训练集句子**合成 + ASR，判断
模型是「没学会音素→谱」还是「学会了但对新句子不泛化」。

只读 ckpt 与数据；ASR 为离线仪器（单进程）；不写生成路径。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
M66 = HERE.parent
AHI4 = M66.parent
for _p in (str(AHI4 / "meeting64" / "实施"), str(AHI4 / "meeting61" / "实施"),
           str(AHI4 / "meeting58" / "实施")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import 模型_学说话 as M  # noqa: E402
import 声码器与评测 as V  # noqa: E402

数据 = M66 / "产物" / "数据_对齐"
ckpt = sys.argv[1] if len(sys.argv) > 1 else str(M66 / "产物" / "模型_对齐" / "ckpt.pt")
出 = M66 / "产物" / "诊断"
出.mkdir(parents=True, exist_ok=True)

meta = json.load(open(数据 / "meta.json", encoding="utf-8"))
样本 = meta["样本"]
pick = [样本[i] for i in (0, 1, 2)]
引 = None
try:
    from 量具_v9 import 载引擎, CER, 对齐
    引 = 载引擎()
except Exception as e:  # noqa: BLE001
    print("ASR 载入失败：", e)

模型 = M.加载(ckpt)
结果 = []
for s in pick:
    z = np.load(数据 / "样本" / s["文件"])
    帧 = z["音节帧数"]
    每 = int(round(float(np.mean(帧))))
    mel = M.合成mel(模型, s["文"], 每音节帧数=每)
    wav = V.mel转波(mel)
    p = 出 / ("训练句_%02d.wav" % s["id"])
    V.写wav(p, wav)
    r = dict(id=s["id"], 文=s["文"][:40], 每音节帧数=每, mel_shape=list(mel.shape),
             帧间变化=float(np.mean(np.abs(np.diff(mel, axis=0)))),
             真mel帧间变化=float(np.mean(np.abs(np.diff(z["mel"], axis=0)))))
    if 引 is not None:
        t = 引["转写"](str(p))
        r.update(转写=t, CER=CER(s["文"], t))
    结果.append(r)
    print(json.dumps(r, ensure_ascii=False))
json.dump(dict(ckpt=ckpt, 结果=结果), open(出 / "诊断_训练句.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("→", 出 / "诊断_训练句.json")
