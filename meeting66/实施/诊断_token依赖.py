# -*- coding: utf-8 -*-
"""快速诊断：模型输出是否真的依赖音素 token（不同声母/韵母 → mel 是否不同）。"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
M66 = HERE.parent
AHI4 = M66.parent
for _p in (str(AHI4 / "meeting64" / "实施"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import 模型_学说话 as M  # noqa: E402

ckpt = sys.argv[1] if len(sys.argv) > 1 else str(M66 / "产物" / "模型_对齐" / "ckpt.pt")
模型 = M.加载(ckpt)
print("ckpt:", ckpt)
print("词表: 声母%d 韵母%d 调%d" % (len(模型.词表["声母"]), len(模型.词表["韵母"]), len(模型.词表["调"])))
测试 = ["撒", "沙", "八", "妈", "衣", "乌", "啊"]
mels = {}
for 文 in 测试:
    m = M.合成mel(模型, 文, 每音节帧数=24)
    mels[文] = m
    print("%-3s mel%s 均值%.3f std%.3f 帧间%.4f" % (文, m.shape, float(m.mean()), float(m.std()),
                                              float(np.mean(np.abs(np.diff(m, axis=0))))))
print("-" * 60)
ks = list(mels)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        a, b = mels[ks[i]], mels[ks[j]]
        n = min(a.shape[0], b.shape[0])
        d = float(np.mean(np.abs(a[:n] - b[:n])))
        print("%s vs %s : 平均绝对差 %.4f（0=完全相同）" % (ks[i], ks[j], d))
