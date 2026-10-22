# -*- coding: utf-8 -*-
"""探针：按帧类型分解带分布（浊帧/噪帧/静帧），看 v6 的变化是否只落在元音层。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

MEET61 = r"<workspace>/03_AHI\AHI4\meeting61"
sys.path.insert(0, os.path.join(MEET61, "实施"))
sys.path.insert(0, os.path.join(os.path.dirname(MEET61), "meeting58", "实施"))

import 韵律_v6 as H5  # noqa: E402
import 韵律_v5  # noqa: E402
import 引擎_v5 as E5  # noqa: E402
import 引擎_v6 as E6  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

FS = 44100
with open(os.path.join(MEET61, "产物", "声线档案", "EmoCompanion.json"), encoding="utf-8") as f:
    档 = 声线档案.从dict(json.load(f))
音 = H5.人类韵律_v5(档, 300).音色()
句s = ("你好，我是EmoCompanion。", "今天的风很轻，适合慢慢地说一会儿话。",
      "语言不只是声音，它把一个人的温度，送到另一个人的心里。")


def 带(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 4096:
        return None
    n, hop = 4096, 2048
    k = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(k)[:, None]
    P = np.mean(np.abs(np.fft.rfft(x[idx] * np.hanning(n)[None, :], axis=1)) ** 2, axis=0)
    P[0] = 0.0
    f = np.fft.rfftfreq(n, 1.0 / FS)
    T = float(P.sum()) or 1.0
    return {k2: round(float(P[(f >= a) & (f < b)].sum() / T), 4)
            for k2, (a, b) in {"0-500": (0, 500), "500-1k": (500, 1000), "1-2k": (1000, 2000),
                               "2-4k": (2000, 4000), "4-8k": (4000, 8000)}.items()}


def 分段(w, 帧, 类s):
    out = []
    for i, t in enumerate(帧["声源"]):
        if int(t) in 类s:
            out.append(w[i * 441:min((i + 1) * 441, len(w))])
    return np.concatenate(out) if out else np.zeros(0)


print("=" * 104)
print("句级带分布：全体 / 仅浊帧（元音+浊声母） / 仅噪帧")
for 名, mod in (("v5", E5), ("v6", E6)):
    print("---", 名)
    for i, 句 in enumerate(句s):
        帧 = 韵律_v5.人类韵律_v5(档, 300 + i).建帧(句)
        w, _ = mod.合成(帧, 音色=音, 种子=300 + i)
        print("  %-16s 全 %s" % (句[:8], json.dumps(带(w), ensure_ascii=False)))
        print("  %-16s 浊 %s" % ("", json.dumps(带(分段(w, 帧, {1})), ensure_ascii=False)))
        print("  %-16s 噪 %s" % ("", json.dumps(带(分段(w, 帧, {0})), ensure_ascii=False)))
print("=" * 104)
