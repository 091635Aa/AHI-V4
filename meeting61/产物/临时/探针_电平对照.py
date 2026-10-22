# -*- coding: utf-8 -*-
"""探针：同一帧（韵律_v6 产出，与 韵律_v5 逐位同）下 v5/v6 引擎的浊音/噪音电平与谱分布。"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, r"<workspace>/03_AHI\AHI4\meeting61\实施")
sys.path.insert(0, r"<workspace>/03_AHI\AHI4\meeting58\实施")

import 韵律_v6 as H5  # noqa: E402
import 引擎_v5 as E5  # noqa: E402
import 引擎_v6 as E6  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

FS = 44100
with open(r"<workspace>/03_AHI\AHI4\meeting61\产物\声线档案\EmoCompanion.json", encoding="utf-8") as f:
    档 = 声线档案.从dict(json.load(f))


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))) if len(x) else 0.0


def 带(x):
    P = np.abs(np.fft.rfft(np.asarray(x) * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    P[0] = 0.0
    T = float(P.sum()) or 1.0
    return {k: round(float(P[(f >= a) & (f < b)].sum() / T), 3)
            for k, (a, b) in {"0-500": (0, 500), "500-1k": (500, 1000), "1-2k": (1000, 2000),
                              "2-4k": (2000, 4000), "4-8k": (4000, 8000)}.items()}


音 = H5.人类韵律_v5(档, 300).音色()
print("音色（EmoCompanion）:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                 for k, v in 音.items()}, ensure_ascii=False, default=str))
for 字 in ("撒", "妈", "啊"):
    帧 = H5.人类韵律_v5(档, 300).建帧(字)
    浊 = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
    噪 = [i for i, t in enumerate(帧["声源"]) if int(t) == 0 and float(帧["f1"][i]) < 450.0]
    静 = [i for i, t in enumerate(帧["声源"]) if int(t) == -1]
    print("=" * 90)
    print("[%s] 帧 %d：浊 %d 噪 %d 静 %d | 幅均值 %.3f" % (字, len(帧["f0"]), len(浊), len(噪),
                                                       len(静), float(np.mean(帧["幅"]))))
    for 名, mod in (("v5并联", E5), ("v6串联", E6)):
        for 音色名, 音色 in (("无音色", None), ("EmoCompanion声线", 音)):
            x, _ = mod.合成(帧, 音色=音色, 种子=300)
            z = b"".join([x[i * 441:(i + 1) * 441].tobytes() for i in 浊])
            n = b"".join([x[i * 441:(i + 1) * 441].tobytes() for i in 噪])
            zv = np.frombuffer(z, dtype=np.float64)
            nv = np.frombuffer(n, dtype=np.float64)
            print("  %-8s %-6s 浊RMS %.5f 噪RMS %.5f 清浊差 %6.2fdB | 浊带 %s"
                  % (名, 音色名, rms(zv), rms(nv),
                     20 * np.log10(max(rms(nv), 1e-12) / max(rms(zv), 1e-12)),
                     json.dumps(带(zv), ensure_ascii=False) if len(zv) > 512 else "-"))
