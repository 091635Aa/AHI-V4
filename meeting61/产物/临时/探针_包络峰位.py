# -*- coding: utf-8 -*-
"""探针：同一帧下 v5/v6 的「谱包络峰位」与逐帧电平（用于设计可信的元音判据 + 定位 B3）。"""
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
    档低 = 档.派生(基频=120.0)
音 = H5.人类韵律_v5(档, 300).音色()


def 包络(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - float(np.mean(x))
    X = np.fft.rfft(x * np.hanning(len(x)))
    cep = np.fft.irfft(np.log(np.abs(X) + 1e-12))
    L = max(4, int(0.0022 * FS))
    cep[L:-L] = 0.0
    env = np.exp(np.fft.rfft(cep).real)
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    return f, env


def 量(seg):
    f, env = 包络(seg)
    m = (f >= 150.0) & (f <= 3500.0)
    ff, ee = f[m], env[m]
    k = int(np.argmax(ee))
    峰位 = float(ff[k])
    # 包络峰相对「低于峰位的局部谷」的高度
    lo = (f >= 150.0) & (f < 峰位 * 0.92)
    谷 = float(np.min(env[lo])) if np.any(lo) else 0.0
    突 = 20 * np.log10(max(ee[k], 1e-12) / max(谷, 1e-12))
    return 峰位, round(突, 1), round(float(np.max(ee)), 4)


目标 = {"啊": (825.0, 1305.0), "哦": (528.0, 887.0), "鹅": (572.0, 1305.0),
       "衣": (330.0, 2349.0), "乌": (352.0, 752.0), "妈": (825.0, 1305.0)}
print("=" * 104)
print("谱包络峰位（150–3500Hz 内包络最大点） vs F1目标 —— 这是「元音有没有成形」的耳听代理")
for 字, (f1t, f2t) in 目标.items():
    帧 = H5.人类韵律_v5(档低, 300).建帧(字)
    idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
    a0 = (idx[len(idx) // 3] if idx else 0) * 441
    print("[%s] F1目标 %.0f F2目标 %.0f" % (字, f1t, f2t))
    for 名, mod in (("v5并联", E5), ("v6串联", E6)):
        x, _ = mod.合成(帧, 音色=音, 种子=300)
        seg = x[a0:a0 + int(0.12 * FS)]
        峰位, 突, _ = 量(seg)
        print("    %-8s 包络峰位 %7.1f Hz | 相对低端谷突出 %6.1f dB | 峰位/F1 %.2f"
              % (名, 峰位, 突, 峰位 / f1t))

print("=" * 104)
print("逐帧电平（字=妈，含 m 浊声母）—— 定位 B3 差异")
帧 = H5.人类韵律_v5(档, 300).建帧("妈")
idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
print("浊帧序号", idx, "| 前4帧=鼻音 m，第9-13帧=元音")
for 名, mod in (("v5并联", E5), ("v6串联", E6)):
    x, _ = mod.合成(帧, 音色=音, 种子=300)
    r = [round(float(np.sqrt(np.mean(x[i * 441:(i + 1) * 441] ** 2))), 5) for i in idx]
    print("  %-8s 逐帧RMS %s" % (名, r))
    print("           鼻(前4) %.5f 元(9-13) %.5f 差 %.2fdB"
          % (float(np.mean(r[:4])), float(np.mean(r[8:13])),
             20 * np.log10(max(float(np.mean(r[:4])), 1e-12)
                           / max(float(np.mean(r[8:13])), 1e-12))))
