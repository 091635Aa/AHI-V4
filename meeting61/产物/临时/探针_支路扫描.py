# -*- coding: utf-8 -*-
"""探针：高频并联支路常数扫描（用同一套客观量选常数，不靠耳朵也不靠调阈值）。

量：
  L1 = 句「你好，我是EmoCompanion。」谱带分布 与 参考原声均值 的 L1 距离（v5: 0.164 通过线 0.25）
  峰比 = 元音谱包络峰位 / F1目标（v5 实测 0.36–0.88，v6 1.0–1.5）
  鼻元 = 妈 的鼻音段/元音段电平（v5 −13.5dB，判据门 −16dB）
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, r"<workspace>/03_AHI\AHI4\meeting61\实施")
sys.path.insert(0, r"<workspace>/03_AHI\AHI4\meeting58\实施")

import 韵律_v6 as H5  # noqa: E402
import 引擎_v6 as E6  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

FS = 44100
with open(r"<workspace>/03_AHI\AHI4\meeting61\产物\声线档案\EmoCompanion.json", encoding="utf-8") as f:
    档 = 声线档案.从dict(json.load(f))
    档低 = 档.派生(基频=120.0)
音 = H5.人类韵律_v5(档, 300).音色()
参考带 = {"0-500": 0.4307, "500-1000": 0.3796, "1000-2000": 0.1263,
        "2000-4000": 0.0372, "4000-8000": 0.0186, "8000-16000": 0.0075}


def 带(x):
    x = np.asarray(x, dtype=np.float64)
    n, hop = 4096, 2048
    k = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(k)[:, None]
    P = np.mean(np.abs(np.fft.rfft(x[idx] * np.hanning(n)[None, :], axis=1)) ** 2, axis=0)
    P[0] = 0.0
    f = np.fft.rfftfreq(n, 1.0 / FS)
    T = float(P.sum()) or 1.0
    return {k2: float(P[(f >= a) & (f < b)].sum() / T)
            for k2, (a, b) in {"0-500": (0, 500), "500-1000": (500, 1000),
                               "1000-2000": (1000, 2000), "2000-4000": (2000, 4000),
                               "4000-8000": (4000, 8000), "8000-16000": (8000, 16000)}.items()}


def 包络峰位(x):
    x = np.asarray(x, dtype=np.float64) - float(np.mean(x))
    X = np.fft.rfft(x * np.hanning(len(x)))
    cep = np.fft.irfft(np.log(np.abs(X) + 1e-12))
    L = max(4, int(0.0022 * FS))
    cep[L:-L] = 0.0
    env = np.exp(np.fft.rfft(cep).real)
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    m = (f >= 150.0) & (f <= 3500.0)
    return float(f[m][int(np.argmax(env[m]))])


配置 = {
    "0 纯串联": (0.0, 1600.0, (0.30, 0.70, 0.40)),
    "1 现状F3/F4/F5": (0.55, 1600.0, (0.30, 0.70, 0.40)),
    "2 F2-F5 HP1k": (0.55, 1000.0, (0.25, 0.30, 0.55, 0.35)),
    "3 F2-F5 HP0.8k强": (0.75, 800.0, (0.35, 0.35, 0.60, 0.40)),
    "4 F3-F5 HP1.2k": (0.60, 1200.0, (0.40, 0.70, 0.45)),
}
元 = {"啊": 825.0, "哦": 528.0, "鹅": 572.0, "衣": 330.0, "乌": 352.0}
print("=" * 108)
for 名, (支, 低切, 增) in 配置.items():
    E6._高频支路 = 支
    E6._高频低切Hz = 低切
    E6._高频支路增 = 增
    句 = [H5.人类韵律_v5(档, 300).说("你好，我是EmoCompanion。")[0],
          H5.人类韵律_v5(档, 301).说("今天的风很轻，适合慢慢地说一会儿话。")[0]]
    bs = [带(w) for w in 句]
    均 = {k: float(np.mean([b[k] for b in bs])) for k in bs[0]}
    L1 = float(sum(abs(均[k] - 参考带[k]) for k in 均))
    # 元音包络峰位比
    峰比 = {}
    for 字, f1t in 元.items():
        帧 = H5.人类韵律_v5(档低, 300).建帧(字)
        idx = [i for i, t in enumerate(帧["声源"]) if int(t) == 1]
        a0 = (idx[len(idx) // 3] if idx else 0) * 441
        w, _ = E6.合成(帧, 音色=音, 种子=300)
        峰比[字] = round(包络峰位(w[a0:a0 + int(0.12 * FS)]) / f1t, 2)
    # 鼻音段/元音段
    帧m = H5.人类韵律_v5(档, 300).建帧("妈")
    idxm = [i for i, t in enumerate(帧m["声源"]) if int(t) == 1]
    wm, _ = E6.合成(帧m, 音色=音, 种子=300)
    rm = [float(np.sqrt(np.mean(wm[i * 441:(i + 1) * 441] ** 2))) for i in idxm]
    鼻元 = round(20 * np.log10(max(float(np.mean(rm[:4])), 1e-12)
                              / max(float(np.mean(rm[8:13])), 1e-12)), 2)
    print("%-18s L1=%.3f | 带 %s | 峰比 %s | 鼻元 %.1fdB"
          % (名, L1, json.dumps({k: round(v, 3) for k, v in 均.items()}, ensure_ascii=False),
             json.dumps(峰比, ensure_ascii=False), 鼻元))
print("=" * 108)
print("参考带", json.dumps(参考带, ensure_ascii=False))
