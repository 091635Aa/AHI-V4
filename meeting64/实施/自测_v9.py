# -*- coding: utf-8 -*-
"""自测：v9 默认路径必须与 v8 逐位同；各单维旋钮能改变时长/F0 且不崩。"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M64 = os.path.dirname(HERE)
AHI4 = os.path.dirname(M64)
for _p in (HERE,
           os.path.join(AHI4, "meeting63", "实施"),
           os.path.join(AHI4, "meeting62", "实施"),
           os.path.join(AHI4, "meeting61", "实施"),
           os.path.join(AHI4, "meeting58", "实施")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import 韵律_v8 as H8  # noqa: E402
import 韵律_v9 as H9  # noqa: E402
from 接口契约_角色声线 import 声线档案  # noqa: E402

档 = 声线档案.从dict(json.load(open(
    os.path.join(AHI4, "meeting61", "产物", "声线档案", "EmoCompanion.json"),
    encoding="utf-8")))
句 = "今天的风很轻，适合慢慢地说一会儿话。"

w8, _, f8 = H8.人类韵律_v8(档, 300).说(句)
w9, _, f9 = H9.人类韵律_v9(档, 300).说(句)
print("默认 v9 vs v8 波形逐位同 =", bool(w8.shape == w9.shape and np.array_equal(w8, w9)))
print("  帧逐键同 =", bool(f8.keys() == f9.keys() and all(f8[k] == f9[k] for k in f8)))

条件 = [
    ("基线(v9默认)", dict()),
    ("调域倍1.5", dict(调域倍=1.5)),
    ("调域倍2.0", dict(调域倍=2.0)),
    ("VOT倍1.6", dict(VOT倍=1.6)),
    ("浊时长1.5", dict(浊时长倍=1.5)),
    ("浊鼻0.5", dict(浊鼻增益=0.5)),
    ("浊时长1.5+鼻0.5", dict(浊时长倍=1.5, 浊鼻增益=0.5)),
    ("擦带宽0.5", dict(擦带宽倍=0.5)),
    ("擦时长1.6", dict(擦时长倍=1.6)),
    ("语速0.85", dict(语速倍=0.85)),
    ("语速0.80", dict(语速倍=0.80)),
    ("爆带宽0.6", dict(爆带宽倍=0.6)),
    ("共振峰单次缩放", dict(共振峰单次缩放=1.0)),
    ("爆破秒倍3.0", dict(爆破秒倍=3.0)),
]
for 名, 参 in 条件:
    h = H9.人类韵律_v9(档, 300, 参数=参)
    w, 元, 帧 = h.说(句)
    f0 = np.array(帧["f0"])
    src = np.array(帧["声源"])
    st = 12.0 * np.log2(f0[src == 1] / h.基准F0)
    音节 = len(__import__("音韵_v5").音节列表(句))
    print("%-16s 时长%.3fs 音节率%.2f/s 浊帧%3d 半音%+.2f..%+.2f 峰%.3f"
          % (名, len(w) / 44100, 音节 / (len(w) / 44100), int((src == 1).sum()),
             st.min(), st.max(), float(np.max(np.abs(w)))))

print("-" * 78)
print("鼻尾借帧 → 鼻韵母(san1) vs 无鼻尾(sa1) 的元音帧数：")
for 借 in (1.0, 0.5, 0.0):
    for 拼 in ("san1", "sa1"):
        h = H9.人类韵律_v9(档, 300, 参数=dict(鼻尾借帧=借))
        w, 元, 帧 = h.说音节(拼)
        浊 = int((np.array(帧["声源"]) == 1).sum())
        print("  借帧=%.1f  %-5s 总帧%3d 浊帧%3d 时长%.2fs" % (借, 拼, len(帧["f0"]), 浊, len(w) / 44100))
