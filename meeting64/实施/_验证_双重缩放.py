# -*- coding: utf-8 -*-
"""验证「共振峰双重缩放」与 v9 `共振峰单次缩放` 旋钮的实际效果（不发 ASR）。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M64 = os.path.dirname(HERE)
AHI4 = os.path.dirname(M64)
for _p in (os.path.join(M64, "实施"),
           os.path.join(AHI4, "meeting63", "实施"),
           os.path.join(AHI4, "meeting62", "实施"),
           os.path.join(AHI4, "meeting61", "实施"),
           os.path.join(AHI4, "meeting58", "实施")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import 韵律_v8 as H8  # noqa: E402
import 韵律_v9 as H9  # noqa: E402
import 声线覆盖层_v7 as 覆  # noqa: E402

档 = 覆.取("EmoCompanion")
s = tuple(float(x) for x in 档.共振峰缩放)
print("覆盖层 共振峰缩放 =", tuple(round(x, 4) for x in s))

for 名, 字 in (("a", "啊"), ("i", "衣"), ("u", "乌"), ("ang", "肮")):
    h8 = H8.人类韵律_v8(档, 300)
    _, _, 帧 = h8.说(字)
    k = len(帧["f1"]) // 2
    f1, f2 = 帧["f1"][k], 帧["f2"][k]
    print("v8   %-3s 帧f1=%.0f 帧f2=%.0f | 引擎再乘 s ⇒ 名义 f1=%.0f f2=%.0f"
          % (名, f1, f2, f1 * s[0], f2 * s[1]))
    h9 = H9.人类韵律_v9(档, 300, 参数=dict(共振峰单次缩放=1.0))
    _, _, 帧9 = h9.说(字)
    k9 = len(帧9["f1"]) // 2
    print("v9单次翻转 %-3s 帧f1=%.0f 帧f2=%.0f | 引擎因子=1 ⇒ 名义 f1=%.0f f2=%.0f"
          % (名, 帧9["f1"][k9], 帧9["f2"][k9], 帧9["f1"][k9], 帧9["f2"][k9]))
