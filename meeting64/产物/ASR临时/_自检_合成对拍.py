# -*- coding: utf-8 -*-
"""meeting64/产物/ASR临时/_自检_合成对拍.py —— 只测合成（不载 ASR）：验证「不经 声带后端_v8」的
波形与 meeting63 官方多句基线 18 个 wav **逐位同**（sha256）。

跑：
  $env:PYTHONIOENCODING="utf-8"
  & "<workspace>/\03_AHI\\.venv\\Scripts\\python.exe" "<workspace>/\03_AHI\\AHI4\\meeting64\\产物\\ASR临时\\_自检_合成对拍.py"
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
M64 = os.path.dirname(os.path.dirname(HERE))
IMP = os.path.join(M64, "实施")
sys.path.insert(0, IMP)

import numpy as np  # noqa: E402

import 量具_v9 as 量  # noqa: E402
import 韵律_v8 as H8  # noqa: E402

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    np.seterr(all="ignore")
    参考 = os.path.join(量.M63, "产物", "ASR临时", "多句")
    合 = 量.合成多句(H8.人类韵律_v8, 角色s=量.多句角色, 标签="自检对拍",
                     临时目录=os.path.join(HERE, "自检对拍"), 种子=300)
    参考表 = sorted(f for f in os.listdir(参考) if f.lower().endswith(".wav"))
    序表 = [(s, n) for s, _ in 量.句集 for n in 量.多句角色]
    print("-" * 100)
    全同 = True
    for i, (句号, 名) in enumerate(序表):
        v = 合["样本"][(句号, 名)]
        p = os.path.join(参考, 参考表[i])
        同 = bool(量._sha256(p) == v["sha256"])
        全同 = bool(全同 and 同)
        print("%-9s 本件 sha=%s ｜ 基线 %-22s sha=%s ｜ 逐位同=%s"
              % ("%s·%s" % (句号, 名), v["sha256"][:16], 参考表[i],
                 量._sha256(p)[:16], 同))
    print("-" * 100)
    print("合成对拍：18 个基线 wav 全同 =", 全同)
    print("时长表 =", {("%s·%s" % (s, n)): 合["样本"][(s, n)]["时长秒"] for s, n in 序表})
    raise SystemExit(0 if 全同 else 1)
