# -*- coding: utf-8 -*-
"""meeting32/probe_反馈.py —— 自我听觉反馈闭环小验证（判据先定）。

判据（先定）：
  F1 闭环运转：挂载后每次 步() → 声觉舱自动 +1（说完听自己）
  F2 自我认知：重新问同样的话（同文本→同响应→同声音）→ 声觉舱 top-1 命中自己的响应
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "meeting31"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "meeting28"))

from synthesis import 综合体  # noqa: E402
from 声带 import 声带器官  # noqa: E402
from 耳朵 import 耳朵器官  # noqa: E402

结果 = []


def 判(名, 条件, 明细=""):
    结果.append((名, bool(条件), 明细))
    print("[%s] %s %s" % ("PASS" if 条件 else "FAIL", 名, 明细), flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    HERE = os.path.dirname(os.path.abspath(__file__))
    台账 = os.path.join(HERE, "反馈_发声台账.jsonl")
    if os.path.exists(台账):
        os.remove(台账)
    带 = 声带器官(seed=300, D=1024, 目录=os.path.join(HERE, "声音"),
                 台账路径=台账)
    耳 = 耳朵器官(seed=400, D=1024)
    桥 = 综合体(seed=777)
    带.接综合体(桥, 耳=耳)

    前 = len(耳.舱)
    r1 = 桥.步("你好")
    r2 = 桥.步("你是谁？")
    判("F1 闭环运转", len(耳.舱) == 前 + 2,
       "步2次→声觉舱 %d 条（每步自动听+学自己）" % len(耳.舱))

    # F2 自我认知：同问句→同响应→同声音→检索命中（从台账取 wav 重听）
    末 = 带.台账流(1)[0]
    wav路径 = os.path.join(带.目录, 末["wav"])
    检 = 耳.辨(耳.听(wav路径), k=1)[0]
    判("F2 自我认知", 检["文本"] == 末["文本"] and 检["相似度"] > 0.99,
       "重听自己(wav)→『%s…』相似度=%.4f" % (检["文本"][:20], 检["相似度"]))

    fail = [n for n, ok, _ in 结果 if not ok]
    print("\n===== 自我听觉反馈小验证：%d/%d PASS =====" % (len(结果) - len(fail),
                                                          len(结果)))
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
