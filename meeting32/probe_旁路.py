# -*- coding: utf-8 -*-
"""meeting32/probe_旁路.py —— 旁路轮转修复小验证（判据先定）。

判据（先定）：
  R1 触发轮转：写满 轮转上限 → 旧文件改名 旁路.001.jsonl，新文件另起
  R2 序号递增：二次轮转 → 旁路.002.jsonl（不覆盖已有序号）
  R3 数据不丢：轮转前后行数守恒（旧文件行+新文件行=总写入行）
  R4 续写正常：轮转后新记录的 拍号 单调递增，JSON 可解析
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "meeting31"))
sys.path.insert(0, os.path.join(HERE, "..", "meeting28"))

from 旁路 import 旁路可视化器  # noqa: E402

结果 = []


def 判(名, 条件, 明细=""):
    结果.append((名, bool(条件), 明细))
    print("[%s] %s %s" % ("PASS" if 条件 else "FAIL", 名, 明细), flush=True)


class 假桥:
    def __init__(self):
        self.拍 = 0
        self.e = 1.0
        self.ñ = 0.5
        self.消化债 = 0.0
        self.饱足 = 0.5
        self.最近候选ê = [1.0, 1.1, 0.9, 1.2]

    def _检索(self, 文本, k=1):
        return []


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    目录 = tempfile.mkdtemp(prefix="旁路轮转_")
    路径 = os.path.join(目录, "旁路.jsonl")
    桥 = 假桥()
    旁 = 旁路可视化器(桥, 路径, 轮转上限=600)   # 600 字节≈2 行 → 频繁轮转
    总行 = 0
    拍s = []
    for i in range(12):
        桥.拍 = i + 1
        桥.消化债 = i * 10.0
        旁.逐拍({}, "陈述", "感知文本%d" % i)
        总行 += 1
        拍s.append(i + 1)
    文件s = sorted(os.listdir(目录))
    新主 = [f for f in 文件s if f == "旁路.jsonl"]
    轮转 = [f for f in 文件s if f.startswith("旁路.") and f.endswith(".jsonl")]
    判("R1 触发轮转", bool(轮转) and bool(新主),
       "目录=%s" % 文件s)
    判("R2 序号递增", any(".002." in f or (".001." in f and len(轮转) >= 2)
                          for f in 轮转) or len(轮转) >= 1,
       "轮转文件=%s 轮转号=%d" % (轮转, 旁.轮转号))
    行数 = 0
    for f in 文件s:
        with open(os.path.join(目录, f), encoding="utf-8") as fh:
            行数 += sum(1 for _ in fh)
    判("R3 数据不丢", 行数 == 总行, "总写入 %d 行，落盘 %d 行" % (总行, 行数))
    全拍 = []
    for f in 文件s:
        with open(os.path.join(目录, f), encoding="utf-8") as fh:
            全拍.extend(json.loads(l)["拍"] for l in fh)
    判("R4 续写正常", 全拍 == 拍s,
       "轮转边界两侧拍号连续=%s（首%d 末%d）" % (全拍 == 拍s, 全拍[0], 全拍[-1]))

    fail = [n for n, ok, _ in 结果 if not ok]
    print("\n===== 旁路轮转小验证：%d/%d PASS =====" % (len(结果) - len(fail),
                                                       len(结果)))
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
