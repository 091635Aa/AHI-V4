# -*- coding: utf-8 -*-
"""meeting31/probe_输出.py —— #253 单次最长输出探针：链式条件生成直到整句重复。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

from synthesis import 综合体
from writer import 写作体

桥 = 综合体(seed=311)
桥.设开放域(["<workspace>/"], ["<workspace>/\03_AHI\\AHI2.1\\backup_20261004_v10自言体"])
# 先读一会儿，让 lm_m 有语料
for _ in range(200):
    桥.断供一拍()
print("细读语料 %d 行后开始写作" % sum(1 for t, _ in 桥.账本 if t == "教材"))

写 = 写作体(桥, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "输出", "单次最长.txt"))
os.makedirs(os.path.dirname(写.路径), exist_ok=True)
r = 写.单次最长输出(上限句=400)
print("单次最长输出：句数 %d ｜ 字数 %d ｜ 唯一句 %d ｜ 触发=%s"
      % (r["句数"], r["字数"], r["唯一句"], r["触发"]))
print("—— 作品开头 8 句 ——")
with open(写.路径, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 8:
            break
        print("  %s" % line.rstrip())
print("—— 作品结尾 4 句 ——")
with open(写.路径, encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines[-4:]:
        print("  %s" % line.rstrip())
