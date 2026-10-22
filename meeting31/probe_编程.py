# -*- coding: utf-8 -*-
"""meeting31/probe_编程.py —— #264 编程考试（诚实基线）：任务=写余弦相似度函数。

生成=以任务描述与细读过的 Python 行为条件，从 lm_m 条件采样 40 个候选代码块；
评分=compile() 语法可编译率 + def/return/cos 结构率。预期偏低=诚实边界。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

import numpy as np

from synthesis import 综合体

任务 = "写一个函数计算两个向量的余弦相似度"

桥 = 综合体(seed=311)
桥.设开放域(["<workspace>/"],
           ["<workspace>/\03_AHI\\AHI2.1\\backup_20261004_v10自言体"])
print("备考：读 500 拍 Python 语料…")
for _ in range(500):
    桥.断供一拍()
py行s = [x for t, x in 桥.账本 if t == "教材"]
print("细读语料 %d 行（含 Python 代码行）" % len(py行s))

候选s = []
可编译 = 0
有结构 = 0
最佳 = None
for i in range(40):
    题材 = py行s[i % max(len(py行s), 1)] if py行s else 任务
    条件 = (任务[-6:] + 题材[-4:])[-8:]
    行s = []
    尾 = ""
    for j in range(8):                              # 链式生成 8 行
        rng = np.random.default_rng((311, 桥.拍, 31, i, j))
        s = 桥.lm_m.生成(种子=(条件 + 尾)[-8:], 温度=0.8, rng=rng, 最大长=44)
        if len(s) >= 4:
            行s.append(s)
            尾 = s[-2:]
    代码 = "\n".join(行s)
    候选s.append(代码)
    结构 = ("def " in 代码) and ("return" in 代码)
    有结构 += int(结构)
    try:
        compile(代码, "候选%d" % i, "exec")
        可编译 += 1
        if 最佳 is None or len(代码) > len(最佳):
            最佳 = 代码
    except SyntaxError:
        pass

print("=" * 60)
print("【编程考试】任务：%s" % 任务)
print("候选 40 个：语法可编译 %d（%.1f%%）｜含 def+return 结构 %d（%.1f%%）"
      % (可编译, 可编译 / 40 * 100, 有结构, 有结构 / 40 * 100))
if 最佳:
    print("—— 最佳可编译样本 ——")
    print(最佳)
else:
    print("—— 无一可编译（字符级模型无程序语义——诚实边界，基线入账）——")
    print("最近似样本（含 return 的行）：")
    for 代码 in 候选s:
        for 行 in 代码.splitlines():
            if "return" in 行:
                print("  " + 行[:70])
                break
        else:
            continue
        break
