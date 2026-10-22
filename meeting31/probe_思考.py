# -*- coding: utf-8 -*-
"""meeting31/probe_思考.py —— #270 延答内驱探针：ack→多拍思考→投递+延迟实测。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

from synthesis import 综合体

桥 = 综合体(seed=311)
桥.设开放域(["<workspace>/"],
           ["<workspace>/\03_AHI\\AHI2.1\\backup_20261004_v10自言体"])

print("【①知识型问题→自发延答】")
桥.步("苹果是一种水果。")                       # 教两跳事实（无单条直接含答案）
桥.步("水果富含维生素。")
t0 = time.time()
r = 桥.步("苹果对人有什么好处？")
print("你> 苹果对人有什么好处？")
print("他> %s（即时表达，耗时 %.2fs）" % (r["响应"], time.time() - t0))

n0 = len(桥.回复台账)
while 桥.思考中 is not None and time.time() - t0 < 60:
    桥.断供一拍()
延迟 = time.time() - t0
投递 = 桥.回复台账[-1] if len(桥.回复台账) >= n0 + 1 else None
if 投递 and 投递 is not 桥.回复台账[n0 - 1 if n0 else 0]:
    pass
print("他> %s（思考 %s 拍后作答，总延迟 %.2fs）"
      % (桥.回复台账[-1]["响应"][:80], 桥.回复台账[-1].get("延迟拍", "?"), 延迟))

print("\n【②证据足的问题→立即答（对照）】")
桥.步("我最喜欢的数字是 9。")
t1 = time.time()
r = 桥.步("我最喜欢的数字是 9，对吗？")
print("你> 我最喜欢的数字是 9，对吗？")
print("他> %s（延迟 %.2fs——证据足不延答）" % (r["响应"][:60], time.time() - t1))

print("\n【③诚实饱和】")
r = 桥.步("量子引力如何量子化？")
print("你> 量子引力如何量子化？")
print("他> %s" % r["响应"])
n0 = len(桥.回复台账)
t2 = time.time()
while 桥.思考中 is not None and time.time() - t2 < 60:
    桥.断供一拍()
投 = 桥.回复台账[-1]
print("他> %s（思考 %s 拍——饱和诚实）" % (投["响应"][:60], 投.get("延迟拍", "?")))
print("\n自言审计：", 桥.自言.审计(桥.账本)["通过"])
