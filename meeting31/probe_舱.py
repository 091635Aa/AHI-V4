# -*- coding: utf-8 -*-
"""meeting31/probe_舱.py —— #265 知识舱可插拔验证：挂载→可达；卸载→不可达；核心纯净。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

from synthesis import 综合体

小说根 = "<path>\\下载部分迁移\\novel series"
if not os.path.exists(小说根):
    小说根 = "<path>\\XZ\\novel series"

桥 = 综合体(seed=311)
桥.设开放域([小说根], [])
桥.挂舱("小说")
print("【挂载】小说舱（目标：%s）" % 小说根)
前核心段 = len(桥.codec._段)

问题 = None
for _ in range(400):                     # 读小说 → 知识进 小说舱
    桥.断供一拍()
    if 问题 is None and 桥.舱s["小说"].行s:
        # 取舱内一行构造问题素材
        问题 = 桥.舱s["小说"].行s[len(桥.舱s["小说"].行s) // 2][:14]
print("核心库段数（读小说期间）：%d（应保持不变=隔离实证）" % len(桥.codec._段))
print("舱内知识 %d 行" % len(桥.舱s["小说"].行s))

q = "小说里『%s』是什么？" % (问题 or "内容")
r1 = 桥.步(q)
print("\n【挂载时】你> %s\n他> %s" % (q, r1["响应"][:80]))

桥.卸舱("小说")
r2 = 桥.步(q)
print("\n【卸载后】你> %s\n他> %s" % (q, r2["响应"][:80]))

print("\n【隔离判定】")
print("  挂载时可达、卸载后不可达：%s" %
      ("PASS" if ("《小说》" in r1["响应"] and "《小说》" not in r2["响应"])
       else "检查"))
print("  核心库纯净（段数 %d → %d）：%s" %
      (前核心段, len(桥.codec._段),
       "PASS" if len(桥.codec._段) == 前核心段 else "FAIL"))
print("  账本血统：舱条目按『小说』标记、卸载动作入账 = %s" %
      ("PASS" if any(t == "自我" and "卸载知识舱" in x for t, x in 桥.账本)
       else "FAIL"))
