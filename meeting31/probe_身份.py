# -*- coding: utf-8 -*-
"""meeting31/probe_身份.py —— #262 身份模板探针（开发用）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

from synthesis import 综合体
from identity import 身份模板

桥 = 综合体(seed=311)
模板 = 身份模板.从文件(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "身份模板示例.json"))
桥.加载身份(模板)
print("【身份已加载】%s（私偏κ=%g）" % (模板.名字, 桥.私偏κ))
print()
for q in ["你是谁？", "你平时喜欢做什么？", "你讨厌什么？", "你为什么叫小澄？"]:
    r = 桥.步(q)
    print("你> %s" % q)
    print("他> %s" % r["响应"])
    print()
