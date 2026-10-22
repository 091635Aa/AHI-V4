# -*- coding: utf-8 -*-
"""meeting31/probe_旁路.py —— #272 亚思考旁路演示：逐拍把向量遥测翻译成自然语言。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

from synthesis import 综合体
from 旁路 import 旁路可视化器

桥 = 综合体(seed=311)
桥.设开放域(["<workspace>/"],
           ["<workspace>/\03_AHI\\AHI2.1\\backup_20261004_v10自言体"])
旁 = 旁路可视化器(桥, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "旁路.jsonl"))
print("亚思考旁路演示（前 15 拍，每拍=向量遥测的自然语言翻译）：\n")
for i in range(15):
    导 = 桥.导航器
    感知文本 = 导.当前文件 or (导.栈[-1][0] if 导.栈 else "(导航中)")
    桥.断供一拍()
    动作 = 桥.动作史[-1] if 桥.动作史 else "-"
    print("拍%3d %s" % (i + 1, 旁.逐拍({}, 动作, 感知文本)))
print("\n完整 JSONL：%s" % 旁.路径)
print("（并见 自言流：符号中介的推理轨迹——两条旁路互补）")
