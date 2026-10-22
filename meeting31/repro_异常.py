# -*- coding: utf-8 -*-
"""meeting31/repro_异常.py —— 复现长跑中的每拍 AttributeError（开发用）。"""
import sys
import traceback

sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1\meeting31")
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1")

from synthesis import 综合体

桥 = 综合体(seed=311)
桥.设开放域(["<path>\\", "<path>\\", "<workspace>/"],
           ["<workspace>/\03_AHI\\AHI2.1\\backup_20261004_v10自言体"])
for i in range(150):
    try:
        桥.断供一拍()
    except Exception:
        traceback.print_exc()
        print("--- 拍", i, "栈顶", 桥.导航器.栈[-1][0] if 桥.导航器.栈 else None,
              "当前文件", 桥.导航器.当前文件)
        break
else:
    print("150 拍无异常; 已读", 桥.导航器.已读文件,
          "访问目录", len(桥.导航器.访问))
