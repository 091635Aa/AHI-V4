# -*- coding: utf-8 -*-
"""V3主程序.py —— AHI V3 根启动器（薄壳）。

真正的总装在 meeting38/实施/V3主程序.py；本文件把实施目录加进 sys.path 后按其 __main__ 运行。
用法（在 AHI3 目录下）：
    python V3主程序.py                 # 四模块一条龙演示
    python V3主程序.py --看网页 本地   # 看一屏本地演示页
    python V3主程序.py --交互          # 纯文本对话
    python V3主程序.py --自主 60       # 自主运行 60 拍
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IMPL = os.path.join(HERE, "meeting38", "实施")
for p in (IMPL, os.path.join(HERE, "meeting38"), HERE, os.path.join(HERE, "meeting31")):
    if p not in sys.path:
        sys.path.insert(0, p)

if __name__ == "__main__":
    runpy.run_path(os.path.join(IMPL, "V3主程序.py"), run_name="__main__")