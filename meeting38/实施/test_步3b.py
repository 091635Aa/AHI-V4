# -*- coding: utf-8 -*-
"""test_步3b.py —— 自我线程+容量桥 单元判据（09 规格 §3 A10 前置）。

跑法：python test_步3b.py
"""
from __future__ import annotations

import sys, threading, time
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1\meeting38")
from 共存态 import 共存态
from 器官_我 import 我器


def main():
    态 = 共存态(种子=3800)
    我 = 我器(态, 种子=310)
    # 预投事件（模拟手/眼回流）
    for i in range(5):
        态.投事件({"类": "点击", "拍": i, "文本": "按钮%d" % i, "源": "手"})
    到点 = threading.Event()
    t = threading.Thread(target=我.运行, args=(到点,), daemon=True)
    t.start()
    time.sleep(30)                       # 首次 import 综合体链耗数秒（实测 ~8s），放宽窗口
    到点.set()
    t.join(timeout=3)
    槽 = 态.取自我()
    c = []
    c.append(("自体拍推进", 槽["自体拍"] >= 4))
    c.append(("内核已启动", 我.桥 is not False and 我.桥 is not None))
    c.append(("综合体拍执行", 我.统计["自体拍数"] >= 4 and 槽["遥测"].get("拍", 0) >= 0))
    c.append(("事件已消费", 我.统计["事件消费"] >= 5))
    c.append(("FSW 库入库", 我.统计["库记录"] >= 5))
    c.append(("自我槽向量非空", 槽["向量"] is not None))
    c.append(("类型覆盖可读", isinstance(槽["类型覆盖"], dict)))
    # 库可检索：用一条已入记录的束查询 → 命中自身
    查 = 我.容量.核.检索内容(list(我.容量.核.记录.values())[0]["束"], k=4)
    c.append(("库检索命中", 查["top1"] is not None))
    c.append(("内核零异常", 我.统计["内核异常"] == 0))
    全过 = all(ok for _, ok in c)
    for 名, ok in c:
        print("  %s %s" % ("PASS" if ok else "FAIL", 名))
    print("A10 前置（自我线程+容量桥）:", "PASS" if 全过 else "FAIL")
    return 0 if 全过 else 1


if __name__ == "__main__":
    sys.exit(main())
