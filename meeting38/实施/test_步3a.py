# -*- coding: utf-8 -*-
"""test_步3a.py —— 三槽追加的并发锤击对照（09 规格 §3 A11 口径）。

跑法：python test_步3a.py
"""
from __future__ import annotations

import sys, threading, time
sys.path.insert(0, r"<workspace>/03_AHI\AHI2.1\meeting38")
from 共存态 import 共存态

def main():
    态 = 共存态()
    停 = threading.Event()
    错误 = []

    def 锤(名, 函数):
        while not 停.is_set():
            try:
                函数()
            except Exception as e:
                错误.append("%s: %r" % (名, e))
                return

    def 六槽():
        态.写视觉([], 帧号=1); 态.取视觉()
        态.写听觉(); 态.取听觉()
        态.写意图(当前="x", 强度=0.5, 来源="t"); 态.取意图()
        态.写动作(状态="忙"); 态.取动作()
        态.排队说("句子"); 态.取下一句(); 态.取表达()
        态.写语义(按钮s={}); 态.取语义()

    def 三槽写者():
        """自我槽/计划槽按契约是单写者——本线程是唯一写者，断言可强。"""
        while not 停.is_set():
            try:
                态.写自我(提案="p", 自体拍=态.当前拍(), 类型覆盖={"复盘": 2}, 审计={"G1": True})
                me = 态.取自我()
                assert isinstance(me["自体拍"], int) and me["提案"] == "p"
                态.写计划(当前="任务A", 待办=["s1", "s2"])
                s = 态.取待办()
                assert s is None or isinstance(s, str)
                态.推待办("s9"); 态.取计划()
            except Exception as e:
                错误.append("三槽写者: %r" % e)
                return

    def 事件与读():
        while not 停.is_set():
            try:
                态.投事件({"类": "点击", "x": 1})
                态.投事件({"类": "滚动"})
                ev = 态.取事件(上限=64)
                assert isinstance(ev, list)
                态.取自我(); 态.取计划(); 态.取视场()   # 多读者
            except Exception as e:
                错误.append("事件与读: %r" % e)
                return

    def 视场槽写者():
        """视场槽按契约是单写者（眼）——本线程是唯一写者。"""
        i = 0
        import numpy as _np
        while not 停.is_set():
            try:
                i += 1
                视场 = _np.sign(_np.random.default_rng(i).standard_normal(1024)).astype(_np.float32)
                逐格 = _np.linspace(0.0, 1.0, 48, dtype=_np.float32)
                态.写视场(视场=视场, 注视=(i % 8, i % 6, i % 3), 误差场=视场, 意外度=0.5,
                          逐格意外=逐格, 眼跳数=i)
                v = 态.取视场()
                assert v["意外度"] == 0.5 and len(v["注视链"]) <= 64
            except Exception as e:
                错误.append("视场槽写者: %r" % e)
                return

    线程s = [threading.Thread(target=锤, args=("六槽#%d" % i, 六槽), daemon=True) for i in range(3)] \
          + [threading.Thread(target=锤, args=("三槽写者", 三槽写者), daemon=True)] \
          + [threading.Thread(target=锤, args=("视场槽写者", 视场槽写者), daemon=True)] \
          + [threading.Thread(target=锤, args=("事件与读#%d" % i, 事件与读), daemon=True) for i in range(2)]
    for t in 线程s:
        t.start()
    time.sleep(6)
    停.set()
    for t in 线程s:
        t.join(timeout=2)
    度 = 态.取度量()
    最长 = 度["最长写ms"]
    总数 = sum(度["读写次数"].values())
    ok = not 错误 and 最长 <= 20.0
    print("并发锤击 6s：操作数=%d 最长写ms=%.2f 错误=%d" % (总数, 最长, len(错误)))
    for e in 错误[:5]:
        print("  ", e)
    print("A11 并发不退化:", "PASS" if ok else "FAIL", "（门槛 最长写ms≤20）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
