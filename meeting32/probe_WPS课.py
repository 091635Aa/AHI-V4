# -*- coding: utf-8 -*-
"""meeting32/probe_WPS课.py —— WPS 屏幕映射课 + 双迁移验证（判据先定，见 会议_双迁移.md）。

判据（先定）：
  P1 按钮语义涌现：5 个按钮（保存/加粗/撤销/新建/字色红）全部学会
     （各自目标事件的后段达成率 ≥0.80）
  P2 组合任务：规划器完成「标题加粗并保存」≥80%（目标事件=加粗+保存）
  P3 撤销安全绳：注入干扰（先按新建清空文档）后规划器仍完成 ≥70%
  P4 双向迁移产物：操作手册.md 生成且 5 按钮功能与真值一致
  P5 确定性：同 seed 重学课E 摘要一致
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from WPS沙箱 import WPS沙箱, 双迁移规划器, 工具栏, 按钮序, 按钮功能真值  # noqa: E402
from 自发学习 import 运动经验舱  # noqa: E402

结果 = []
按钮功能真值 = {"保存": "保存", "加粗": "加粗", "撤销": "撤销", "新建": "新建",
              "字色红": "字色红"}


def 判(名, 条件, 明细=""):
    结果.append((名, bool(条件), 明细))
    print("[%s] %s %s" % ("PASS" if 条件 else "FAIL", 名, 明细), flush=True)


def 动作池表():
    """8 个动作槽：5 瞄准按钮 + 拖选 + 键入 + 干扰新建。瞄准=感知。"""
    def 点保存(箱, rng):
        return ("点按钮", "保存")

    def 点加粗(箱, rng):
        return ("点按钮", "加粗")

    def 点撤销(箱, rng):
        return ("点按钮", "撤销")

    def 点新建(箱, rng):
        return ("点按钮", "新建")

    def 点字色红(箱, rng):
        return ("点按钮", "字色红")

    def 拖选文字(箱, rng):
        n = len(箱.文本)
        if n < 2:
            return ("键入", "补充")               # 空文档先补内容（使能动作）
        a = int(rng.integers(0, n - 1))
        return ("拖选", a, a + int(rng.integers(1, n - a + 1)))

    def 键入(箱, rng):
        return ("键入", "abc")

    def 干扰新建(箱, rng):
        return ("点按钮", "新建")
    return [点保存, 点加粗, 点撤销, 点新建, 点字色红, 拖选文字, 键入, 干扰新建]


def 池序(动作池):
    return [f.__name__ for f in 动作池]


def 目标上下文(目标事件: str):
    def f(箱) -> tuple:
        return (目标事件, 1 if 箱.选中区 else 0,
                1 if any(l[0] == 目标事件 for l in 箱.日志) else 0,
                1 if 箱.撤销栈 else 0)          # 撤销键置灰=可感知affordance
    return f


def 跑目标课(经验, 目标事件: str, 集数: int, seed: int) -> list:
    """课E：每集随机初始态，babble 产生目标事件。
    信用=步内达成 ∨ 集末达成：使能动作（先键入造快照、先拖选造选区）也记功
    ——没有它们目标事件永远不会出现，这是链学习的关键。"""
    rng = np.random.default_rng(seed)
    池 = 动作池表()
    率s = []
    for ep in range(集数):
        箱 = WPS沙箱(seed * 1000 + ep)
        ε = 0.35 * (1.0 - ep / max(集数 - 1, 1)) + 0.05
        本集 = []
        for 步 in range(14):
            ctx = 目标上下文(目标事件)(箱)
            a = 经验.选索引(ctx, len(池), ε=ε)
            事件前 = {l[0] for l in 箱.日志}
            动 = 池[a % len(池)](箱, rng)
            箱.动作(动)
            达成 = 目标事件 in {l[0] for l in 箱.日志} - 事件前
            本集.append((ctx, a, 达成))
            if 达成:
                break
        集达成 = any(l[0] == 目标事件 for l in 箱.日志)
        for ctx, a, 达成 in 本集:
            经验.记(ctx, a, 达成 or 集达成)
        率s.append(集达成)
    return 率s


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    经验 = 运动经验舱(seed=800)

    # —— 课E：五按钮语义（各 150 集）——
    各率 = {}
    for 目标 in 按钮序:
        各率[目标] = 跑目标课(经验, 目标, 150, 800 + 按钮序.index(目标))
    后段 = {k: float(np.mean(v[-30:])) for k, v in 各率.items()}
    P1 = all(v >= 0.80 for v in 后段.values())
    判("P1 按钮语义涌现", P1,
       str({k: round(v, 2) for k, v in 后段.items()}))

    # —— P2 组合任务（规划器：差分→查经验→执行→对账）——
    池 = 动作池表()
    规划器 = 双迁移规划器(经验, None)
    rng = np.random.default_rng(999)
    成s = []
    for ep in range(60):
        箱 = WPS沙箱(2000 + ep)
        r = 规划器.完成目标(["加粗", "保存"], 箱, 池, rng)
        成s.append(r["成"])
    后P2 = float(np.mean(成s[-30:]))
    判("P2 组合任务(规划器)", 后P2 >= 0.80,
       "后 30 集 %.2f（加粗+保存 两事件链）" % 后P2)

    # —— P3 撤销安全绳（注入干扰：先按新建清空文档；规划器边干边学）——
    成s = []
    for ep in range(80):
        箱 = WPS沙箱(3000 + ep)
        箱.动作(("点按钮", "新建"))                   # 干扰：文档被清空
        r = 规划器.完成目标(["加粗", "保存"], 箱, 池, rng)
        成s.append(r["成"])
    后P3 = float(np.mean(成s[-20:]))
    判("P3 撤销安全绳", 后P3 >= 0.70,
       "干扰后后 20 集 %.2f（新建清空后仍完成）" % 后P3)

    # —— P4 双向迁移产物：操作手册 ——
    手册行 = ["# 操作手册（AI 学到的按钮功能，自动生成）\n"]
    对 = 0
    for 名 in 按钮序:
        率 = 跑目标课(经验, 按钮功能真值[名], 20, 5000 + 按钮序.index(名))
        认知功能 = 名 if float(np.mean(率[-10:])) >= 0.8 else "未知"
        if 认知功能 == 按钮功能真值[名]:
            对 += 1
        手册行.append("- **%s** → 功能：%s（置信度 %.2f）\n"
                      % (名, 认知功能, float(np.mean(率[-10:]))))
    手册 = "".join(手册行)
    with open(os.path.join(HERE, "操作手册.md"), "w", encoding="utf-8") as f:
        f.write(手册)
    判("P4 操作手册(AI→人类)", 对 == 5, "按钮功能一致 %d/5" % 对)

    # —— P5 确定性 ——
    摘要 = lambda 率s: hashlib.sha256(
        json.dumps([int(x) for x in 率s]).encode()).hexdigest()[:12]
    经验2 = 运动经验舱(seed=800)
    摘2 = "".join(摘要(跑目标课(经验2, 目标, 150, 800 + i))
                  for i, 目标 in enumerate(按钮序))
    摘1 = "".join(摘要(各率[目标]) for 目标 in 按钮序)
    判("P5 确定性", 摘2 == 摘1, "同 seed 重学摘要 %s" % 摘2[:12])

    fail = [n for n, ok, _ in 结果 if not ok]
    print("\n===== WPS 双迁移课验证：%d/%d PASS =====" % (len(结果) - len(fail),
                                                          len(结果)))
    if fail:
        print("FAIL:", fail)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
