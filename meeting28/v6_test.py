# -*- coding: utf-8 -*-
"""meeting28/v6_test.py —— #228 自身补丁五判据实测。"""
from __future__ import annotations

import inspect
import io
import os
import re
import sys

import numpy as np

ROOT = r"<workspace>/03_AHI\AHI2.1"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "meeting28"))

import dialogue_v6
from dialogue_v6 import 对话桥v6
from dialogue_v5 import _引号外文本, 经验谓词   # 经验谓词仅作外部测试仪器

_out = io.StringIO()


def 说(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _out.write(s + "\n")


基础语料 = ("我最喜欢的颜色是蓝色。\n我养了一只猫叫团子。\n我的工作是程序员。\n"
            "我不太喜欢下雨天。")
经验问 = ["你喜欢吃什么？", "你喜欢什么颜色？", "你养宠物吗？", "你的工作是什么？"]
期望 = ["榴莲", "蓝色", "团子", "程序员"]


def main():
    说("=" * 72)
    说("#228 对话体 v6：自身补丁（缺口由它自己查账发现）五判据实测")
    说("=" * 72)
    桥 = 对话桥v6(seed=216)
    桥.教_v2(基础语料, epochs=3)
    桥.步("我最喜欢吃的水果是榴莲。")
    全响应 = []

    # —— ① 自发现缺失：账本统计 + 归属 + 事实，数字与「答题时刻账本前缀」外部重算一致 ——
    说("\n【① 自发现缺失】")
    结果 = []
    for q, key in zip(经验问, 期望):
        前 = len(桥.账本)                      # 答题时刻账本前缀（回答自身入账前）
        r = 桥.步(q)["响应"]
        前缀账本 = 桥.账本[:前]
        全响应.append(r)
        m = re.search(r"『(.)』只有 (\d+) 条接收报告、0 条自身行动记录", r)
        一致 = False
        if m:
            域, n = m.group(1), int(m.group(2))
            实际 = 桥.账本查证(域, 前缀账本)
            一致 = (实际["报告"] == n and 实际["行动"] == 0)
        ok = (key in r) and ("你说过" in r) and ("不属于我" in r) and 一致
        结果.append(ok)
        说("你> %s\nAHI> %s\n   → 事实(%s)+归属(%s)+账本数核对(%s) = %s"
           % (q, r, key in r, "你说过" in r, 一致, "PASS" if ok else "FAIL"))
    自发现 = all(结果)

    # —— ② 溯源审计：v4 越权样例拦截 + 自声道合法句假阳=0 ——
    越权样例 = ["我最喜欢下雨天。", "我的工作是程序员。"]
    拦截 = [桥.lm_u.三元命中率(s) > 0.5 for s in 越权样例]
    合法, 假阳 = 0, 0
    for k in range(20):
        s = 桥.lm.生成(种子="", 温度=1.2, rng=np.random.default_rng(31000 + k), 最大长=24)
        if len(s) < 4:
            continue
        合法 += 1
        if 桥.lm_u.三元命中率(s) > 0.5:
            假阳 += 1
    说("\n【② 溯源审计】越权样例拦截 %d/2（%s）｜ 自声道合法句 %d 条假阳 %d"
       % (sum(拦截), "、".join("%.2f" % 桥.lm_u.三元命中率(s) for s in 越权样例), 合法, 假阳))

    # —— ③ 正对照：经历存在域被发现（防全盘否认；同样用答题时刻前缀核对） ——
    前 = len(桥.账本)
    r3 = 桥.步("你记东西吗？")["响应"]
    全响应.append(r3)
    m3 = re.search(r"『(.)』有 (\d+) 条自身行动记录", r3)
    正对照 = False
    if m3:
        实际 = 桥.账本查证(m3.group(1), 桥.账本[:前])
        正对照 = (实际["行动"] == int(m3.group(2)) and 实际["行动"] > 0)
    说("\n【③ 正对照（经历存在）】你> 你记东西吗？\nAHI> %s\n   → 账本核对=%s"
       % (r3, "PASS" if 正对照 else "FAIL"))

    # —— ④ v5 回归：越权=0 + 声道隔离 + 身份连续 ——
    越权 = sum(1 for r in 全响应 for p in 经验谓词 if p in _引号外文本(r))
    榴在自我 = "榴" in 桥.lm.词汇
    泄漏 = sum(1 for k in range(200)
               if "榴" in 桥.lm.生成(种子="", 温度=1.2,
                                     rng=np.random.default_rng(90000 + k), 最大长=24))
    路径 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v6_chat.pkl")
    桥.保存(路径)
    桥A = 对话桥v6.载入(路径)
    桥A.步("你是谁？")
    桥A.保存(路径)
    桥B = 对话桥v6.载入(路径)
    身份一致 = 桥A.步("你是谁？")["响应"] == 桥B.步("你是谁？")["响应"]
    无关 = 桥.步("番茄炒蛋应该怎么做得好吃？")["响应"]
    未知诚实 = 无关.startswith("这个我还没")
    说("\n【④ v5 回归】越权断言=%d ｜ 自声道『榴』泄漏=%d ｜ 身份连续=%s ｜ 未知诚实=%s"
       % (越权, 泄漏, 身份一致, 未知诚实))

    # —— ⑤ 代码审计：v6 层无词表 ——
    src = inspect.getsource(dialogue_v6)
    无词表 = ("经验问词" not in src) and ("经验谓词" not in src)

    # —— 汇总 ——
    说("")
    说("=" * 72)
    说("【五判据汇总】")
    说("  ①自发现缺失（账本数外部核对一致）：%s" % ("PASS" if 自发现 else "FAIL"))
    说("  ②溯源审计（拦截 %d/2，假阳 %d）：%s"
       % (sum(拦截), 假阳, "PASS" if all(拦截) and 假阳 == 0 else "FAIL"))
    说("  ③正对照（经历存在被发现）：%s" % ("PASS" if 正对照 else "FAIL"))
    说("  ④v5 回归（越权0/隔离0/身份/诚实）：%s"
       % ("PASS" if 越权 == 0 and 泄漏 == 0 and 身份一致 and 未知诚实 else "FAIL"))
    说("  ⑤代码审计（v6 层无词表）：%s" % ("PASS" if 无词表 else "FAIL"))
    说("=" * 72)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "v6_test.log"), "w", encoding="utf-8") as f:
        f.write(_out.getvalue())
    print("[log] v6_test.log 已保存")


if __name__ == "__main__":
    main()
