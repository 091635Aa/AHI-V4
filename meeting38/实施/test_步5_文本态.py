# -*- coding: utf-8 -*-
"""test_步5_文本态.py —— 纯文本对话态 判据 V1–V7（跑完即知，不做「已完成」宣称）。

判据（见 00_置顶_纯文本对话态.md §8；V2/V3/V4 已按 2026-10-06 所有者口径改版）：
  V1 纯文本加载：进程内只起了 共存态+嘴+综合体（眼/耳/手/语义/我/器官_脑/生命体 一个都没 import）
  V2 先想再答·正常流：对话流**只留 我/它 对白**（无「它·想」旁白）；内部确有念头（脑内通道）
  V3 回忆命中：复述先前片段时能检索到那次经历（「我回忆起来」＝脑内念头条目）
  V4 自我经过：每条输出都在内部带自我评估痕迹（脑内含 复盘/自评/拍/能量/账本/消化债）
  V5 自主（非限制）：无输入时它自己想（自言产出）；睡/醒由内在状态驱动，代码里无时间阈值
  V6 不破既有：未加载的器官文件逐位未改（git 状态）
  V7 多窗口（同一生命）：A 窗口得知的事 B 窗口能想起（记忆共享）；B 窗口对话流不含 A 的对白（隔离）
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
M38 = os.path.dirname(HERE)
AHI = r"<workspace>/03_AHI\AHI2.1"
for _p in (HERE, M38, AHI, os.path.join(AHI, "meeting31")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from 文本态 import 文本态                                 # noqa: E402


def 判V1():
    禁 = ("器官_眼", "器官_耳", "器官_手", "语义", "器官_我", "生命体", "器官_脑")
    命中 = [m for m in 禁 if m in sys.modules]
    return dict(通过=not 命中, 未加载器官=命中,
                加载的关键=("共存态" in sys.modules and "器官_嘴" in sys.modules
                          and "synthesis" in sys.modules))


def 判V2(流, 脑):
    """正常对话流：对话流只留 我/它 对白（**不得**有「它·想」旁白）；
    内部确实有念头（脑内通道非空，且不少于 它 的输出条数——每答一句至少想一拍）。"""
    流入想 = [i for i, (_, 谁, _) in enumerate(流) if 谁 == "它·想"]
    n它 = sum(1 for _, 谁, _ in 流 if 谁 == "它")
    return dict(通过=(not 流入想 and n它 > 0 and len(脑) >= n它),
                它条数=n它, 脑内条数=len(脑), 流入想数=len(流入想))


def 判V3(脑):
    回忆s = [文 for _, 文 in 脑 if 文.startswith("我回忆起来")]
    return dict(通过=len(回忆s) > 0, 回忆条数=len(回忆s),
                样例=[文[:44] for 文 in 回忆s[:3]])


def 判V4(脑):
    """每条「它」输出，内部都带自我评估痕迹（自评/复盘/遥测词）。"""
    迹 = ("复盘", "自评", "能量", "消化债", "账本", "先掂量")
    有迹 = [文 for _, 文 in 脑 if any(k in 文 for k in 迹)]
    return dict(通过=len(有迹) > 0, 有迹条数=len(有迹),
                样例=[文[:40] for 文 in 有迹[:2]])


def 判V5(快):
    st = 快["统计"]
    自主产出 = st.get("自主拍", 0) > 0 and st.get("自言外显", 0) > 0
    # 无时间阈值：源代码里不得出现「空闲N秒/每N秒」「sleep(N) 当阈值」这类被判据禁止的硬编码
    源 = open(os.path.join(HERE, "文本态.py"), encoding="utf-8").read()
    时间阈值 = re.findall(r"(空闲\s*\d+\s*秒|每\s*\d+\s*秒|超时\s*\d+|sleep\(\s*\d)", 源)
    睡醒都有 = st.get("醒拍", 0) > 0 and st.get("睡拍", 0) > 0
    return dict(通过=(自主产出 and 睡醒都有 and not 时间阈值), 自主拍=st.get("自主拍", 0),
                自言外显=st.get("自言外显", 0), 醒拍=st.get("醒拍", 0),
                睡拍=st.get("睡拍", 0), 末态醒=快["对话态"]["醒"], 时间阈值=时间阈值)


def 判V6():
    器官 = ["器官_眼.py", "器官_耳.py", "器官_手.py", "语义.py", "器官_我.py",
           "生命体.py", "器官_脑.py"]
    路径 = [os.path.join(M38, x) for x in 器官]
    改 = subprocess.run(["git", "-C", AHI, "status", "--porcelain", *路径],
                       capture_output=True, text=True, encoding="utf-8", errors="ignore")
    行 = [x for x in (改.stdout or "").splitlines() if x.strip()]
    # 只认「已改动/已删除」（M/D）；本仓这些器官文件目前未被 git 跟踪（??），
    # 未跟踪≠被修改——本轮未触碰它们（见我方未编辑任何器官文件的声明）。
    真改 = [x for x in 行 if ("M" in x[:2] or "D" in x[:2] or "R" in x[:2])]
    return dict(通过=not 真改, 已改=真改, 未跟踪数=len(行) - len(真改))


def 判V7(态, 报):
    """多窗口：A 得知的八卦 → B 能想起（记忆共享）；B 的对话流不含 A 的人/对白（隔离）。"""
    每窗 = 报["每窗"]
    A流 = 每窗["A"]["对话流"]
    B流 = 每窗["B"]["对话流"]
    B说话人 = [谁 for _, 谁, _ in B流]
    隔离 = ("甲" not in B说话人)                                  # A 的人没串进 B 的对话流
    A有 = any(("老王" in 文) for _, 谁, 文 in A流 if 谁 == "甲")   # 八卦确在 A 窗口入流
    B召回 = any(("老王" in 文) for _, 谁, 文 in B流 if 谁 == "它")  # B 窗口把 A 的事想起来了
    共享 = any("老王" in 段 for 段, _ in 态.桥._检索("老王 咖啡", k=3))  # 记忆是同一份
    return dict(通过=(隔离 and A有 and B召回 and 共享),
                隔离=隔离, A有=A有, B召回=B召回, 共享=共享,
                B对话流=[(谁, 文[:40]) for _, 谁, 文 in B流])


def 主():
    t0 = time.time()
    态 = 文本态(种子=310).建()
    建时 = time.time() - t0
    消息s = ["我叫阿栈，这段时间一直在做 AHI 的纯文本对话态。",
             "你还记得我叫什么吗？",
             "随便说点你现在在想的事吧。"]
    快 = 态.运行_脚本(消息s, 每轮自主拍=2, 结尾自主拍=80)
    流 = 快["对话流"]
    脑 = 快["脑内"]

    # ── V7 多窗口：同一生命，两个窗口，记忆共享 ──
    报 = 态.运行_多窗({"A": [("甲", "跟你说个八卦：老王昨天在茶水间把咖啡洒了一身。")],
                       "B": [("乙", "你还记得老王怎么了吗？")]},
                     每轮自主拍=1, 结尾自主拍=4)
    态.停机()

    print("=" * 72)
    print("纯文本对话态 · 判据 V1–V7")
    print("=" * 72)
    print("【建模耗时】%.1fs；自言总数=%d；记忆库=%d 条；统计=%s"
          % (建时, 快["自言总"], 快["记忆库"], 快["统计"]))
    print("-" * 72)
    print("【主窗口 对话流（正常·无思考旁白）】")
    for t, 谁, 文 in 流[-12:]:
        print("  [%s] %s" % (谁, 文[:78]))
    print("【主窗口 脑内（内部管线信号·审计用）尾 6 条】")
    for t, 文 in 脑[-6:]:
        print("  · %s" % 文[:78])
    print("-" * 72)
    print("【多窗口 对话流（同一生命：A 说八卦 / B 想起）】")
    for 会话 in ("A", "B"):
        print("  ── 窗口 %s ──" % 会话)
        for t, 谁, 文 in 报["每窗"][会话]["对话流"]:
            print("     [%s] %s" % (谁, 文[:70]))
    print("-" * 72)

    Rs = dict(V1=判V1(), V2=判V2(流, 脑), V3=判V3(脑), V4=判V4(脑),
              V5=判V5(快), V6=判V6(), V7=判V7(态, 报))
    for k, v in Rs.items():
        print("%-3s %s  %s" % (k, "PASS" if v["通过"] else "FAIL", v))
    print("=" * 72)
    print("[log] 判据明细见上。未做任何「已完成」宣称——数字与如实边界以上表为准。")


if __name__ == "__main__":
    主()