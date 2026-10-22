# -*- coding: utf-8 -*-
"""meeting32/probe_文件表格.py —— 沙箱 v2 课程验证（判据先定）。

判据（先定）：
  W1 右键发现：课A 平均步数后段 < 前段（发现「右键文件」比「右键空白」有效）
  W2 删除链自发：课B（续课A经验）后 50 集 ≥85%（右键→菜单→点删除 三步链）
  W3 重命名链自发：课C（续课B）后 50 集 ≥75%（右键→重命名→键入 四步链）
  W4 表格课：后 50 集「A1 写EmoCompanion并保存」≥80%（三步链）
  W5 确定性：同 seed 重跑全课表 → 成功率序列摘要一致
课程式学习：课A/B/C 共享同一经验舱（先会右键，再学删除，再学重命名——像人）。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "meeting31"))
sys.path.insert(0, os.path.join(HERE, "..", "meeting28"))

from 自发学习 import 运动经验舱  # noqa: E402
from 沙箱进阶 import 文件沙箱, 表格沙箱, 格名位, 格子位  # noqa: E402

结果 = []
键入词 = ("测试甲", "新名1", "abc", "EmoCompanion", "报告副本")


def 判(名, 条件, 明细=""):
    结果.append((名, bool(条件), 明细))
    print("[%s] %s %s" % ("PASS" if 条件 else "FAIL", 名, 明细), flush=True)


def 跑课(经验, 集数, 成功f, 动作池, 上下文f, seed, 步上限=14, 表格=False) -> list:
    """babble→步内稠密信用（动作按当场引发的世界事件记账）。
    ε 退火：0.35→0.05（探索率随经验下降，像人越熟越少乱试）。"""
    rng = np.random.default_rng(seed + 集数)
    率s, 步s = [], []
    for ep in range(集数):
        箱 = 文件沙箱(seed * 1000 + ep) if not 表格 else 表格沙箱(seed * 1000 + ep)
        ε = 0.35 * (1.0 - ep / max(集数 - 1, 1)) + 0.05
        本集 = []
        for 步 in range(步上限):
            ctx = 上下文f(箱)
            a = 经验.选索引(ctx, len(动作池), ε=ε)
            事件前 = len(箱.日志)
            动 = 动作池[a % len(动作池)](箱, rng)
            箱.动作(动)
            本集.append((ctx, a, len(箱.日志) > 事件前))
            if 成功f(箱):
                break
        成 = 成功f(箱)
        for ctx, a, 事件 in 本集:
            经验.记(ctx, a, 事件 and 成)   # 事件∧集成功：成功路径上的事件动作才得正分
        率s.append(成)
        步s.append(len(本集))
    return 率s, 步s


def 文件动作池():
    """瞄准=感知（它看得见文件/菜单项，动作槽=瞄准目标）；选哪个槽=它学的。"""
    def 右键文件(箱, rng):
        if not 箱.文件s:
            return ("Esc",)
        f = 箱.文件s[int(rng.integers(len(箱.文件s)))]
        return ("右键", f["x"], f["y"])

    def 右键报告(箱, rng):
        f = next((x for x in 箱.文件s if x["名"] == "报告.txt"), None)
        return ("右键", f["x"], f["y"]) if f else ("Esc",)

    def 右键空白(箱, rng):
        return ("右键", float(rng.integers(400, 900)), float(rng.integers(380, 560)))

    def 左键报告(箱, rng):
        f = next((x for x in 箱.文件s if x["名"] == "报告.txt"), None)
        return ("左键", f["x"], f["y"]) if f else ("Esc",)

    def 点菜单项0(箱, rng):
        return 菜单点击(箱, 0, rng)

    def 点菜单项1(箱, rng):
        return 菜单点击(箱, 1, rng)

    def 点菜单项2(箱, rng):
        return 菜单点击(箱, 2, rng)

    def 点菜单项3(箱, rng):
        return 菜单点击(箱, 3, rng)

    def 拖到文件夹(箱, rng):
        if not 箱.文件s:
            return ("Esc",)
        d = 箱.文件夹s[int(rng.integers(len(箱.文件夹s)))]
        return ("拖到", d["x"], d["y"])

    def 键入(箱, rng):
        return ("键入", 键入词[int(rng.integers(len(键入词)))])

    def 回车(箱, rng):
        return ("回车",)

    def esc(箱, rng):
        return ("Esc",)

    return [右键文件, 右键报告, 右键空白, 左键报告, 点菜单项0, 点菜单项1,
            点菜单项2, 点菜单项3, 拖到文件夹, 键入, 回车, esc]


def 菜单点击(箱, k, rng):
    if 箱.菜单开 is None:                      # 没菜单时兜底：右键目标文件
        f = next((x for x in 箱.文件s if x["名"] == "报告.txt"), None)
        return ("右键", f["x"], f["y"]) if f else ("Esc",)
    _, mx, my = 箱.菜单开
    return ("左键", mx + 30, my + 24 * k + 12)


def 文件上下文(箱, 目标: str = "") -> tuple:
    """目标入上下文：同一界面，想做的不一样→动作不一样（人知道自己的目标）。
    没有这一维，课B 的删除经验会灾难性遗忘课 C 的重命名。"""
    return (目标,
            1 if 箱.菜单开 is not None else 0,
            箱.菜单项位() if 箱.菜单开 is not None else -1,
            {"空白": 0, "文件": 1, "文件夹": 2}[箱.光标下()] if 箱.菜单开 is None
            else 3,
            1 if 箱.选中 else 0,
            1 if 箱.改名中 else 0)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    池 = 文件动作池()
    经验 = 运动经验舱(seed=700)

    # —— 课A 右键发现（120 集）：学习信号=平均步数下降 ——
    _, 步A = 跑课(经验, 120,
                 lambda 箱: 箱.菜单开 is not None and 箱.菜单开[0] == "文件",
                 池, 文件上下文, 31)
    前A = float(np.mean(步A[:12]))
    后A = float(np.mean(步A[-12:]))
    判("W1 右键发现", 后A < 前A,
       "平均步数 前 %.1f → 后 %.1f（发现右键文件有效）" % (前A, 后A))

    # —— 课B 删除链（250 集，续课A经验）——
    率B, _ = 跑课(经验, 250,
                 lambda 箱: any(e[0] == "删除" for e in 箱.日志),
                 池, 文件上下文, 41)
    后B = float(np.mean(率B[-50:]))
    判("W2 删除链自发", 后B >= 0.85,
       "后 50 集 %.2f（右键→菜单→点删除 三步链）" % 后B)

    # —— 课C 重命名链（250 集，续课B）——
    率C, _ = 跑课(经验, 250,
                 lambda 箱: any(e[0] == "改名" and e[1][0] == "报告.txt"
                                for e in 箱.日志),
                 池, 文件上下文, 51)
    后C = float(np.mean(率C[-50:]))
    判("W3 重命名链自发", 后C >= 0.75,
       "后 50 集 %.2f（右键→重命名→键入 四步链）" % 后C)

    # —— 课D 表格课（150 集，独立世界）——
    表经验 = 运动经验舱(seed=760)
    表rng = np.random.default_rng(761)

    def 表动作池():
        """瞄准=感知：12 个格子各一个动作槽 + 键入任务文本 + 保存。"""
        def 点格(格):
            名, 位 = 格, 格子位[格]
            return lambda 箱, rng: ("左键", 位[0], 位[1])
        池 = [点格(名) for 名, _ in 格名位]

        def 键入任务文本(箱, rng):
            # 打什么字=任务给定（人说的话）；学的是何时何处打
            return ("键入", "EmoCompanion")

        def 保存(箱, rng):
            return ("保存",)
        池 += [键入任务文本, 保存]
        return 池

    def 表上下文(箱) -> tuple:
        return (1 if 箱.选中格 == "A1" else 0,
                1 if 箱.内容.get("A1") == "EmoCompanion" else 0)

    表率, _ = 跑课(表经验, 150,
                  lambda 箱: 箱.内容.get("A1") == "EmoCompanion" and 箱.已保存,
                  表动作池(), 表上下文, 61, 步上限=12, 表格=True)
    后D = float(np.mean(表率[-50:]))
    判("W4 表格课", 后D >= 0.80,
       "后 50 集 %.2f（A1=EmoCompanion+保存，三步链）" % 后D)

    # —— W5 确定性：同 seed 重跑课A-C ——
    def 全课表():
        e = 运动经验舱(seed=700)
        r1, _ = 跑课(e, 120,
                     lambda 箱: 箱.菜单开 is not None and 箱.菜单开[0] == "文件",
                     文件动作池(), 文件上下文, 31)
        r2, _ = 跑课(e, 250,
                     lambda 箱: any(x[0] == "删除" for x in 箱.日志),
                     文件动作池(), 文件上下文, 41)
        r3, _ = 跑课(e, 250,
                     lambda 箱: any(x[0] == "改名" and x[1][0] == "报告.txt"
                                    for x in 箱.日志),
                     文件动作池(), 文件上下文, 51)
        return r1 + r2 + r3

    摘要 = lambda 率s: hashlib.sha256(
        json.dumps([int(x) for x in 率s]).encode()).hexdigest()[:12]
    再率 = 全课表()
    判("W5 确定性", 摘要(再率[120:370]) == 摘要(率B)
       and 摘要(再率[370:620]) == 摘要(率C),
       "同 seed 重跑课表摘要一致")

    fail = [n for n, ok, _ in 结果 if not ok]
    print("\n===== 沙箱 v2 课程验证：%d/%d PASS =====" % (len(结果) - len(fail),
                                                         len(结果)))
    if fail:
        print("FAIL:", fail)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
