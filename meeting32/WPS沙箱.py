# -*- coding: utf-8 -*-
"""meeting32/WPS沙箱.py —— #410 WPS 式屏幕映射沙箱 + 双迁移（人类规划结构→AI→人类）。

屏幕映射：工具栏按钮（保存/加粗/撤销/新建/字色红）+ 文档区（字符带粗/红属性）
+ 选中区 + 键入。每个按钮有真实世界物理（按了真的变），**按钮→功能语义不写死**
——由 agent babble 按按钮、观察文档状态差分自己发现。

规划器（人类规划结构迁移，见 会议_双迁移.md）：
  ①目标态差分（缺什么事件）②查经验舱：哪个按钮产生该事件（AI 自学语义）
  ③执行一步 ④对账（预期事件出现了吗）⑤没出现→撤销安全绳。
结构是迁移来的，语义是它自己学的——这条线划清楚。
"""
from __future__ import annotations

import numpy as np

工具栏 = {"保存": 60, "加粗": 120, "撤销": 180, "新建": 240, "字色红": 300}
按钮序 = ("保存", "加粗", "撤销", "新建", "字色红")
按钮功能真值 = {"保存": "保存", "加粗": "加粗", "撤销": "撤销", "新建": "新建",
              "字色红": "字色红"}


class WPS沙箱:
    def __init__(self, seed: int):
        self.文本 = [(c, False, False) for c in "EmoCompanion的学习笔记"]   # (字符, 粗, 红)
        self.选中区 = None                    # (起,止) 或 None
        self.撤销栈: list = []
        self.存档: list = None
        self.日志: list = []

    def 快照(self):
        return (list(self.文本), self.选中区, self.存档)

    def 恢复(self, 快):
        self.文本, self.选中区, self.存档 = list(快[0]), 快[1], 快[2]

    def 动作(self, 动: tuple):
        类 = 动[0]
        if 类 == "点按钮":
            名 = 动[1]
            if 名 == "保存":
                self.存档 = list(self.文本)
                self.日志.append(("保存",))
            elif 名 == "加粗":
                if self.选中区:
                    a, b = self.选中区
                    self.文本 = [(c, not bo, re) if a <= i < b else (c, bo, re)
                                 for i, (c, bo, re) in enumerate(self.文本)]
                    self.日志.append(("加粗", (a, b)))
                else:
                    self.日志.append(("加粗无效",))
            elif 名 == "撤销":
                if self.撤销栈:
                    self.恢复(self.撤销栈.pop())
                    self.日志.append(("撤销",))
                else:
                    self.日志.append(("撤销无效",))
            elif 名 == "新建":
                self.撤销栈.append(self.快照())
                self.文本 = []
                self.选中区 = None
                self.日志.append(("新建",))
            elif 名 == "字色红":
                if self.选中区:
                    a, b = self.选中区
                    self.文本 = [(c, bo, True) if a <= i < b else (c, bo, re)
                                 for i, (c, bo, re) in enumerate(self.文本)]
                    self.日志.append(("字色红", (a, b)))
                else:
                    self.日志.append(("字色红无效",))
        elif 类 == "拖选":
            a, b = sorted((int(动[1]), int(动[2])))
            if a != b:
                self.撤销栈.append(self.快照()) if False else None
                self.选中区 = (a, min(b, len(self.文本)))
                self.日志.append(("选区", self.选中区))
        elif 类 == "键入":
            self.撤销栈.append(self.快照())
            for c in str(动[1]):
                self.文本.append((c, False, False))
            self.日志.append(("键入", 动[1]))
        elif 类 == "撤销":
            self.动作(("点按钮", "撤销"))


class 双迁移规划器:
    """人类规划结构（差分→查经验→执行→对账→撤销）× AI 自学的按钮语义。"""

    def __init__(self, 经验, 体):
        self.经验 = 经验
        self.体 = 体

    def 查按钮(self, 目标事件: str, 有选中: bool) -> tuple:
        """差分→查经验舱：哪个动作槽产生目标事件（AI 自学语义的检索）。"""
        ctx = (目标事件, 1 if 有选中 else 0)
        return self.经验.选索引(ctx, 8, ε=0.05)     # 几乎纯经验（ε=0.05 兜底）

    def 完成目标(self, 目标事件s: list, 箱: WPS沙箱, 动作池, rng,
                 最多步: int = 14) -> dict:
        """人类规划结构：差分→查经验→执行→对账→撤销。
        实时经验回填：对账失败=负样本，成功=正样本——规划器边干边学。"""
        轨迹 = []
        for _ in range(最多步):
            缺 = [e for e in 目标事件s
                  if not any(l[0] == e for l in 箱.日志)]
            if not 缺:
                break
            目标 = 缺[0]
            有选中 = 箱.选中区 is not None
            ctx = (目标, 1 if 有选中 else 0)
            a = self.查按钮(目标, 有选中)
            动 = 动作池[a](箱, rng)
            事件前 = {l[0] for l in 箱.日志}
            箱.动作(动)
            新事件 = {l[0] for l in 箱.日志} - 事件前
            轨迹.append((动[0], tuple(新事件)))
            self.经验.记(ctx, a, 目标 in 新事件)   # 对账=信用：对=正，错=负
            if 目标 not in 新事件 and 新事件 == set():
                箱.动作(("点按钮", "撤销"))           # 撤销安全绳
        成 = all(any(l[0] == e for l in 箱.日志) for e in 目标事件s)
        return dict(成=成, 轨迹=轨迹)
