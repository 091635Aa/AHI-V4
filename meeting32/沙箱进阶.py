# -*- coding: utf-8 -*-
"""meeting32/沙箱进阶.py —— #400 沙箱 v2：文件操作 + 右键菜单 + WPS式表格（自发学习延伸）。

所有者令：在基础沙盒里学拖拽、右键、文件基础操作；再给他 WPS/表格看看——
他没经历过 WPS。铁律不变：**零手写操作流程**——只有动作空间、世界物理、
结果感知和计数关联经验舱。他会什么，全看他自己摸到过什么。

文件沙箱（世界物理）：
  对象=文件(名,位)与文件夹(名,位,内容数)；右键文件→菜单[打开,重命名,删除,复制]；
  右键空白→菜单[新建,刷新]；菜单开着时左键菜单项=执行：
    删除→文件-1；重命名→下一条 键入 生效；复制→「副本」+1；新建→「新建文件」+1；
  左键文件=选中；拖文件到文件夹=移入（文件夹内容+1）；Esc=关菜单；回车=确认改名。
表格沙箱（WPS式）：
  4×4 格（A1..D4）；左键格子=选中；键入=写入选中格；Ctrl+S=保存（保存后
  内容锁定进「存档」）。任务：在 A1 写「EmoCompanion」并保存——三步链，自己摸。
"""
from __future__ import annotations

import math

import numpy as np

from 自发学习 import 运动经验舱  # noqa: F401  (babbler 复用其计数机制)

菜单 = {"文件": ["打开", "重命名", "删除", "复制"],
       "空白": ["新建", "刷新"]}

格子位 = {"A1": (110, 110), "B1": (190, 110), "C1": (270, 110), "D1": (350, 110),
          "A2": (110, 170), "B2": (190, 170), "C2": (270, 170), "D2": (350, 170),
          "A3": (110, 230), "B3": (190, 230), "C3": (270, 230), "D3": (350, 230)}
格名位 = list(格子位.items())


class 文件沙箱:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.重置()

    def 重置(self):
        self.文件s = [{"名": "报告.txt", "x": 120, "y": 150},
                      {"名": "笔记.txt", "x": 120, "y": 240},
                      {"名": "数据.txt", "x": 260, "y": 150}]
        self.文件夹s = [{"名": "归档", "x": 620, "y": 150, "内容": 0}]
        self.光标 = [420.0, 420.0]
        self.菜单开 = None                      # None 或 (类, x, y)
        self.选中 = None
        self.改名中 = None
        self.日志: list = []                    # 世界事件（结果感知源）

    def 光标下(self) -> str:
        for f in self.文件s:
            if abs(self.光标[0] - f["x"]) < 40 and abs(self.光标[1] - f["y"]) < 30:
                return "文件"
        for d in self.文件夹s:
            if abs(self.光标[0] - d["x"]) < 40 and abs(self.光标[1] - d["y"]) < 30:
                return "文件夹"
        if self.菜单开 is not None:
            return "菜单项"
        return "空白"

    def 菜单项位(self) -> int:
        """菜单开着时，光标指在第几个菜单项（每项高 24）。"""
        if self.菜单开 is None:
            return -1
        _, mx, my = self.菜单开
        idx = int((self.光标[1] - my) // 24)
        类 = self.菜单开[0]
        if 0 <= idx < len(菜单[类]):
            return idx
        return -1

    def 动作(self, 动: tuple):
        """动=(类, 参数)。类∈{左键,右键,拖到,键入,回车,Esc}。世界物理全在这。"""
        类 = 动[0]
        if 类 == "移":
            self.光标 = [float(动[1]), float(动[2])]
        elif 类 == "左键":
            self.光标 = [float(动[1]), float(动[2])]
            if self.菜单开 is not None:
                类型, mx, my = self.菜单开
                idx = self.菜单项位()
                项 = 菜单[类型][idx] if idx >= 0 else None
                self.菜单开 = None
                if 项 == "删除" and 类型 == "文件" and self.选中:
                    self.文件s = [f for f in self.文件s if f["名"] != self.选中]
                    self.日志.append(("删除", self.选中))
                elif 项 == "重命名" and 类型 == "文件" and self.选中:
                    self.改名中 = self.选中
                    self.日志.append(("改名模式", self.选中))
                elif 项 == "复制" and 类型 == "文件" and self.选中:
                    源 = next((f for f in self.文件s if f["名"] == self.选中), None)
                    if 源:
                        self.文件s.append({"名": 源["名"].replace(".", "副本."),
                                           "x": 源["x"] + 30, "y": 源["y"] + 30})
                        self.日志.append(("复制", self.选中))
                elif 项 == "新建" and 类型 == "空白":
                    self.文件s.append({"名": "新建文件.txt", "x": self.光标[0],
                                       "y": self.光标[1]})
                    self.日志.append(("新建", "新建文件.txt"))
                return
            hit = next((f for f in self.文件s
                        if abs(self.光标[0] - f["x"]) < 40
                        and abs(self.光标[1] - f["y"]) < 30), None)
            self.选中 = hit["名"] if hit else None
            if self.改名中 and self.选中 is None:
                self.改名中 = None
        elif 类 == "右键":
            self.光标 = [float(动[1]), float(动[2])]
            类型 = self.光标下()
            hit = next((f for f in self.文件s
                        if abs(self.光标[0] - f["x"]) < 40
                        and abs(self.光标[1] - f["y"]) < 30), None)
            if hit:
                self.选中 = hit["名"]                # 真实资源管理器：右键即选中
            if self.菜单开 is None:
                self.菜单开 = (类型 if 类型 in ("文件", "文件夹") else "空白",
                               self.光标[0], self.光标[1])
                self.日志.append(("菜单开", 类型))
        elif 类 == "拖到":
            目x, 目y = float(动[1]), float(动[2])
            抓 = next((f for f in self.文件s
                       if abs(self.光标[0] - f["x"]) < 40
                       and abs(self.光标[1] - f["y"]) < 30), None)
            if 抓:
                夹 = next((d for d in self.文件夹s
                           if abs(目x - d["x"]) < 45 and abs(目y - d["y"]) < 35),
                          None)
                if 夹:
                    self.文件s = [f for f in self.文件s if f["名"] != 抓["名"]]
                    夹["内容"] += 1
                    self.日志.append(("移入", (抓["名"], 夹["名"])))
                else:
                    抓["x"], 抓["y"] = 目x, 目y
            self.光标 = [目x, 目y]
        elif 类 == "键入":
            文 = 动[1]
            if self.改名中 and 文:
                self.文件s = [({**f, "名": 文} if f["名"] == self.改名中 else f)
                              for f in self.文件s]
                self.日志.append(("改名", (self.改名中, 文)))
                self.改名中 = None
        elif 类 == "回车":
            self.改名中 = None
        elif 类 == "Esc":
            self.菜单开 = None
            self.改名中 = None


class 表格沙箱:
    def __init__(self, seed: int):
        self.内容: dict = {}
        self.选中格 = None
        self.已保存 = False
        self.保存时刻内容: dict = {}
        self.日志: list = []                    # 世界事件（稠密信用源）

    def 动作(self, 动: tuple):
        类 = 动[0]
        if 类 == "左键":
            x, y = float(动[1]), float(动[2])
            hit = min(格名位, key=lambda kv: math.hypot(x - kv[1][0], y - kv[1][1]))
            if math.hypot(x - hit[1][0], y - hit[1][1]) < 38:
                if self.选中格 != hit[0]:
                    self.日志.append(("选中", hit[0]))
                self.选中格 = hit[0]
        elif 类 == "键入":
            if self.选中格 and self.内容.get(self.选中格) != 动[1]:
                self.内容[self.选中格] = 动[1]
                self.日志.append(("写入", self.选中格))
        elif 类 == "保存":
            if not self.已保存:
                self.日志.append(("保存",))
            self.已保存 = True
            self.保存时刻内容 = dict(self.内容)


class babbler:
    """自发动作发生器：动作空间均匀探索 + 经验舱引导（ε-greedy）。"""

    键入词 = ("测试甲", "新名1", "abc", "EmoCompanion", "报告副本")
    格子名 = ["A1", "B1", "C1", "A2", "B2", "C2"]

    def __init__(self, 经验: 运动经验舱, rng):
        self.经验 = 经验
        self.rng = rng
        self.上动作 = None
        self.上上下文 = None

    @staticmethod
    def 上下文(箱) -> tuple:
        类型 = 箱.光标下()
        菜 = 1 if 箱.菜单开 is not None else 0
        菜项 = 箱.菜单项位() if 菜 else -1
        return (菜, {"空白": 0, "文件": 1, "文件夹": 2, "菜单项": 3}[类型], 菜项)

    def 动(self, 箱) -> tuple:
        ctx = self.上下文(箱)
        a = self.经验.选动作(("沙箱2",) + ctx, ε=0.35)   # 复用经验舱 ε-greedy
        类 = ["左键", "右键", "拖到", "键入", "回车", "Esc"][a[0] % 6]
        if 类 == "键入":
            动 = ("键入", self.键入词[int(self.rng.integers(len(self.键入词)))])
        elif 类 == "拖到":
            动 = ("拖到", float(self.rng.integers(80, 900)),
                  float(self.rng.integers(100, 500)))
        else:
            动 = (类, float(self.rng.integers(60, 940)),
                  float(self.rng.integers(100, 560)))
        self.上上下文, self.上动作 = ctx, a
        return 动

    def 学(self, 成: bool):
        if self.上上下文 is not None:
            self.经验.记(("沙箱2",) + self.上上下文, self.上动作, 成)
