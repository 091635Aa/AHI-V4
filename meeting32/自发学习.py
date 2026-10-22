# -*- coding: utf-8 -*-
"""meeting32/自发学习.py —— #390 自发性运动学习：沙箱 babble → 经验涌现（无写死轨迹）。

所有者令：不是写死代码教他会，而是在模拟沙箱里让他自己学——自发性的像人。
本文件没有任何一条手写的"应该怎么拖"——只有：
  动作空间（步长比例档×微偏档）、结果感知（离目标近了/远了、入框/没入框）、
  和一个 HDC 计数关联经验舱（上下文→动作→成败计数）。
策略=经验舱的 argmax（拉普拉斯平滑）+ ε 探索。他会什么，全看他摸到过什么。

课表（沙箱内自练）：
  课1 够着：随机球位+光标位，babble 移动，结果=离球更近？（20 步内够到=成）
  课2 拖放：够球→按下→往框走→松手，结果=入框计数+1？（成=入框）
自发涌现预期（判据先定，写死=false 的证明）：
  X2 最优步长比例自发收敛到 [0.45,0.7]——比例控制律是它自己发现的；
  X1 学习曲线显著下降；X3 拖放成功率 ≥90%；X5 同 seed 重学同策略。
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
沙箱界 = (0, 0, 1000, 700)                    # 模拟画布界（无头）


class 运动经验舱:
    """HDC 计数关联：上下文→动作→成败。策略=经验 argmax+ε。无任何手写策略。"""

    步长档 = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)   # 误差比例动作
    偏档 = (-2, -1, 0, 1, 2)                        # 垂直微偏档

    def __init__(self, seed: int = 400):
        self.rng = np.random.default_rng(seed)
        self.表: dict = {}                     # 上下文键 → {动作键: [成,试]}

    @staticmethod
    def 上下文(err: tuple) -> tuple:
        d = math.hypot(*err)
        距档 = min(int(d / 60.0), 7)                    # 60px 一档
        角 = math.atan2(err[1], err[0])
        向档 = int((角 + math.pi) / (math.pi / 4)) % 8   # 8 方向
        return (距档, 向档)

    def 选动作(self, 上下文键, ε=0.2) -> tuple:
        if self.rng.random() < ε or 上下文键 not in self.表:
            return (int(self.rng.integers(len(self.步长档))),
                    int(self.rng.integers(len(self.偏档))))
        表 = self.表[上下文键]
        best, best分 = None, -1.0
        for a, (成, 试) in 表.items():
            分 = (成 + 1.0) / (试 + 2.0)                # 拉普拉斯平滑成功率
            if 分 > best分:
                best, best分 = a, 分
        return best

    def 选索引(self, 上下文键, n: int, ε=0.2) -> int:
        """通用 n 动作槽赌臂：课程动作池用（动作槽不受步长档限制）。
        并列时随机挑（防索引0死锁：冷启动全 0.5 并列时固定选 0 会恶性循环）。"""
        if self.rng.random() < ε or 上下文键 not in self.表:
            return int(self.rng.integers(n))
        表 = self.表[上下文键]
        分s = [( (表.get(a, (0, 0))[0] + 1.0) / (表.get(a, (0, 0))[1] + 2.0), a)
               for a in range(n)]
        best分 = max(分s)[0]
        并列 = [a for 分, a in 分s if abs(分 - best分) < 1e-9]
        return int(self.rng.choice(并列))

    def 记(self, 上下文键, 动作键, 成: bool):
        表 = self.表.setdefault(上下文键, {})
        cell = 表.setdefault(动作键, [0, 0])
        cell[0] += int(成)
        cell[1] += 1

    def 步(self, err: tuple) -> tuple:
        """策略一步：返回光标位移（自发=从经验采样）。"""
        ctx = self.上下文(err)
        a = self.选动作(ctx)
        d = math.hypot(*err)
        p = self.步长档[a[0]]
        偏 = self.偏档[a[1]]
        base = np.array(err, dtype=np.float64) * p
        垂直 = np.array([-err[1], err[0]], dtype=np.float64)
        n = np.linalg.norm(垂直) + 1e-9
        delta = base + 垂直 / n * 偏 * 8.0
        return delta, (ctx, a)

    def 学(self, ctx, a, 成: bool):
        self.记(ctx, a, 成)


class 沙箱:
    """无头模拟画布：球、框、光标。
    物理：抓中=球粘手直到松手；执行噪声=位移±15%∝步长（Weber 定律）——
    这条世界物理是「近目标减速」自发涌现的来源，不是写进策略的。"""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.重置()

    def 重置(self):
        self.球 = np.array([self.rng.uniform(120, 880),
                            self.rng.uniform(140, 560)])
        self.框心 = np.array([self.rng.uniform(600, 900),
                              self.rng.uniform(150, 550)])
        self.光标 = self.rng.uniform(100, 500, 2)
        self.按住 = False
        self.抓中 = False
        self.入框 = False

    def 移(self, delta) -> tuple:
        delta = np.asarray(delta, dtype=np.float64)
        n = float(np.linalg.norm(delta))
        if n > 1e-6:                                   # 执行噪声（∝步长）
            delta = delta + self.rng.normal(0, 0.15 * n, 2)
        self.光标 = np.clip(self.光标 + delta, 沙箱界[0], [沙箱界[2], 沙箱界[3]])
        if self.抓中:
            self.球 = self.光标.copy()                 # 抓中=球粘手
        return tuple(self.光标)

    def 下(self):
        self.按住 = True
        if np.hypot(*(self.光标 - self.球)) < 45:
            self.抓中 = True
            self.球 = self.光标.copy()

    def 上(self):
        self.按住 = False
        入框 = self.抓中 and np.hypot(*(self.光标 - self.框心)) < 130
        self.抓中 = False
        self.入框 = bool(入框)


感知阈 = 0.30                                   # 沙箱感官：进展≥30%才算"明显接近"


def 课1_够着(经验: 运动经验舱, 集: int, rng) -> list:
    """够着课：babble→结果→学。返回每集步数（学习曲线）。"""
    曲线 = []
    for ep in range(集):
        箱 = 沙箱(rng.integers(0, 1 << 30))
        目标 = 箱.球.copy()
        for 步 in range(20):
            err = (目标[0] - 箱.光标[0], 目标[1] - 箱.光标[1])
            d前 = math.hypot(*err)
            if d前 < 6:
                break
            delta, (ctx, a) = 经验.步(err)
            箱.移(delta)
            d后 = math.hypot(目标[0] - 箱.光标[0], 目标[1] - 箱.光标[1])
            进展 = (d前 - d后) / max(d前, 1.0)
            经验.学(ctx, a, 进展 >= 感知阈)      # 结果反馈：明显接近=成
        曲线.append(步)
    return 曲线


def 课2_拖放(经验: 运动经验舱, 集: int, rng, 回报: list) -> list:
    """拖放课：够球→按下→往框心走→松。结果=入框（任务级回报）。"""
    成率 = []
    for ep in range(集):
        箱 = 沙箱(rng.integers(0, 1 << 30))
        for _ in range(20):                      # 阶段1：够球（用已学策略）
            err = (箱.球[0] - 箱.光标[0], 箱.球[1] - 箱.光标[1])
            if math.hypot(*err) < 6:
                break
            delta, (ctx, a) = 经验.步(err)
            d前 = math.hypot(*err)
            箱.移(delta)
            d后 = math.hypot(箱.球[0] - 箱.光标[0], 箱.球[1] - 箱.光标[1])
            经验.学(ctx, a, (d前 - d后) / max(d前, 1.0) >= 感知阈)
        箱.下()
        for _ in range(25):                      # 阶段2：按下拖向框心（同一条策略）
            err = (箱.框心[0] - 箱.光标[0], 箱.框心[1] - 箱.光标[1])
            if math.hypot(*err) < 6:
                break
            delta, (ctx, a) = 经验.步(err)
            d前 = math.hypot(*err)
            箱.移(delta)
            d后 = math.hypot(箱.框心[0] - 箱.光标[0], 箱.框心[1] - 箱.光标[1])
            经验.学(ctx, a, (d前 - d后) / max(d前, 1.0) >= 感知阈)
        箱.上()
        回报.append(1 if 箱.入框 else 0)
        成率.append(箱.入框)
    return 成率


def 自发步长(经验: 运动经验舱) -> list:
    """它自己发现的控制律：各距离档下经验选出的步长比例。"""
    出 = []
    for 距档 in range(7):
        ctx = (距档, 0)
        表 = 经验.表.get(ctx, {})
        if not 表:
            continue
        best = max(表, key=lambda a: (表[a][0] + 1.0) / (表[a][1] + 2.0))
        出.append((距档, 运动经验舱.步长档[best[0]]))
    return 出
