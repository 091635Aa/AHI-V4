# -*- coding: utf-8 -*-
"""meeting69/实施/算子_流分离.py —— S1 流分离（活跃图 4-邻域 BFS；09 §2.5）。

输入：ε 的 32×32 网格；活跃格 = ε > mean(ε)+1.0·std(ε)。
连通块 ≥4 格算一个流；与上一拍流表按格重叠率 ≥0.5 继承流 id；否则按
(最小时槽, 最小带) 字典序开新流；流存活拍上限 20，超时删。
全程确定性（固定遍历顺序与破平规则）。
"""
from __future__ import annotations

import numpy as np

带数 = 32
槽数 = 32
最少格 = 4
存活上限 = 20
重叠阈 = 0.5


def 活跃格(ε):
    g = np.asarray(ε, np.float64).reshape(带数, 槽数)
    return g > (g.mean() + 1.0 * g.std()), g


def 连通块(活):
    """4-邻域 BFS；返回按（最小时槽, 最小带）排序的连通块（每块 = 格 (b,slot) 集合）。"""
    已 = np.zeros_like(活, bool)
    块 = []
    for b in range(带数):
        for s in range(槽数):
            if not 活[b, s] or 已[b, s]:
                continue
            栈 = [(b, s)]; 已[b, s] = True; 格 = []
            while 栈:
                x, y = 栈.pop()
                格.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    u, v = x + dx, y + dy
                    if 0 <= u < 带数 and 0 <= v < 槽数 and 活[u, v] and not 已[u, v]:
                        已[u, v] = True
                        栈.append((u, v))
            块.append(sorted(格, key=lambda p: (p[1], p[0])))
    块.sort(key=lambda g: (min(s for _, s in g), min(b for b, _ in g)))
    return [set(g) for g in 块 if len(g) >= 最少格]


def 分离(ε, 上拍流表, 拍号):
    """返回 (流表, 指派)。流表: dict[id] = dict(格=set[(b,slot)], 龄=int, 始=拍号)。"""
    活, _ = 活跃格(ε)
    块 = 连通块(活)
    流表 = {}
    已用id = set()
    新流 = []
    for 格 in 块:
        最佳 = None; 最佳率 = 0.0
        for sid, st in 上拍流表.items():
            旧 = st["格"]
            交 = len(格 & 旧)
            率 = 交 / max(1, min(len(格), len(旧)))
            if 率 > 最佳率:
                最佳, 最佳率 = sid, 率
        if 最佳 is not None and 最佳率 >= 重叠阈 and 最佳 not in 已用id:
            流表[最佳] = dict(格=格, 龄=0, 始=上拍流表[最佳]["始"])
            已用id.add(最佳)
        else:
            新流.append(格)
    # 旧流未匹配：龄 +1，超时删
    for sid, st in 上拍流表.items():
        if sid in 已用id:
            continue
        龄 = st["龄"] + 1
        if 龄 <= 存活上限:
            流表[sid] = dict(格=st["格"], 龄=龄, 始=st["始"])
    # 新流按字典序开 id
    nid = (max(流表) + 1) if 流表 else 0
    for 格 in 新流:
        while nid in 流表:
            nid += 1
        流表[nid] = dict(格=格, 龄=0, 始=拍号)
    指派 = np.full((带数, 槽数), -1, np.int32)
    for sid, st in 流表.items():
        for b, s in st["格"]:
            指派[b, s] = sid
    return 流表, 指派


def 流统计(流表, ε):
    """每流：格数、平均 ε。"""
    g = np.asarray(ε, np.float64).reshape(带数, 槽数)
    out = {}
    for sid, st in 流表.items():
        格 = st["格"]
        out[sid] = dict(格数=len(格), 平均ε=float(np.mean([g[b, s] for b, s in 格])) if 格 else 0.0)
    return out
