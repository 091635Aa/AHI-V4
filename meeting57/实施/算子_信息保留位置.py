# -*- coding: utf-8 -*-
"""AHI4/meeting57/实施/算子_信息保留位置.py —— P0：信息保留位置消融（A / B / C / S 四臂）。

━━━ 这一件回答什么 ━━━
「窄递归状态（1024）单独回读」是否致命？「离散锚点/账本直传」是否更好？「宽状态上界」有多高？

  A 窄递归状态（D=1024）：所有事件写进同一个 1024 维叠加态；回读=解绑 + 硬阈值迭代吸引子清理；
  B 窄状态 + 离散锚点/账本旁路：关键事件（目标/约束/实体/细节/矛盾/更正）原样入账；受控回读：
    只对「任务相关」锚点按 (权限, 步) 取权威值；干扰键不入账 ⇒ 回读「未知」；
  C 宽状态/无瓶颈上界：每键一个 1024 维独立槽位（宽态），无叠加干扰；精确回读；
  S B + sham：随机锚点 / 打乱账本 / 惰性回读（零假设对照；不得优于 A）。

━━━ 铁律对齐 ━━━
C1  无注意力/无 S 字头全局归一化：回读用**硬阈值迭代吸引子**（Hopfield 式 sign 迭代），
    不调用 S 字头归一化，不含 QK^T / 全局注意力；判据另做源码级禁词零命中自检。
C13 随机只来自 default_rng(显式种子)；状态写入后经 `_归`（2^-20 定点格）量化；
C14 纯新增；只读 import meeting41 的 `_归/软限`（不修改）。
本件不含学习、不含外挂模型、不含 OCR；A/B/C/S 只是记忆载体，不冒充 V4 内核。

━━━ 直跑（冒烟）━━━
    python 算子_信息保留位置.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AHI4 = os.path.dirname(os.path.dirname(HERE))
_能力核py = os.path.join(AHI4, "meeting41", "实施", "算子_能力核.py")


def _载(路径, 名):
    spec = importlib.util.spec_from_file_location(名, 路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_能力 = _载(_能力核py, "能力核_只读_信息保留位置")
_归 = _能力._归
软限 = _能力.软限

# ── 冻结常量（判据不据结果改；开发集只校准这些值，判据/留出集只判不调）──
D默认 = 1024
ρ默认 = 0.995
状态幅 = 8.0
吸引子步数 = 2
回读阈值 = 0.10
干扰写权 = 0.25
账本容量 = 4096
随机锚点率 = 1.00          # sham：写入即随机锚点（内容不可用；结构保留）
打乱账本率 = 0.50          # sham：回读时打乱键→值
惰性回读率 = 0.30          # sham：回读旧缓存
缺失回读率 = 0.10          # sham：回读缺失
S种子偏移 = 990001


def _值域(键):
    if str(键) in ("goal", "constraint", "entity"):
        return 8
    return 16


class 码本:
    """确定性双极码本：键码 / 值码；值码固定 0..15。"""

    def __init__(self, 键列, D=D默认, 种子=0):
        self.D = int(D)
        self.种子 = int(种子)
        self.键列 = sorted({str(k) for k in 键列})
        rng = np.random.default_rng(self.种子 + 7717)
        self.键码 = {}
        for k in self.键列:
            self.键码[k] = (np.where(rng.integers(0, 2, size=self.D) == 1, 1.0, -1.0)
                          / np.sqrt(float(self.D)))
        self.值码 = (np.where(rng.integers(0, 2, size=(16, self.D)) == 1, 1.0, -1.0)
                   / np.sqrt(float(self.D)))

    def 取键码(self, 键):
        return self.键码[str(键)]

    def 绑(self, 键, 值):
        """k ⊗ v，乘 D 后逐元素 ∈ {−1,+1}；不是注意力/不是外积矩阵。"""
        return self.取键码(键) * self.值码[int(值)] * float(self.D)


def 吸引子清理(u, 值码, 步数=吸引子步数):
    """硬阈值迭代吸引子（Hopfield 式 sign 迭代；无 S 字头归一化、无全局注意力）。

    对解绑向量做 x_{t+1}=sign(Σ_i (v_i·x_t) v_i) 的定点迭代，最后归一化到
    「Rademacher 码」尺度（‖x‖=√D），便于与值码做内积读出。
    """
    x = np.asarray(u, dtype=np.float64).copy()
    V = 值码.shape[0]
    for _ in range(max(1, int(步数))):
        s = 值码 @ x
        x = 值码.T @ s / float(V)
        x = np.where(x >= 0.0, 1.0, -1.0)                  # 硬阈值，非 S 字头归一化
    n = float(np.linalg.norm(x)) / np.sqrt(float(x.shape[0]))
    if not np.isfinite(n) or n < 1e-12:
        return np.zeros_like(x)
    return x / n


class A区室:
    """A：纯窄递归状态（1024）。所有事件写进同一叠加态；回读=解绑+吸引子+阈值。"""

    名 = "A"

    def __init__(self, 任务, D=D默认, 种子=0, ρ=ρ默认, 写权=None,
                 清理步数=吸引子步数, **kw):
        self.任务 = 任务
        self.D = int(D)
        self.种子 = int(种子)
        self.ρ = float(ρ)
        self.清理步数 = int(清理步数)
        self.码 = 码本(任务["细节键列"] + list(任务["关键键列"]) + 任务["干扰键列"],
                       D=self.D, 种子=self.种子)
        self.h = np.zeros(self.D, dtype=np.float64)
        self.写权 = dict(写权 or {})

    def _权(self, e):
        t = str(e["类型"])
        if t in self.写权:
            return float(self.写权[t])
        if not e.get("任务相关", True):
            return float(干扰写权)
        return float(e.get("关注度", 1.0))

    def 写(self, 事件):
        if 事件["键"] not in self.码.键码:
            return
        w = self._权(事件)
        if w == 0.0:
            return
        self.h = _归(self.h + w * self.码.绑(事件["键"], 事件["值"]), 幅=64.0)

    def 步进(self, 步):
        self.h = _归(self.ρ * self.h, 幅=64.0)
        m = float(np.max(np.abs(self.h))) if self.h.size else 0.0
        if m > 状态幅:
            self.h = _归(self.h * (状态幅 / m), 幅=64.0)

    def 读(self, 键):
        if str(键) not in self.码.键码:
            return None
        k = self.码.取键码(键)
        u = k * self.h
        if self.清理步数 <= 0:
            s = self.码.值码 @ u
        else:
            x = 吸引子清理(u, self.码.值码, self.清理步数)
            s = self.码.值码 @ x
        i = int(np.argmax(s))
        if float(s[i]) < float(回读阈值):
            return None
        return i

    def 指纹(self):
        return "%.10f" % float(np.sum(self.h))


class B区室:
    """B：窄状态 + 离散锚点/账本旁路。关键事件原样入账；受控回读（权限优先，同权限取新）。"""

    名 = "B"

    def __init__(self, 任务, D=D默认, 种子=0, 账本容量=账本容量, **kw):
        self.任务 = 任务
        self.种子 = int(种子)
        self.容量 = int(账本容量)
        self.账本 = []

    def 写(self, 事件):
        if not 事件.get("任务相关", True):
            return
        if 事件["类型"] == "干扰":
            return
        self.账本.append(dict(键=str(事件["键"]), 值=int(事件["值"]),
                             权限=int(事件["权限"]), 步=int(事件["步"]),
                             序号=int(事件.get("序号", 0)), 类型=str(事件["类型"]),
                             关注度=float(事件.get("关注度", 1.0))))
        if len(self.账本) > self.容量:
            # 受控容量：优先级=权限，其次关注度，其次新近；淘汰最低
            self.账本.sort(key=lambda e: (e["权限"], e["关注度"], e["步"], e["序号"]))
            self.账本 = self.账本[len(self.账本) - self.容量:]

    def 步进(self, 步):
        return

    def _候选(self, 键):
        return [e for e in self.账本 if e["键"] == str(键)]

    def 读(self, 键):
        c = self._候选(键)
        if not c:
            return None
        e = max(c, key=lambda x: (x["权限"], x["步"], x["序号"]))
        return int(e["值"])

    def 指纹(self):
        return "账本%d" % len(self.账本)


class C区室:
    """C：宽状态/无瓶颈上界（每键一个 D 维独立槽位；精确回读；不是目标架构）。"""

    名 = "C"

    def __init__(self, 任务, D=D默认, 种子=0, 槽数=None, **kw):
        self.任务 = 任务
        self.D = int(D)
        self.种子 = int(种子)
        keys = sorted(set(self.任务["细节键列"] + list(self.任务["关键键列"]) + self.任务["干扰键列"]))
        self.槽数 = int(槽数 or max(1, len(keys)))
        self.键到槽 = {k: i for i, k in enumerate(keys)}
        self.宽态 = np.zeros((self.槽数, self.D), dtype=np.float64)
        self.宽元 = [None] * self.槽数
        self.码 = 码本(keys, D=self.D, 种子=self.种子)

    def 写(self, 事件):
        if not 事件.get("任务相关", True):
            return
        k = str(事件["键"])
        if k not in self.键到槽:
            return
        i = self.键到槽[k]
        old = self.宽元[i]
        rank = (int(事件["权限"]), int(事件["步"]), int(事件.get("序号", 0)))
        if old is not None and rank <= old["rank"]:
            return
        self.宽态[i] = self.码.绑(k, 事件["值"])
        self.宽元[i] = dict(rank=rank, 值=int(事件["值"]), 键=k)

    def 步进(self, 步):
        return

    def 读(self, 键):
        k = str(键)
        if k not in self.键到槽:
            return None
        i = self.键到槽[k]
        e = self.宽元[i]
        if e is None:
            return None
        # 从宽态精确回读：解绑 + 码本内积（单槽无干扰 ⇒ 精确）
        u = self.码.取键码(k) * self.宽态[i]
        s = self.码.值码 @ u
        return int(np.argmax(s))

    def 指纹(self):
        return "宽态%dx%d" % (self.宽态.shape[0], self.宽态.shape[1])


class S区室(B区室):
    """S：B + sham（随机锚点 / 打乱账本 / 惰性回读）；零假设对照，不得优于 A。"""

    名 = "S"

    def __init__(self, 任务, D=D默认, 种子=0, **kw):
        super().__init__(任务, D=D, 种子=种子, **kw)
        self.乱 = np.random.default_rng(int(种子) + S种子偏移)
        self.缓存 = {}

    def 写(self, 事件):
        if not 事件.get("任务相关", True) or 事件["类型"] == "干扰":
            return
        v = int(事件["值"])
        if float(self.乱.random()) < float(随机锚点率):
            v = int(self.乱.integers(0, _值域(事件["键"])))
        self.账本.append(dict(键=str(事件["键"]), 值=int(v),
                             权限=int(事件["权限"]), 步=int(事件["步"]),
                             序号=int(事件.get("序号", 0)), 类型=str(事件["类型"]),
                             关注度=float(事件.get("关注度", 1.0))))
        if len(self.账本) > self.容量:
            self.账本 = self.账本[-self.容量:]

    def 读(self, 键):
        k = str(键)
        if float(self.乱.random()) < float(缺失回读率):
            return None
        if float(self.乱.random()) < float(惰性回读率) and k in self.缓存:
            return self.缓存[k]
        if float(self.乱.random()) < float(打乱账本率) and self.账本:
            e = self.账本[int(self.乱.integers(0, len(self.账本)))]
            v = int(e["值"])
        else:
            c = self._候选(k)
            if not c:
                return None
            e = max(c, key=lambda x: (x["权限"], x["步"], x["序号"]))
            v = int(e["值"])
        self.缓存[k] = v
        return v


class R随机区室:
    """仪器自证/零假设用：对任何查询返回随机值（不是候选架构）。"""

    名 = "R"

    def __init__(self, 任务, D=D默认, 种子=0, **kw):
        self.任务 = 任务
        self.乱 = np.random.default_rng(int(种子) + 123457)
        self.缓存 = {}

    def 写(self, 事件):
        return

    def 步进(self, 步):
        return

    def 读(self, 键):
        return int(self.乱.integers(0, _值域(键)))

    def 指纹(self):
        return "随机"


def 建(臂, 任务, 种子=0, D=D默认, **kw):
    if 臂 == "A":
        return A区室(任务, D=D, 种子=种子, **kw)
    if 臂 == "B":
        return B区室(任务, D=D, 种子=种子, **kw)
    if 臂 == "C":
        return C区室(任务, D=D, 种子=种子, **kw)
    if 臂 == "S":
        return S区室(任务, D=D, 种子=种子, **kw)
    if 臂 == "R":
        return R随机区室(任务, D=D, 种子=种子, **kw)
    raise KeyError("未知臂：%r" % (臂,))


if __name__ == "__main__":
    # 冒烟：只跑 dev seed 100（不是判据/留出集）
    任务 = None
    try:
        envpy = os.path.join(HERE, "环境_长程合成任务.py")
        E = _载(envpy, "环境_冒烟")
        任务 = E.生成(100, "P0")
    except Exception as ex:                                   # noqa: BLE001
        print("环境未就绪：", repr(ex))
        raise SystemExit(1)
    行 = {}
    for 臂 in ("A", "B", "C", "S", "R"):
        体 = 建(臂, 任务, 种子=100)
        记 = dict(目标报告=[], 槽位报告={k: [] for k in 任务["关键键列"]}, 查询回答={})
        按步 = E.事件按步(任务)
        查步 = E.查询按步(任务)
        for t in range(任务["T"]):
            for e in 按步.get(t, ()):
                体.写(e)
            体.步进(t)
            记["目标报告"].append(体.读(E.目标键))
            for k in 任务["关键键列"]:
                记["槽位报告"][k].append(体.读(k))
            for q in 查步.get(t, ()):
                记["查询回答"][q["编号"]] = 体.读(q["键"])
        行[臂] = E.评分_P0(任务, 记)
    print(json.dumps(行, ensure_ascii=False, indent=1, default=str))
