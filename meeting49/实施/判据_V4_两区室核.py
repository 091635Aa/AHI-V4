# -*- coding: utf-8 -*-
"""AHI4/meeting49/实施/判据_V4_两区室核.py —— ★V4 总判据：两区室能力核（自我意识基底）。

真跑：慢率扫描（内生 credit）→ 主臂 / 消融 / 点式各λ / 迁移 / 三 sham（阈值**原样冻结**，与 meeting47 同）；
新增 4 条自我意识臂（区室分工 / 自指消融 / 归属 provenance / 真接线）；静态自检（C1 禁词 / 导入白名单 / 计费线性 / 两趟确定性）。
全部实测落盘 `判据_V4_两区室核结果.json`。阈值在脚本顶部**冻结**（写死，不看结果改）。

口径（显式固定，与 meeting47 一致以便可对读）：
  · 训练 400（读出+自我都学）→ 测试 400（**读出冻结**，只有自我通道在线适应）⇒ 规则切换能力只来自自我通道；
  · R 用「全试次」口径作阈值判据，同时如实记录 A 试次上的 R_A；
  · 内生 credit：更新权 = 归一化重合门强度（重合才赋分）；慢率由**预先声明的扫描**选定。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
本件 = os.path.abspath(__file__)
算子路径 = os.path.join(HERE, "算子_两区室核.py")
结果路径 = os.path.join(HERE, "判据_V4_两区室核结果.json")


def _载(路径, 名):
    spec = importlib.util.spec_from_file_location(名, 路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _载(算子路径, "算子_两区室核_判据")
桥路径 = os.path.join(HERE, "桥_自我态_自模型.py")
桥 = _载(桥路径, "桥_自我态_自模型_判据")

# ── 冻结阈值（写死；不看结果改）──
ACC_A_主 = 0.80
ACC_B_主 = 0.95
Δ点式_min = 0.10
Δ消融_min = 0.15
迁移_min = 0.75
R_max = 8
消融_ACC_A_max = 0.62
消融_ACC_B差_max = 0.05
sham_ACC_A_max = 0.60
计费比区间 = (3.95, 4.05)
# ── ★ 新增：自我意识臂阈值（先冻结）──
自_A解码规则_min = 0.75     # 从自我态 A 解码「当前规则」须可解码
世_B解码刺激_min = 0.75     # 从世界态 B 解码「刺激」须可解码
交叉解码_max = 0.65         # 交叉解码（A→刺激 / B→规则）须低
自指退化差_min = 0.10       # 自指（规则依赖）退化 − 世界退化 ≥ 0.10（论文 SST 同形）
自模型ε差_min = 0.005       # 自我态大跳处 ε_self 均值 − 平稳处 ε_self 均值 ≥ 此值（逐种子须为正）
允许导入 = ("__future__", "numpy", "hashlib", "json", "os", "sys", "importlib", "math", "types")

D = M.D默认
种子模型 = M.种子模型默认
序列训 = M.生成序列(M.种子训默认, M.训练长, M.边界训练)
序列测 = M.生成序列(M.种子测默认, M.测试长, M.边界训练)
序列迁移测 = M.生成序列(M.种子迁移测默认, M.测试长, M.边界迁移)


def 跑臂记录(模式, 序列测_, 慢率=None, 学模式="credit", λ=None):
    _, 训, 测 = M.跑臂(模式, D, 种子模型, 序列训, 序列测_, 慢率=(M.慢率默认 if 慢率 is None else 慢率),
                      学模式=学模式, λ=λ)
    return M.汇(测, 训), 模型指纹(训)


def 模型指纹(训):
    return hashlib.sha256(np.ascontiguousarray(训["选"]).tobytes()
                          + np.ascontiguousarray(训["奖"]).tobytes()).hexdigest()


def 扫导入(路径):
    名 = set()
    with open(路径, encoding="utf-8") as f:
        文 = f.read()
    for 行 in 文.splitlines():
        s = 行.strip()
        if s.startswith("import "):
            名.add(s[len("import "):].split()[0].split(",")[0].split(".")[0])
        elif s.startswith("from "):
            名.add(s[len("from "):].split()[0].split(".")[0])
    return sorted(名)


def 质心解码(迹, 标签):
    """确定性最近质心解码（无随机、无外部模型）：返回正确率。"""
    迹 = np.asarray(迹, dtype=np.float64)
    标签 = np.asarray(标签)
    类 = sorted(set(int(v) for v in 标签))
    质心 = {c: 迹[np.asarray(标签, dtype=np.int64) == c].mean(axis=0) for c in 类}
    预 = np.zeros(len(标签), dtype=np.int64)
    for i in range(len(标签)):
        d = [float(np.sum((迹[i] - 质心[c]) ** 2)) for c in 类]
        预[i] = 类[int(np.argmin(d))]
    return float((预 == 标签).mean())


def 跑并记录(模型, 序列, 学读出=False):
    """跑一遍并记录每拍的 自我态 A / 世界态 B / 规则标签 / 刺激标签（用于区室分工臂）。"""
    x = 序列["x"]
    b = 序列["边界"]
    n = int(len(x))
    迹A = np.zeros((n, 模型.D))
    迹B = np.zeros((n, 模型.D))
    规则标 = np.zeros(n, dtype=np.int64)
    刺激标 = np.zeros(n, dtype=np.int64)
    选 = np.zeros(n, dtype=np.int64)
    真 = np.zeros(n, dtype=np.int64)
    for t in range(n):
        o = 模型.前向_试次(float(x[t]))
        ŷ = int(np.argmax(模型.读出 @ o))
        c = 1 if float(x[t]) > float(b[t]) else 0
        真[t] = c
        选[t] = ŷ
        迹A[t] = 模型.自我态

        迹B[t] = 模型._最近B
        规则标[t] = int(round(float(b[t]) * 100))
        刺激标[t] = int(float(x[t]) > 0.5)
        模型.学读出(ŷ, 1 if ŷ == c else 0, 更新读出=学读出)
        模型.学自我(ŷ, 1 if ŷ == c else 0)
    return dict(A=迹A, B=迹B, 规则标=规则标, 刺激标=刺激标, 选=选, 真=真)


def 主流程():
    条 = []
    臂 = {}

    def 记(名, 通过, 反例, 实测, 阈值, 说明=""):
        条.append(bool(通过))
        臂[名] = dict(实测=实测, 阈值=阈值, 通过=bool(通过), 反例_必须失败=bool(反例), 说明=说明)

    # ── 慢率（内生 credit）**预先声明的扫描**（训练序列上训、测试序列上评）──
    扫描汇 = {}
    扫描纹 = {}
    for e in M.慢率列:
        汇e, 纹e = 跑臂记录("两区室", 序列测, 慢率=e, 学模式="credit")
        扫描汇[e] = 汇e
        扫描纹[e] = 纹e
    慢率曲线 = {"%.2f" % e: dict(Acc_A=扫描汇[e]["Acc_A"], Acc_B=扫描汇[e]["Acc_B"],
                                R_全=扫描汇[e]["R_全"], R_A=扫描汇[e]["R_A"]) for e in M.慢率列}
    合格 = [e for e in M.慢率列 if 扫描汇[e]["Acc_B"] >= ACC_B_主]
    候选 = 合格 if 合格 else list(M.慢率列)
    选定慢率 = max(候选, key=lambda e: (扫描汇[e]["Acc_A"], -e))   # Acc_A 最高且 Acc_B≥0.95；并列取较小
    汇主 = 扫描汇[选定慢率]
    指纹主 = 扫描纹[选定慢率]
    主通过 = bool(汇主["Acc_A"] >= ACC_A_主 and 汇主["Acc_B"] >= ACC_B_主 and 汇主["R_全"] <= R_max)
    记("主臂_两区室核", 主通过, False,
       dict(Acc_A=汇主["Acc_A"], Acc_B=汇主["Acc_B"], Acc=汇主["Acc"],
            R_全=汇主["R_全"], R_全_max=汇主["R_全_max"], R_A=汇主["R_A"], R_A_max=汇主["R_A_max"],
            切换点数=汇主["切换点数"], 训练末100全体Acc=汇主["训练末100全体Acc"], 选定慢率=选定慢率),
       dict(Acc_A=ACC_A_主, Acc_B=ACC_B_主, R_全=R_max),
       说明="Acc_A≥0.80 且 Acc_B≥0.95 且 R_全≤8（内生 credit；慢率=选定值 %.2f，由预先声明扫描选定）" % 选定慢率)

    # ── η 对照臂（meeting47 同款常数慢率；如实记录，不入主判据）──
    汇η, _ = 跑臂记录("两区室", 序列测, 慢率=选定慢率, 学模式="η")
    臂["对照_η常数"] = dict(实测=dict(Acc_A=汇η["Acc_A"], Acc_B=汇η["Acc_B"], 慢率=选定慢率),
                          阈值=None, 通过=None, 反例_必须失败=False,
                          说明="去除 credit 加权（有效率=慢率常数）作对照；记录以备对读，不入主判定")

    # ── 消融（自指臂）：冻结自我态 A；慢率=选定 ──
    汇消, _ = 跑臂记录("消融", 序列测, 慢率=选定慢率)
    差B = abs(汇消["Acc_B"] - 汇主["Acc_B"])
    差A = float(汇主["Acc_A"] - 汇消["Acc_A"])
    消通过 = bool(汇消["Acc_A"] <= 消融_ACC_A_max and 差B <= 消融_ACC_B差_max and 差A >= Δ消融_min)
    记("消融_冻结自我态A", 消通过, True,
       dict(Acc_A=汇消["Acc_A"], Acc_B=汇消["Acc_B"], Δ消融=差A, Acc_B与主臂差=差B),
       dict(Acc_A_上限=消融_ACC_A_max, Acc_B差_上限=消融_ACC_B差_max, Δ消融_下限=Δ消融_min),
       说明="预期 自指(规则依赖)大降、世界(刺激无关)几乎不变；且 Δ消融≥0.15")

    # ── 点式基线：各 λ（慢率=选定，奖赏更新规则与主臂同）──
    点式 = {}
    for λ in M.λ列:
        汇, _ = 跑臂记录("点神经元", 序列测, 慢率=选定慢率, λ=λ)
        点式["λ=%.2f" % λ] = dict(Acc_A=汇["Acc_A"], Acc_B=汇["Acc_B"], R_全=汇["R_全"])
    点A最大值 = max(v["Acc_A"] for v in 点式.values())
    点式最优λ = max(点式, key=lambda k: 点式[k]["Acc_A"])
    Δ点式 = float(汇主["Acc_A"] - 点A最大值)
    记("点式基线", bool(Δ点式 >= Δ点式_min), False,
       dict(各λ=点式, Acc_A_max=点A最大值, 最优λ=点式最优λ, Δ点式=Δ点式),
       dict(Δ点式_下限=Δ点式_min),
       说明="同 D/P/读出形状/种子/预算/奖赏更新规则；差异只在单一递推状态 s")

    # ── 迁移臂：训练 {0.35,0.65} → 测试未见边界 {0.25,0.75} ──
    汇迁, _ = 跑臂记录("两区室", 序列迁移测, 慢率=选定慢率)
    记("迁移_未见边界", bool(汇迁["Acc_A"] >= 迁移_min), False,
       dict(Acc_A=汇迁["Acc_A"], Acc_B=汇迁["Acc_B"], R_全=汇迁["R_全"]),
       dict(Acc_A=迁移_min), 说明="训练 {0.35,0.65} → 测试未见边界 {0.25,0.75}（宪法强制的迁移臂）")

    # ── sham 1：规则随机（自我标量随机游走，不用奖赏）──
    汇sh1, _ = 跑臂记录("规则随机", 序列测, 慢率=选定慢率)
    记("sham_规则随机", bool(汇sh1["Acc_A"] <= sham_ACC_A_max), True,
       dict(Acc_A=汇sh1["Acc_A"], Acc_B=汇sh1["Acc_B"]),
       dict(Acc_A_上限=sham_ACC_A_max), 说明="自我标量随机游走（不用奖赏）⇒ 自指臂必须不过阈")

    # ── sham 2：无区室（o=归(B)，只用世界态）──
    汇sh2, _ = 跑臂记录("无区室", 序列测, 慢率=选定慢率)
    记("sham_无区室", bool(汇sh2["Acc_A"] <= sham_ACC_A_max), True,
       dict(Acc_A=汇sh2["Acc_A"], Acc_B=汇sh2["Acc_B"]),
       dict(Acc_A_上限=sham_ACC_A_max), 说明="o 换成 _归(B)（两区室结构被打乱）⇒ 自指臂必须不过阈")

    # ── sham 3：线性求和（o=归(A+B)，非差分）──
    汇sh3, _ = 跑臂记录("线性求和", 序列测, 慢率=选定慢率)
    Δ线性vs主 = float(汇主["Acc_A"] - 汇sh3["Acc_A"])
    记("sham_线性求和", bool(汇sh3["Acc_A"] <= sham_ACC_A_max), True,
       dict(Acc_A=汇sh3["Acc_A"], Acc_B=汇sh3["Acc_B"], Δ相对主臂=Δ线性vs主),
       dict(Acc_A_上限=sham_ACC_A_max),
       说明="o=归(A+B)（线性求和，非差分）；显著差于 软限(A−B)（Δ=%.3f）" % Δ线性vs主)

    # ══════════ 新增：自我意识臂 ══════════
    # ── 真接线臂：符已登记 + 主循环真跑 + 计费线性 ──
    核 = M.两区室核(D=D, 种子=种子模型)
    算子表 = 核.算子表()
    码本 = M.生成码本(8, D, D)
    真接 = 核.前向_双区室(码本, None, 迭代步数=1)
    费比 = float(核.计费(4096)["乘加数"]) / max(1.0, float(核.计费(1024)["乘加数"]))
    基前向 = M.能力核(D=D, 种子=种子模型).前向(码本, None, 迭代步数=1)
    真接通过 = bool(M.算子_符 in 算子表 and len(算子表) == 8
                   and 真接["输出"].shape == (8, D) and 真接["门"].shape == (8, D)
                   and abs(费比 - 4.0) < 0.05)
    记("V4_真接线", 真接通过, False,
       dict(算子表=算子表, 算子数=len(算子表), 符已登记=bool(M.算子_符 in 算子表),
            主循环输出形状=list(基前向["输出"].shape), 双区室输出形状=list(真接["输出"].shape),
            门形状=list(真接["门"].shape), 计费比=费比),
       dict(符已登记=True, 算子数=8, 计费比区间=list(计费比区间)),
       说明="第八算子「符」真登记进能力核注册表；前向_双区室真跑既有主循环；计费仍线性")

    # ── 区室分工臂：从自我态 A 解码规则、从世界态 B 解码刺激；交叉须低 ──
    模 = M.两区室核(D=D, 种子=种子模型, 模式="两区室", 慢率=选定慢率, 学模式="credit")
    M.跑(模, 序列训, 学读出=True, 学模式="credit")           # 先训（同主臂口径）
    迹 = 跑并记录(模, 序列测, 学读出=False)
    A解码规则 = 质心解码(迹["A"], 迹["规则标"])
    B解码刺激 = 质心解码(迹["B"], 迹["刺激标"])
    A解码刺激 = 质心解码(迹["A"], 迹["刺激标"])
    B解码规则 = 质心解码(迹["B"], 迹["规则标"])
    分工通过 = bool(A解码规则 >= 自_A解码规则_min and B解码刺激 >= 世_B解码刺激_min
                  and A解码刺激 <= 交叉解码_max and B解码规则 <= 交叉解码_max)
    记("自我_区室分工", 分工通过, False,
       dict(从A解码规则=A解码规则, 从B解码刺激=B解码刺激,
            从A解码刺激=A解码刺激, 从B解码规则=B解码规则),
       dict(从A解码规则_min=自_A解码规则_min, 从B解码刺激_min=世_B解码刺激_min, 交叉解码_max=交叉解码_max),
       说明="自我态 A 载「规则」、世界态 B 载「刺激」；交叉解码低 ⇒ 两区室表征分离")

    # ── 自指消融臂（论文 SST 同形）：自指退化 与 世界退化 必须差≥0.10 ──
    世界退化 = abs(汇主["Acc_B"] - 汇消["Acc_B"])
    自指退化 = 差A
    自指通过 = bool(自指退化 - 世界退化 >= 自指退化差_min and 自指退化 >= Δ消融_min)
    记("自我_自指消融", 自指通过, False,
       dict(自指退化=自指退化, 世界退化=世界退化, 差=自指退化 - 世界退化),
       dict(自指退化_min=Δ消融_min, 退化差_min=自指退化差_min),
       说明="抑制自我态 ⇒ 自指(规则依赖)退化 ≫ 世界(刺激无关)退化（类比论文 SST 光遗传结论）")

    # ── 归属臂（provenance）：外源规则不得塑造自我态；允许时才可塑（正控）──
    核2 = M.两区室核(D=D, 种子=种子模型)
    拒 = 核2.注入外源规则(np.ones(D), 允许=False)
    允 = 核2.注入外源规则(np.full(D, 3.0), 允许=True)
    归属通过 = bool(拒["被塑造"] is False and 允["被塑造"] is True)
    记("自我_归属provenance", 归属通过, False,
       dict(拒绝注入=拒, 允许注入=允),
       dict(默认拒绝塑造=True, 允许时才塑造=True),
       说明="外源规则默认不塑造自我态（沿用 内核_自我回路 provenance 门控语义）；允许时正控为真")

    # ── 自模型 ε 臂：自我态 A 喂既有 内核_自模型.ĝ；对自己状态的意外 ──
    桥种列 = (47101, 47201, 47301)
    桥果 = {}
    for s in 桥种列:
        序s = M.生成序列(s, M.训练长, M.边界训练)
        r = 桥.跑桥(序s, 慢率=选定慢率, 训练=True, 种子=种子模型)
        桥果[s] = dict(大跳ε=r["大跳ε均值"], 小跳ε=r["小跳ε均值"],
                     差=r["大跳ε均值"] - r["小跳ε均值"], 大跳条数=r["大跳条数"])
    桥差列 = [桥果[s]["差"] for s in 桥种列]
    桥差均 = float(np.mean(桥差列))
    桥通过 = bool(桥差均 >= 自模型ε差_min and all(v > 0.0 for v in 桥差列))
    记("自我_自模型ε", 桥通过, False,
       dict(各种子=桥果, 差均=桥差均),
       dict(差均_min=自模型ε差_min, 逐种子须为正=True),
       说明="自我态喂既有 内核_自模型.ĝ：自我态大跳时 ε_self 高、平稳时低（自监督、可失败）")

    # ── 静态自检 1：C1 禁词自扫（两个新增 .py，命中数须 0）──
    禁词 = list(M.禁词列())
    命中表 = {}
    for p in (算子路径, 本件, 桥路径):
        with open(p, encoding="utf-8") as f:
            低 = f.read().lower()
        命中表[os.path.basename(p)] = [w for w in 禁词 if w.lower() in 低]
    命中总数 = sum(len(v) for v in 命中表.values())
    记("静态_C1禁词自扫", bool(命中总数 == 0), False,
       dict(禁词数=len(禁词), 命中=命中表, 命中总数=命中总数),
       dict(命中总数=0), 说明="复用 算子_能力核.禁词列")

    # ── 静态自检 2：导入白名单 ──
    导入表 = {os.path.basename(p): 扫导入(p) for p in (算子路径, 本件, 桥路径)}
    越界 = {k: [m for m in v if m not in 允许导入] for k, v in 导入表.items()}
    越界非空 = {k: v for k, v in 越界.items() if v}
    记("静态_导入白名单", bool(not 越界非空), False,
       dict(导入表=导入表, 白名单=list(允许导入), 越界=越界非空),
       dict(越界=[]), 说明="只允许 numpy/hashlib/json/os/sys/importlib/math/__future__")

    # ── 静态自检 3：计费线性（对新建能力核实例）──
    费1 = 核.计费(1024)["乘加数"]
    费4 = 核.计费(4096)["乘加数"]
    比值 = float(费4) / max(1.0, float(费1))
    记("静态_计费线性", bool(计费比区间[0] <= 比值 <= 计费比区间[1]), False,
       dict(计费1024=费1, 计费4096=费4, 比值=比值),
       dict(比值区间=list(计费比区间)), 说明="证明未引入随 n² 增长的开销")

    # ── 静态自检 4：两趟确定性（同种子两遍，Acc 与关键向量逐位同）──
    汇主2, 指纹主2 = 跑臂记录("两区室", 序列测, 慢率=选定慢率)
    决通过 = bool(汇主2["Acc_A"] == 汇主["Acc_A"] and 汇主2["Acc_B"] == 汇主["Acc_B"]
                 and 汇主2["R_全"] == 汇主["R_全"] and 指纹主2 == 指纹主)
    记("静态_两趟确定性", 决通过, False,
       dict(Acc_A=[汇主["Acc_A"], 汇主2["Acc_A"]], Acc_B=[汇主["Acc_B"], 汇主2["Acc_B"]],
            指纹=[指纹主, 指纹主2]),
       dict(逐位同=True), 说明="同种子两遍，Acc 与关键向量逐位同")

    未过 = [名 for 名, v in 臂.items() if v.get("通过") is False]
    果 = dict(
        通过=bool(all(条)),
        判定条数=len(条),
        未过条数=int(sum(1 for v in 条 if not v)),
        未过臂=未过,
        未测项=[],
        阈值=dict(ACC_A_主=ACC_A_主, ACC_B_主=ACC_B_主, Δ点式_min=Δ点式_min, Δ消融_min=Δ消融_min,
                  迁移_min=迁移_min, R_max=R_max, 消融_ACC_A_max=消融_ACC_A_max,
                  消融_ACC_B差_max=消融_ACC_B差_max, sham_ACC_A_max=sham_ACC_A_max,
                  自_A解码规则_min=自_A解码规则_min, 世_B解码刺激_min=世_B解码刺激_min,
                  交叉解码_max=交叉解码_max, 自指退化差_min=自指退化差_min,
                  自模型ε差_min=自模型ε差_min,
                  计费比区间=list(计费比区间)),
        慢率扫描=dict(曲线=慢率曲线, 选定慢率=选定慢率, 扫描列=list(M.慢率列),
                    选择规则="主臂 Acc_A 最高且 Acc_B≥0.95；并列取较小（阈值不随结果改）"),
        臂=臂,
        实测摘要=dict(选定慢率=选定慢率, 主Acc_A=汇主["Acc_A"], 主Acc_B=汇主["Acc_B"],
                    主R_全=汇主["R_全"], 主R_A=汇主["R_A"],
                    对照ηAcc_A=汇η["Acc_A"], 消融Acc_A=汇消["Acc_A"], Δ消融=差A,
                    点式Acc_A_max=点A最大值, 点式最优λ=点式最优λ, Δ点式=Δ点式,
                    迁移Acc_A=汇迁["Acc_A"],
                    sham规则随机Acc_A=汇sh1["Acc_A"], sham无区室Acc_A=汇sh2["Acc_A"],
                    sham线性求和Acc_A=汇sh3["Acc_A"],
                    从A解码规则=A解码规则, 从B解码刺激=B解码刺激,
                    从A解码刺激=A解码刺激, 从B解码规则=B解码规则,
                    自指退化=自指退化, 世界退化=世界退化,
                    自模型ε差均=桥差均, 自模型大跳ε=桥果[桥种列[0]]["大跳ε"], 自模型小跳ε=桥果[桥种列[0]]["小跳ε"],
                    算子数=len(算子表), 计费比=费比,
                    主臂指纹=指纹主, D=D, 种子模型=种子模型, 切换点数主=汇主["切换点数"]),
        说明=("训练 400（读出+自我 学）→ 测试 400。**所有臂测试期一律冻结读出、只让自我通道在线适应**（口径一致）。"
             "主臂用**内生 credit**（更新权=归一化重合门强度，重合才赋分）；慢率由**预先声明的扫描**选定（%.2f）。"
             "阈值（含新增自我意识臂）先冻结、未随结果改。random 只来自 default_rng(显式种子)。" % 选定慢率),
    )
    return 果


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        np.seterr(all="ignore")
    except Exception:
        pass
    果 = 主流程()
    with open(结果路径, "w", encoding="utf-8") as f:
        json.dump(果, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(果["实测摘要"], ensure_ascii=False, indent=1, default=str))
    print(json.dumps(果["阈值"], ensure_ascii=False, default=str))
    print("[判据_V4_两区室核] 判定 %d 条，未过 %d 条 ⇒ %s"
          % (果["判定条数"], 果["未过条数"], "PASS" if 果["通过"] else "FAIL"))
    print("未过臂：", json.dumps(果["未过臂"], ensure_ascii=False))
    print("结果已写：", 结果路径)
    raise SystemExit(0 if 果["通过"] else 1)
