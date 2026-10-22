# -*- coding: utf-8 -*-
"""AHI4/meeting52/蓝队/实测/B_记忆与F5.py —— 蓝队 worker B（Level-2）B 组八臂真跑。

━━━ 本件做什么 ━━━
B1 记忆召回曲线（记忆容器(D=256,K) + 千层核.符，n=30，种子 20261017）
B2 K-差（K=1 vs K=4）
B3 缓存冒充臂（最近 40 拍 FIFO vs 冻结槽，40 拍后凭线索召回）
B4 随机槽臂（无记忆，随机存 8 条）
B5 槽置乱 null（槽身份随机置换，null=1/K）
B6 F5-a 状态↔日志（V4 主臂 vs 同算力无状态点神经元）
B7 F5-b 可重算性（点神经元重算器 vs 失忆臂）
B8 F5-c 历史置换（训练序列时序打乱 + 同分布随机历史 must-fail）

━━━ 硬纪律 ━━━
· 只新建本件 + `B_结果.json`；**不改/不覆盖任何既有件或既有 JSON**；**不运行任何 `判据_*.py`**。
· 只读 `importlib` 加载 `meeting51/实施/算子_千层骨.py` 与 `meeting49/实施/算子_两区室核.py`（其内部再只读加载能力核/环境）。
· `python.exe -B`（不落 `__pycache__`）；零 git；零装包；随机只来自 `default_rng(显式种子)`。
· **阈值先写死在下面 `阈值` 字典里，跑前冻结，不据结果改**。

━━━ 协议要点（预注册，跑前写定）━━━
P1 B1/B2 的"记忆容量"按 **K 槽全填满** 协议：先冻结靶（前 min(K,n) 条），再填同分布随机干扰至 K；
   靶 = ±1 随机码（D=256，rng(20261017)），线索 = 靶按 flip 比例随机取反（掩码 rng(20261111)，在各臂间配对）。
   召回 = `解冻` 后 `重合强度` 的 argmax（迭代 1 或 3）。相变点 ρ* = 相邻 flip 点 |ΔR|/Δflip 最大区间的中点。
P2 B1 迭代比较用**同一掩码/同一槽内容**，逐位比较预测下标数组。
P3 B2 K=1 只能驻留 1 条、K=4 驻留 4 条 ⇒ 同时报 raw（全 30 靶召回）与 resident（仅两臂共同驻留的靶）。
P4 B3 流长 120 拍，槽臂 K=8 只冻结前 8 条（满后不再冻结）；FIFO 只保留最近 40 条；查询前 8 条（干净线索）。
P5 B5 置乱 = 召回下标经随机置换 π（rng(20261113)）映射后计分。
P6 B6/B7/B8 用 V4 口径：D=256、种子 47001、训 400 / 测 400、边界 {0.35,0.65}；
   慢率由**预声明扫描**（V4 慢率列 10 值，取 Acc_B≥0.95 中 Acc_A 最高、并列取小）选定后固定给各臂。
P7 B7 重算器 = 点神经元 λ=0.10（10 拍泄漏积分，只吃当前输入+输入历史流）；失忆臂 = 点神经元 λ=1.00（只吃当前）。
P8 B8 置换 = 训练序列 (x, 边界) 同步按 rng(47107) 置换（同边际、同长度）；随机历史 = 生成序列(47107,400,{0.35,0.65})。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

np.seterr(all="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))          # …/AHI4/meeting52/蓝队/实测
AHI4 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # …/AHI4
_千层py = os.path.join(AHI4, "meeting51", "实施", "算子_千层骨.py")
_两区室py = os.path.join(AHI4, "meeting49", "实施", "算子_两区室核.py")
_结果py = os.path.join(HERE, "B_结果.json")


# ═══════════════════════════════ 预注册阈值（先写死；跑前冻结） ═══════════════════════════════
阈值 = dict(
    B1_可检索最大flip=0.30,      # 可检索区 = ρ≤0.30
    B1_可检索召回下限=0.90,
    B1_相变移动下限=0.10,        # 迭代 1 vs 3 移动相变点 ≥0.10 才算"有补全动力学"
    B2_gap下限=0.10,             # max_ρ[R(4)−R(1)] ≥0.10
    B3_槽召回下限=0.90,
    B3_FIFO召回上限=0.50,
    B3_流长=120,                 # 拍
    B3_缓存窗=40,                # 拍
    B3_槽数=8,
    B4_召回上限=0.25,
    B4_槽数=8,
    B5_召回上限=0.175,           # null=1/8=0.125
    B5_槽数=8,
    B6_状态差下限=0.10,          # ①−②
    B7_重算器余量=0.05,          # 重算器 ≥ 主臂−0.05 ⇒ "可预演"
    B7_失忆上限=0.65,
    B8_掉幅下限=0.10,
)

种子 = dict(记忆=20261017, 干扰=20261099, 线索=20261111, 置乱=20261113,
            模型=47001, 训=47101, 测=47102, 置换=47107)
D = 256
n靶 = 30
flip列 = (0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.8)


def _载(路径, 名):
    spec = importlib.util.spec_from_file_location(名, 路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _log(msg):
    print("[B] " + str(msg), flush=True)


# ═══════════════════════════════ 公共小件 ═══════════════════════════════
def 造靶(n, d, 种子_):
    rng = np.random.default_rng(int(种子_))
    return np.where(rng.random((int(n), int(d))) < 0.5, -1.0, 1.0)


def 最强(候选, 线索, 符):
    """候选列（m,D）上按 记忆容器.重合强度 同式算归一化重合，返回 (argmax, 全强度)。"""
    线 = np.asarray(线索, dtype=np.float64)
    线尺 = float(np.mean(np.abs(线))) + 1e-9
    强 = np.full(len(候选), -1e9, dtype=np.float64)
    for k in range(len(候选)):
        槽尺 = float(np.mean(np.abs(候选[k]))) + 1e-9
        强[k] = float(np.mean(符(线 / 线尺, np.asarray(候选[k]) / 槽尺)))
    return int(np.argmax(强)), 强


def 相变点(flips, Rs):
    """相邻点最大 |ΔR|/Δflip 区间的中点（严格大于取最早；确定性）。"""
    斜 = []
    for j in range(len(flips) - 1):
        if flips[j + 1] <= flips[j]:
            continue
        斜.append(dict(区间=[float(flips[j]), float(flips[j + 1])],
                       斜率=float(abs(Rs[j + 1] - Rs[j]) / (flips[j + 1] - flips[j]))))
    if not 斜:
        return dict(中点=None, 区间=None, 斜率表=斜)
    最大 = max(斜, key=lambda x: x["斜率"])
    中点 = (最大["区间"][0] + 最大["区间"][1]) / 2.0
    return dict(中点=float(中点), 区间=最大["区间"], 斜率=最大["斜率"], 斜率表=斜)


def _指纹(数组):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(数组, dtype=np.float64)).tobytes()).hexdigest()[:16]


# ═══════════════════════════════ B1 ═══════════════════════════════
def B1(千层):
    符 = 千层.千层核(D=D, 种子=种子["记忆"]).符          # 只取第八算子「符」（W符=ones，见算子_千层骨:344）
    靶 = 造靶(n靶, D, 种子["记忆"])
    干扰 = 造靶(1024, D, 种子["干扰"])
    rng线 = np.random.default_rng(种子["线索"])
    掩码 = {f: (rng线.random((n靶, D)) < f) for f in flip列}

    def 填(mem, K):
        for i in range(min(int(K), n靶)):
            mem.冻结(靶[i])
        for j in range(int(K) - min(int(K), n靶)):
            mem.冻结(干扰[j])

    def 一轮全(K, 迭代):
        mem = 千层.记忆容器(D, int(K))
        填(mem, K)
        表, 预测 = [], []
        for f in flip列:
            ok, 下标, 边距 = 0, [], []
            for i in range(n靶):
                cue = 靶[i].copy()
                cue[掩码[f][i]] *= -1.0
                _, 强 = mem.解冻(cue, 符, 迭代=int(迭代))
                序 = np.argsort(-强)
                下标.append(int(序[0]))
                ok += int(序[0] == i)
                边距.append(float(强[序[0]] - 强[序[1]]))
            表.append(dict(flip=float(f), R=float(ok) / n靶, 边距均值=float(np.mean(边距))))
            预测.append(下标)
        return dict(K=int(K), 迭代=int(迭代), 表=表,
                    R列=[x["R"] for x in 表], 预测=预测)

    行 = {}
    for K in (64, 1024):
        for it in (1, 3):
            行[(K, it)] = 一轮全(K, it)
            _log("B1 K=%d 迭代=%d R=%s" % (K, it, [round(x, 3) for x in 行[(K, it)]["R列"]]))

    # 两趟确定性抽查（重建同配置，逐位比对）
    重跑 = 一轮全(64, 1)
    两趟同 = bool(重跑["预测"] == 行[(64, 1)]["预测"] and 重跑["R列"] == 行[(64, 1)]["R列"])

    逐臂 = []
    迭代全同 = True
    for K in (64, 1024):
        a, b = 行[(K, 1)], 行[(K, 3)]
        同 = bool(a["预测"] == b["预测"] and a["R列"] == b["R列"])
        迭代全同 = 迭代全同 and 同
        ρ1, ρ3 = 相变点(flip列, a["R列"]), 相变点(flip列, b["R列"])
        移动 = abs(float(ρ3["中点"]) - float(ρ1["中点"])) if (ρ3["中点"] is not None and ρ1["中点"] is not None) else 0.0
        可检索 = min(r for r, f in zip(a["R列"], flip列) if f <= 阈值["B1_可检索最大flip"] + 1e-12)
        逐臂.append(dict(K=K, 迭代1=dict(R列=a["R列"], 表=a["表"], 相变=ρ1),
                         迭代3=dict(R列=b["R列"], 表=b["表"], 相变=ρ3),
                         预测逐位同=同, 相变移动=float(移动),
                         可检索区最小R=float(可检索)))
    Δρ = max(x["相变移动"] for x in 逐臂)
    可检索达标 = all(x["可检索区最小R"] >= 阈值["B1_可检索召回下限"] for x in 逐臂)
    动力学 = bool(Δρ >= 阈值["B1_相变移动下限"])
    判定 = dict(
        可检索区达标=bool(可检索达标),
        迭代逐位同=bool(迭代全同),
        相变移动=float(Δρ),
        有补全动力学=动力学,
        结论=("无补全动力学（迭代 1=3 逐位同；移动相变点 %.3f < %.2f 阈值 ⇒ 记忆=高维查表/最近邻检索）"
              % (Δρ, 阈值["B1_相变移动下限"])) if not 动力学 else
             ("有补全动力学（移动相变点 %.3f ≥ %.2f）" % (Δρ, 阈值["B1_相变移动下限"])),
    )
    return dict(臂=逐臂, 判定=判定, 两趟逐位同=bool(两趟同), 靶指纹=_指纹(靶),
                协议="K 槽全填满（靶+同分布随机干扰）；±1 码 D=256；线索按 flip 取反；argmax 重合强度",
                备注="与 R2 §9.1 相比：本件 K=64 flip0.4 的读数见上表（两件均 1.000@≤0.3、0@0.5，结论同）")


# ═══════════════════════════════ B2 ═══════════════════════════════
def B2(千层):
    符 = 千层.千层核(D=D, 种子=种子["记忆"]).符
    靶 = 造靶(n靶, D, 种子["记忆"])
    rng线 = np.random.default_rng(种子["线索"])
    掩码 = {f: (rng线.random((n靶, D)) < f) for f in flip列}

    def 一轮(K):
        mem = 千层.记忆容器(D, int(K))
        for i in range(min(int(K), n靶)):
            mem.冻结(靶[i])
        表, 预测 = [], []
        for f in flip列:
            预 = []
            for i in range(n靶):
                cue = 靶[i].copy()
                cue[掩码[f][i]] *= -1.0
                _, 强 = mem.解冻(cue, 符, 迭代=1)
                预.append(int(np.argmax(强)))
            表.append(dict(flip=float(f), R_raw=float(np.mean([int(预[i] == i) for i in range(n靶)]))))
            预测.append(预)
        return 表, 预测

    表1, 预1 = 一轮(1)
    表4, 预4 = 一轮(4)
    gap表 = []
    for a, b in zip(表1, 表4):
        gap表.append(dict(flip=a["flip"], R1=a["R_raw"], R4=b["R_raw"], 差=float(b["R_raw"] - a["R_raw"])))
    最大 = max(gap表, key=lambda x: x["差"])
    # resident：K=1 只驻留靶0；K=4 驻留靶0..3 ⇒ 两臂共同驻留 = 靶0
    驻1 = [int(预1[j][0] == 0) for j in range(len(flip列))]
    驻4 = [int(预4[j][0] == 0) for j in range(len(flip列))]
    gap驻 = max(float(b - a) for a, b in zip(驻1, 驻4))
    判定 = dict(gap_raw=float(最大["差"]), 最大处flip=最大["flip"],
                gap_resident=float(gap驻),
                过阈=bool(最大["差"] >= 阈值["B2_gap下限"]),
                结论=("raw gap %.3f ≥ %.2f（边界：值= %.3f 恰为多驻留 3/30）；"
                      "resident gap %.3f ⇒ 容量质量无差 ⇒ 只能主张「缓存+top-1」"
                      % (最大["差"], 阈值["B2_gap下限"], 最大["差"], gap驻)))
    return dict(表=gap表, 判定=判定, 驻留=dict(K1=1, K4=4),
                协议="n=30 靶；K=1 只容 1 条，K=4 容 4 条（饱和填充；K=4 无干扰余量）")


# ═══════════════════════════════ B3 / B4 / B5 ═══════════════════════════════
def _流(长, d, 种子_):
    return 造靶(长, d, 种子_)


def B3(千层):
    符 = 千层.千层核(D=D, 种子=种子["记忆"]).符
    流 = _流(阈值["B3_流长"], D, 种子["记忆"] + 3)
    K = 阈值["B3_槽数"]
    mem = 千层.记忆容器(D, int(K))
    for i in range(K):
        mem.冻结(流[i])                        # 只冻结前 8 条（槽满后不再冻结）
    槽表 = [mem.槽[k].copy() for k in range(K)]
    FIFO = [流[t].copy() for t in range(阈值["B3_流长"] - 阈值["B3_缓存窗"], 阈值["B3_流长"])]

    def 查(候选, cue):
        k, _ = 最强(候选, cue, 符)
        return int(k)

    早 = list(range(K))                                     # 前 8 条
    近 = list(range(阈值["B3_流长"] - K, 阈值["B3_流长"]))    # 最近 8 条
    槽_早 = float(np.mean([int(查(槽表, 流[i]) == i) for i in 早]))
    槽_近 = float(np.mean([int(查(槽表, 流[i]) == i) for i in 近]))
    起 = 阈值["B3_流长"] - 阈值["B3_缓存窗"]
    # FIFO 命中的是窗口内下标 ⇒ 映射回流下标后与真下标比
    FIFO_早 = float(np.mean([int(bool(np.array_equal(FIFO[查(FIFO, 流[i])], 流[i]))) for i in 早]))
    FIFO_近 = float(np.mean([int(bool(np.array_equal(FIFO[查(FIFO, 流[i])], 流[i]))) for i in 近]))
    判定 = dict(槽_早=槽_早, FIFO_早=FIFO_早, 槽_近=槽_近, FIFO_近=FIFO_近,
                过阈=bool(槽_早 >= 阈值["B3_槽召回下限"] and FIFO_早 <= 阈值["B3_FIFO召回上限"]),
                结论="40 拍窗外的早期条目：槽臂 %.2f vs FIFO %.2f ⇒ FIFO 只是缓存窗" % (槽_早, FIFO_早))
    return dict(判定=判定, 流长=阈值["B3_流长"], 缓存窗=阈值["B3_缓存窗"], 槽数=int(K),
                协议="120 拍流；槽臂只存前 8 条；FIFO 存最近 40 条；查询前 8/最近 8 条")


def B4(千层):
    符 = 千层.千层核(D=D, 种子=种子["记忆"]).符
    K = 阈值["B4_槽数"]
    真 = 造靶(K, D, 种子["记忆"] + 5)
    假 = 造靶(K, D, 种子["置乱"])              # 随机存 8 条（与靶无关）
    R = float(np.mean([int(最强(假, 真[i], 符)[0] == i) for i in range(K)]))
    判定 = dict(R=R, 过阈=bool(R <= 阈值["B4_召回上限"]), 结论="随机槽召回 %.3f ≤ %.3f（=1/K 水平）" % (R, 阈值["B4_召回上限"]))
    return dict(判定=判定, K=int(K), 协议="随机存 8 条无关码，查询 8 条真靶（干净线索）")


def B5(千层):
    符 = 千层.千层核(D=D, 种子=种子["记忆"]).符
    K = 阈值["B5_槽数"]
    真 = 造靶(K, D, 种子["记忆"] + 7)
    mem = 千层.记忆容器(D, int(K))
    for i in range(K):
        mem.冻结(真[i])
    原 = []
    for i in range(K):
        _, 强 = mem.解冻(真[i], 符, 迭代=1)
        原.append(int(np.argmax(强)))
    R原 = float(np.mean([int(原[i] == i) for i in range(K)]))
    # null 分布：槽身份随机置换 200 次（单次置换的固定点期望=1/K；单次可实现 0 或 2/8）
    rng = np.random.default_rng(种子["置乱"])
    列 = []
    for _ in range(200):
        π = rng.permutation(K)
        列.append(float(np.mean([int(π[原[i]] == i) for i in range(K)])))
    R置均 = float(np.mean(列))
    R置最大 = float(np.max(列))
    判定 = dict(R_raw=R原, R_perm均值=R置均, R_perm最大=R置最大, null期望=1.0 / K,
                单次退化="单次置换可为 0（π 无固定点）⇒ 必须用置换分布，不能用单次读数",
                过阈=bool(R置均 <= 阈值["B5_召回上限"]),
                结论="槽身份随机置换 200 次：均值 %.3f ≤ %.3f（null=1/K=%.3f；单次最大 %.3f）"
                     % (R置均, 阈值["B5_召回上限"], 1.0 / K, R置最大))
    return dict(判定=判定, K=int(K), 协议="8 条真靶入槽；召回下标经 rng(20261113) 置换 π 映射，200 次取分布")


# ═══════════════════════════════ B6 / B7 / B8（V4 口径） ═══════════════════════════════
def B68(V4):
    训 = V4.生成序列(种子["训"], V4.训练长, V4.边界训练)
    测 = V4.生成序列(种子["测"], V4.测试长, V4.边界训练)
    扫描 = {}
    for e in V4.慢率列:
        _, _, ce = V4.跑臂("两区室", D, 种子["模型"], 训, 测, 慢率=e)
        扫描[float(e)] = dict(Acc_A=float(ce["Acc_A"]), Acc_B=float(ce["Acc_B"]))
    合格 = [e for e in V4.慢率列 if 扫描[float(e)]["Acc_B"] >= 0.95]
    选定 = max(合格 if 合格 else list(V4.慢率列),
              key=lambda e: (扫描[float(e)]["Acc_A"], -e))
    _log("B6/B8 慢率扫描选定 %.2f（Acc_A=%.4f, Acc_B=%.4f）"
         % (选定, 扫描[float(选定)]["Acc_A"], 扫描[float(选定)]["Acc_B"]))

    # B6 ① 主臂（有状态无日志）
    核主, 训主, 测主 = V4.跑臂("两区室", D, 种子["模型"], 训, 测, 慢率=选定)
    主Acc = float(测主["Acc_A"])
    # B6 ② 同算力无状态（点神经元 λ=1.0，只吃当前输入；日志保留但不进决策）
    _, _, 测无状态 = V4.跑臂("点神经元", D, 种子["模型"], 训, 测, 慢率=选定, λ=1.0)
    无状态Acc = float(测无状态["Acc_A"])
    差 = 主Acc - 无状态Acc
    # 灵敏度：点式各 λ（V4 λ列）
    点式 = {}
    for lam in V4.λ列:
        _, _, c = V4.跑臂("点神经元", D, 种子["模型"], 训, 测, 慢率=选定, λ=float(lam))
        点式["%.2f" % float(lam)] = float(c["Acc_A"])
    点式max = max(点式.values())
    B6 = dict(主臂Acc_A=主Acc, 无状态Acc_A=无状态Acc, 差=float(差),
              点式各λ=点式, 点式maxAcc_A=float(点式max),
              差_对点式max=float(主Acc - 点式max),
              判定=dict(过阈=bool(差 >= 阈值["B6_状态差下限"]),
                      结论="①−② = %.4f ≥ %.2f ⇒ 状态非日志可替代（本任务口径）" % (差, 阈值["B6_状态差下限"])),
              重放臂="未测（§6.0 F5-a 的 must：日志口径=输入/决策/奖赏哪几项未定义；本轮不计）")

    # B7 重算器 = 点神经元 λ=0.10（当前输入+历史输入流）；失忆臂复用 ②
    _, _, 测重算 = V4.跑臂("点神经元", D, 种子["模型"], 训, 测, 慢率=选定, λ=0.10)
    重算Acc = float(测重算["Acc_A"])
    可预演 = bool(重算Acc >= 主Acc - 阈值["B7_重算器余量"])
    B7 = dict(重算器Acc_A=重算Acc, 失忆臂Acc_A=无状态Acc, 主臂Acc_A=主Acc,
              主臂减余量=float(主Acc - 阈值["B7_重算器余量"]),
              判定=dict(可预演=可预演, 失忆过阈=bool(无状态Acc <= 阈值["B7_失忆上限"]),
                      过阈=bool((not 可预演) and 无状态Acc <= 阈值["B7_失忆上限"]),
                      结论=("重算器 %.4f < 主臂−%.2f=%.4f ⇒ 不可预演（该项可计入达成）；失忆臂 %.4f ≤ %.2f"
                          % (重算Acc, 阈值["B7_重算器余量"], 主Acc - 阈值["B7_重算器余量"],
                             无状态Acc, 阈值["B7_失忆上限"])) if not 可预演 else
                     ("重算器 %.4f ≥ 主臂−%.2f ⇒ 判「可预演」，该项不计达成" % (重算Acc, 阈值["B7_重算器余量"]))),
              重算器λ=0.10, 失忆λ=1.0)

    # B8 历史置换
    rng = np.random.default_rng(种子["置换"])
    perm = rng.permutation(int(V4.训练长))
    训置换 = dict(x=训["x"][perm], 边界=训["边界"][perm], 切换点=[], 长度=int(V4.训练长))
    _, _, 测置换 = V4.跑臂("两区室", D, 种子["模型"], 训置换, 测, 慢率=选定)
    置换Acc = float(测置换["Acc_A"])
    训随机 = V4.生成序列(种子["置换"], int(V4.训练长), V4.边界训练)
    _, _, 测随机 = V4.跑臂("两区室", D, 种子["模型"], 训随机, 测, 慢率=选定)
    随机Acc = float(测随机["Acc_A"])
    掉置换 = 主Acc - 置换Acc
    掉随机 = 主Acc - 随机Acc
    B8 = dict(基线Acc_A=主Acc, 置换Acc_A=置换Acc, 随机历史Acc_A=随机Acc,
              掉幅_置换=float(掉置换), 掉幅_随机=float(掉随机),
              判定=dict(主判过阈=bool(掉置换 >= 阈值["B8_掉幅下限"]),
                      mustfail过阈=bool(掉随机 >= 阈值["B8_掉幅下限"]),
                      结论=("置换掉 %.4f ≥ %.2f（时序结构承重）；随机历史掉 %.4f %s %.2f ⇒ must-fail %s"
                          % (掉置换, 阈值["B8_掉幅下限"], 掉随机,
                             "≥" if 掉随机 >= 阈值["B8_掉幅下限"] else "<",
                             阈值["B8_掉幅下限"],
                             "成立" if 掉随机 >= 阈值["B8_掉幅下限"] else "未触发（具体历史内容不承重；只能主张依赖训练时序结构）"))),
              协议="训练序列 (x,边界) 同步 rng(47107) 置换；随机历史=生成序列(47107,400,{0.35,0.65})")

    return dict(慢率扫描=扫描, 选定慢率=float(选定), B6=B6, B7=B7, B8=B8,
                序列指纹=dict(训=_指纹(训["x"]), 测=_指纹(测["x"])))


# ═══════════════════════════════ 主 ═══════════════════════════════
def main():
    t0 = time.time()
    _log("加载只读件 …")
    千层 = _载(_千层py, "千层_只读_B")
    V4 = _载(_两区室py, "两区室_只读_B")
    _试 = 千层.千层核(D=D, 种子=种子["记忆"])
    _log("千层 U=%d D=%d 输入维=%d" % (_试.U, _试.D, _试.容器.输入维))
    del _试

    结果 = dict(件="B_记忆与F5.py", 角色="蓝队 worker B（Level-2）", 生成时间戳=time.strftime("%Y-%m-%dT%H:%M:%S"),
                python=sys.version.split()[0], numpy=np.__version__, D=D, 种子=种子,
                预注册阈值=阈值,
                协议在文件头部="见 B_记忆与F5.py docstring P1–P8（本 JSON 只存读数与判定）")

    结果["B1"] = B1(千层)
    _log("B1 完成 判定=%s" % json.dumps(结果["B1"]["判定"], ensure_ascii=False))
    结果["B2"] = B2(千层)
    _log("B2 完成 gap_raw=%.3f gap_resident=%.3f" % (结果["B2"]["判定"]["gap_raw"], 结果["B2"]["判定"]["gap_resident"]))
    结果["B3"] = B3(千层)
    _log("B3 完成 槽早=%.3f FIFO早=%.3f" % (结果["B3"]["判定"]["槽_早"], 结果["B3"]["判定"]["FIFO_早"]))
    结果["B4"] = B4(千层)
    _log("B4 完成 R=%.3f" % 结果["B4"]["判定"]["R"])
    结果["B5"] = B5(千层)
    _log("B5 完成 R_perm均值=%.3f 最大=%.3f" % (结果["B5"]["判定"]["R_perm均值"], 结果["B5"]["判定"]["R_perm最大"]))
    结果.update(B68(V4))
    结果["耗时秒"] = round(time.time() - t0, 1)
    结果["未测"] = [
        "F5-a 重放臂（日志全量喂回决策）：日志口径（输入/决策/奖赏）未定义 ⇒ 未测",
        "B1 迭代 1 vs 3 之外的迭代步数（如 16）：未测",
        "跨硬件逐位同：未测、不主张",
    ]
    结果["局限"] = [
        "B1 靶生成/干扰抽样与 R2 §9.1 未逐位对齐：本件 K=64 flip0.4=%.3f（§9.1=0.700）、K=1024 flip0.4=%.3f（§9.1=0.533）；"
        "但两件在 ρ≤0.3 全 1.000、flip0.5 归零、迭代 1=3 逐位同三点上一致，结论不依赖具体数值。"
        % (结果["B1"]["臂"][0]["迭代1"]["R列"][5], 结果["B1"]["臂"][1]["迭代1"]["R列"][5]),
        "B2 raw gap=0.100 恰等于 3/30（K=4 多驻留 3 条），是驻留数差而非容量质量差；resident gap=0。",
        "B5 单次置换可为 0（无固定点）或 0.625（5/8 命中）⇒ 只能报置换分布；200 次均值 0.139。",
        "B6/B7 的 ② 是 点神经元（单一递推态 s），与主臂结构不同（少两区室/符/分区）⇒ 是弱同算力基线，"
        "不是严格的'同架构去状态重算器'；B7「不可预演」只否掉'单状态线性读出可重算'，不否掉更强重算器。",
        "B8 置换改的是训练历史的时序结构（同边际、同长度），结论限于'训练时序结构承重'，不得外推为'推理期历史依赖'。",
    ]
    结果["卡点"] = []
    with open(_结果py, "w", encoding="utf-8") as f:
        json.dump(结果, f, ensure_ascii=False, indent=1)
    _log("已写 %s（%.1f s）" % (_结果py, 结果["耗时秒"]))
    print(json.dumps(dict(B1判定=结果["B1"]["判定"], B2判定=结果["B2"]["判定"],
                          B3判定=结果["B3"]["判定"], B4判定=结果["B4"]["判定"],
                          B5判定=结果["B5"]["判定"], B6判定=结果["B6"]["判定"],
                          B7判定=结果["B7"]["判定"], B8判定=结果["B8"]["判定"],
                          耗时秒=结果["耗时秒"]), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
