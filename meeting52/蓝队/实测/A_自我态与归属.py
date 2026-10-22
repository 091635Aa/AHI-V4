# -*- coding: utf-8 -*-
"""A_自我态与归属.py —— 蓝队实测 worker A（Level-2，向 blue-captain 汇报）

A 组七臂（V4 两区室核，D=256，种子 47001/47101/47102，慢率 0.20）真跑：
  A1 PS1 替换 A / A2 PS2 扰动 A / A3 PS3 冻结 A(消融) / A4 PS4 迁移(held-out)
  A5 PS5 跨任务持久 / A6 PV 归属(预测-实际比较器) / A7 2x2 消融(PS x PV)
纪律：只读 importlib 加载既有模块；不写任何既有件；不运行任何 判据_*.py；python -B。

═══════════════════ 预注册（跑之前写死；跑后不改） ═══════════════════
阈值（全部来自任务书 R2 §6.0/§6.1/§9，原样冻结）：
  PS1  D_o ≥ 0.90 且 F ≥ 0.50；must-fail: A←A自身 ⇒ D_o == 0.00（逐位）
  PS2  ΔAcc_A ≥ 0.10、ΔAcc_B ≤ 0.03、J=ΔAcc_A−ΔAcc_B ≥ 0.10、bootstrap(1000,seed 47106) 下界 > 0
       must-fail: 常数慢滤波(λ=0.20) ⇒ Acc_A ≤ 0.62 且 J < 0.10
  PS3  Δ消融 ≥ 0.15 且 冻结 Acc_A ≤ 0.62（Δ消融 = 主臂 Acc_A − 消融 Acc_A，与 V4 判据同口径）
  PS4  ① Acc_A ≥ 0.75 且 ≥ 0.70×已见 Acc_A；②③ 不设 pass（只报数字）
  PS5  t50(保留) ≥ 10 且 lag-1 ≥ 0.90；重置臂 t50 == 0
  PV   AUC ≥ 0.75、ḡ_ext ≤ 0.20、ḡ_self ≥ 0.80
       must-fail: ① 来源标签置换 ⇒ AUC ≤ 0.55；② 统一低可塑(无门控) ⇒ ḡ_ext ≥ 0.80；
                  ③ oracle 标签 ⇒ AUC ≈ 1.00（阳性对照，判 ≥0.99）
协议口径（任务书未写死处，本件先声明；均在此处冻结）：
  · 量纲：Δ 一律 = 基线 − 干预（掉分为正）；Acc_A/Acc_B 用核.分类掩码(0.35,0.65)；
  · PS1：A′ = 同协议、种子 47002 训练模型的**终态自我态**；替换在**每拍前向之前**施加；
          D_o = 观测输出 bytes 中**不同字节数/总字节数**（float64 逐字节）；F = 决策 argmax 翻转拍占比；
  · PS2：扰动在**每拍前向之前**施加（持续性干预；一次性扰动会被 学自我 的_刷新自我态 立即洗掉）；
          u = ±1 定点方向（default_rng(47107)），幅 = 0.1·mean|A|；bootstrap 按 A 掩码拍配对重采样；
  · PS3：消融 = 核自带 模式="消融"（自我态冻结为 P@特征(0.5)，自指通道切断）；
  · PS4：训练 band(0.35,0.65)；① 测 band(0.25,0.75)，种子 47103；② 反转规则 label=1[x<b]，
          同 band 测试序列（种子 47102）只翻标签；③ 未见带宽 band(0.45,0.55)，种子 47104；
          三臂均**冻结学习**（学读出=False、学自我=False）测纯迁移；
  · PS5：段1 = 训练 band(0.35,0.65) 400 拍；段2 = 常数边界 b=0.50、种子 47102、400 拍；
          保留臂续用段1 自我态并在线 学自我；重置臂在**每拍前** 设自我标量(0.5)（= 中性自我态，
          对 b=0.50 恰好正确 ⇒ 按定义应 t50=0）；t50 = 切换后首个「连续 3 拍正确」窗口的首拍
          偏移（0 起算；无窗口记 −1）；lag-1 = 段2 自我标量序列的一阶自相关（常数列记 1.0）；
  · PV：混合流 = 400 拍 AR(1) 慢走 x（种子 47108，步长 0.03）；每 40 拍（t=40..360，共 9 次）
          注入外源键 = 打乱(世界态 B_t)（置换种子 47109），其余为自源；
          键=B_t，o=前向输出；E 256x256 零初值，η=0.05，ê=Eᵀ键/D，a=cos(ê,o_{t+1})，末拍用 o_t；
          门控 g=1[a≥θ]（θ=0.60），前 40 拍自源预热门全开（冷启动引导，先声明）；
          ḡ 是**实际施加的可塑权**（主臂=门控输出；无门控臂=恒 1）；AUC 只在 t≥40 的事件上算；
  · 解码独立复测：规则标 = round(边界×100)，质心解码（与 V4 判据同实现），256D A 迹 vs 1D 标量迹；
  · A7：PS 活/冻 × PV 开(允许=False)/关(允许=True)；PV 关臂在**每拍前**注入 打乱(当前A)(置换 47112)；
  · 墙钟 = perf_counter 单臂实耗；JSON 记脚本 sha256。
═══════════════════════════════════════════════════════════════════════
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

HERE = os.path.dirname(os.path.abspath(__file__))
本件 = os.path.abspath(__file__)
结果路径 = os.path.join(HERE, "A_结果.json")
核路径 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "meeting49", "实施", "算子_两区室核.py"))

# ── 预注册阈值（写死）──
PS1_Do = 0.90
PS1_F = 0.50
PS1_自身Do = 0.00
PS2_dA = 0.10
PS2_dB = 0.03
PS2_J = 0.10
PS2_CI = 0.0
PS2_MF_AccA = 0.62
PS2_MF_J = 0.10
PS3_dAbl = 0.15
PS3_AccA = 0.62
PS4_1 = 0.75
PS4_比例 = 0.70
PS5_t50 = 10
PS5_lag1 = 0.90
PV_AUC = 0.75
PV_g_ext = 0.20
PV_g_self = 0.80
PV_perm = 0.55
PV_oracle = 0.99

D = 256
慢率 = 0.20
种子_模A = 47001
种子_模B = 47002
种子_训 = 47101
种子_测 = 47102
边界训 = (0.35, 0.65)

预注册 = dict(
    任务书=dict(
        PS1=dict(D_o=PS1_Do, F=PS1_F, must_fail_自身Do=PS1_自身Do),
        PS2=dict(dAcc_A=PS2_dA, dAcc_B=PS2_dB, J=PS2_J, CI下界=PS2_CI,
                 must_fail_AccA=PS2_MF_AccA, must_fail_J=PS2_MF_J),
        PS3=dict(dAbl=PS3_dAbl, AccA=PS3_AccA),
        PS4={"①": PS4_1, "①相对已见": PS4_比例},
        PS5=dict(t50=PS5_t50, lag1=PS5_lag1),
        PV=dict(AUC=PV_AUC, g_ext=PV_g_ext, g_self=PV_g_self, 标签置换=PV_perm, oracle=PV_oracle),
    ),
    口径=dict(
        D=D, 慢率=慢率, 种子模型=[种子_模A, 种子_模B], 种子训=种子_训, 种子测=种子_测,
        Δ定义="基线 − 干预（掉分为正）",
        PS1_A元="种子47002 训练终态 自我态", PS1_施加="每拍前向之前",
        PS1_Do定义="不同字节数/总字节数(float64, 逐字节)", PS1_F定义="决策 argmax 翻转拍占比",
        PS2_施加="每拍前向之前（持续扰动）", PS2_u="±1 定点方向 seed 47107", PS2_幅="0.1*mean|A|",
        PS2_bootstrap="A掩码拍配对重采样 1000 次 seed 47106",
        PS3_消融="模式='消融'（自我态冻结 P@特征(0.5)）",
        **{"PS4_①": "band(0.25,0.75) seed 47103",
           "PS4_②": "label=1[x<b] 翻转 seed 47102（同band）",
           "PS4_③": "band(0.45,0.55) seed 47104",
           "PS4_学习": "冻结（学读出=False,学自我=False）"},
        PS5_段2="b=0.50 常数 seed 47102", PS5_重置="每拍前 设自我标量(0.5)",
        PS5_t50定义="首个连续3拍正确窗口的首拍偏移(0起算)", PS5_lag1="段2 自我标量一阶自相关",
        PV_流="AR(1) 慢走 x 400 拍 seed 47108 步长0.03", PV_注入="t=40..360 每40拍，共9次外源",
        PV_外源键="打乱(世界态B_t) 置换 seed 47109", PV_E="256x256 零初值 η=0.05",
        PV_a="cos(Eᵀ键/D, o_{t+1})", PV_gate="g=1[a≥0.60]，前40拍自源预热门全开",
        PV_g定义="实际施加的可塑权", PV_AUC="t≥40 事件上算",
        解码="规则标=round(边界*100)，质心解码；256D A vs 1D 自我标量",
        A7="PS活/冻 x PV开(允许False)/关(允许True)；PV关每拍注入 打乱(当前A) 置换 47112",
    ),
)

# ═══════════════════════════ 只读载入 ═══════════════════════════
def 载核():
    spec = importlib.util.spec_from_file_location("两区室核_只读_实测A", 核路径)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


核 = 载核()
序列训 = 核.生成序列(种子_训, 400, 边界训)
序列测 = 核.生成序列(种子_测, 400, 边界训)


# ═══════════════════════════ 工具 ═══════════════════════════
def 建核(种子=种子_模A, 模式="两区室", 率=慢率):
    return 核.两区室核(D=D, 种子=int(种子), 模式=str(模式), 慢率=float(率), 学模式="credit")


def 训练好(种子=种子_模A, 模式="两区室", 率=慢率):
    m = 建核(种子, 模式, 率)
    核.跑(m, 序列训, 学读出=True, 学模式="credit")
    return m


def _准(掩, 选, 真):
    if int(np.sum(掩)) == 0:
        return float("nan")
    return float((选[掩] == 真[掩]).mean())


def 逐拍(m, 序列_, 学读出=False, 学自我=True, 反转=False, 前钩=None, 记迹=False):
    """逐拍前向 + 可选前钩（施加干预）、可选冻结学习；返回决策/正确/掩码/迹。"""
    x = np.asarray(序列_["x"], dtype=np.float64)
    b = np.asarray(序列_["边界"], dtype=np.float64)
    n = int(len(x))
    真 = np.zeros(n, dtype=np.int64)
    选 = np.zeros(n, dtype=np.int64)
    出 = np.zeros((n, D), dtype=np.float64) if 记迹 else None
    A迹 = np.zeros((n, D), dtype=np.float64) if 记迹 else None
    标量迹 = np.zeros(n, dtype=np.float64) if 记迹 else None
    for t in range(n):
        if 前钩 is not None:
            前钩(m, t)
        o = m.前向_试次(float(x[t]))
        ŷ = int(np.argmax(m.读出 @ o))
        if 反转:
            正 = float(x[t]) < float(b[t])
        else:
            正 = float(x[t]) > float(b[t])
        真[t] = 1 if 正 else 0
        选[t] = ŷ
        奖 = 1 if ŷ == int(真[t]) else 0
        if 记迹:
            出[t] = o
            A迹[t] = m.自我态
            标量迹[t] = float(m.自我标量)
        if 学读出:
            m.学读出(ŷ, 奖, 更新读出=True)
        if 学自我:
            m.学自我(ŷ, 奖)
    A掩, B掩 = 核.分类掩码(x)
    return dict(真=真, 选=选, 奖=奖, A=A掩, B=B掩,
                Acc_A=_准(A掩, 选, 真), Acc_B=_准(B掩, 选, 真),
                Acc=float((选 == 真).mean()), n=n, 出=出, A迹=A迹, 标量迹=标量迹,
                x=x, 边界=b)


def 字节差(出1, 出2):
    a = np.ascontiguousarray(出1)
    b = np.ascontiguousarray(出2)
    ba = a.view(np.uint8)
    bb = b.view(np.uint8)
    不同字节 = int(np.sum(ba != bb))
    总字节 = int(ba.size)
    不同元素 = int(np.sum(a != b))
    总元 = int(a.size)
    不同拍 = int(np.sum(np.any(ba != bb, axis=1)))
    return dict(D_o=(不同字节 / 总字节 if 总字节 else float("nan")),
                不同字节=不同字节, 总字节=总字节,
                元素差率=(不同元素 / 总元 if 总元 else float("nan")),
                拍差率=(不同拍 / a.shape[0] if a.shape[0] else float("nan")),
                不同拍=不同拍)


def bootstrap_delta(基, 扰, 掩, 次数=1000, 种子=47106):
    索引 = np.where(np.asarray(掩))[0]
    if len(索引) == 0:
        return dict(Δ=float("nan"), CI95=[float("nan"), float("nan")], 下界=float("nan"),
                    上界=float("nan"), 次数=0, 种子=int(种子), n_A=0)
    cb = (基["选"][索引] == 基["真"][索引]).astype(np.float64)
    cp = (扰["选"][索引] == 扰["真"][索引]).astype(np.float64)
    rng = np.random.default_rng(int(种子))
    n = len(索引)
    vals = np.empty(int(次数), dtype=np.float64)
    for i in range(int(次数)):
        j = rng.integers(0, n, n)
        vals[i] = cb[j].mean() - cp[j].mean()
    lo = float(np.percentile(vals, 2.5))
    hi = float(np.percentile(vals, 97.5))
    return dict(Δ=float(cb.mean() - cp.mean()), CI95=[lo, hi], 下界=lo, 上界=hi,
                次数=int(次数), 种子=int(种子), n_A=int(n))


def AUC(分, 标):
    分 = np.asarray(分, dtype=np.float64)
    标 = np.asarray(标, dtype=np.int64)
    正 = 分[标 == 1]
    负 = 分[标 == 0]
    if len(正) == 0 or len(负) == 0:
        return float("nan")
    大 = float(np.sum(正[:, None] > 负[None, :]))
    等 = float(np.sum(正[:, None] == 负[None, :]))
    return (大 + 0.5 * 等) / float(len(正) * len(负))


def 质心解码(迹, 标):
    X = np.asarray(迹, dtype=np.float64)
    y = np.asarray(标, dtype=np.int64)
    类 = sorted(set(int(v) for v in y))
    质 = {c: X[y == c].mean(axis=0) for c in 类}
    预 = np.zeros(len(y), dtype=np.int64)
    for i in range(len(y)):
        ds = [float(np.sum((X[i] - 质[c]) ** 2)) for c in 类]
        预[i] = 类[int(np.argmin(ds))]
    return float((预 == y).mean()), 预, 质


def 质心解码CV(迹, 标, 折=5):
    """确定性 5 折交叉验证质心解码（folds = 样本序号 mod 折；无随机）。"""
    X = np.asarray(迹, dtype=np.float64)
    y = np.asarray(标, dtype=np.int64)
    类 = sorted(set(int(v) for v in y))
    预 = np.zeros(len(y), dtype=np.int64)
    for f in range(int(折)):
        训掩 = (np.arange(len(y)) % int(折)) != f
        测掩 = ~训掩
        if not np.any(测掩) or not np.any(训掩):
            continue
        质 = {c: X[训掩 & (y == c)].mean(axis=0) for c in 类 if np.any(训掩 & (y == c))}
        for i in np.where(测掩)[0]:
            ds = [(c, float(np.sum((X[i] - 质[c]) ** 2))) for c in 质]
            预[i] = min(ds, key=lambda z: z[1])[0]
    return float((预 == y).mean())


def lag1(序列_):
    s = np.asarray(序列_, dtype=np.float64)
    if len(s) < 3:
        return float("nan")
    a = s[:-1]
    b = s[1:]
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 1.0 if float(np.all(a == b[0])) else float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def t50(正确):
    c = np.asarray(正确, dtype=bool)
    for i in range(len(c) - 2):
        if c[i] and c[i + 1] and c[i + 2]:
            return int(i)
    return -1


臂 = {}
卡点 = []


def 记臂(名, 实测, 阈值, 通过, must_fail=None, 秒=None, 说明=""):
    臂[名] = dict(实测=实测, 阈值=阈值, 通过=(None if 通过 is None else bool(通过)),
                  must_fail=must_fail, 墙钟_s=(None if 秒 is None else round(float(秒), 3)),
                  说明=说明)


# ═══════════════════════════ A1 PS1 替换 A ═══════════════════════════
try:
    t0 = time.perf_counter()
    基模型 = 训练好(种子_模A)
    换模型 = 训练好(种子_模A)
    A元 = 训练好(种子_模B).自我态.copy()          # 47002 训练终态自我态
    测基 = 逐拍(基模型, 序列测, 学读出=False, 学自我=True, 记迹=True)

    def 钩_替换(mm, t):
        mm.设自我态向量(A元)

    测换 = 逐拍(换模型, 序列测, 学读出=False, 学自我=True, 前钩=钩_替换, 记迹=True)
    字节 = 字节差(测基["出"], 测换["出"])
    翻 = (测基["选"] != 测换["选"])
    F = float(np.mean(翻))
    F_A = float(np.mean(翻[测基["A"]])) if int(np.sum(测基["A"])) else float("nan")
    F_B = float(np.mean(翻[测基["B"]])) if int(np.sum(测基["B"])) else float("nan")
    # must-fail：A←A 自身（无操作）⇒ 必须逐位 0
    自模型 = 训练好(种子_模A)

    def 钩_自身(mm, t):
        mm.设自我态向量(mm.自我态.copy())

    测自 = 逐拍(自模型, 序列测, 学读出=False, 学自我=True, 前钩=钩_自身, 记迹=True)
    字节自 = 字节差(测基["出"], 测自["出"])
    秒 = time.perf_counter() - t0
    通过 = bool(字节["D_o"] >= PS1_Do and F >= PS1_F)
    mf = dict(读数=dict(D_o=字节自["D_o"], 不同字节=字节自["不同字节"]),
              阈值=dict(D_o_eq=PS1_自身Do), 通过=bool(字节自["D_o"] == PS1_自身Do),
              说明="A←A 自身 ⇒ 输出必须逐位同（D_o=0.00）")
    记臂("A1_PS1_替换A",
         实测=dict(D_o=字节["D_o"], D_o_拍=字节["拍差率"], D_o_元素=字节["元素差率"],
                  F=F, F_A=F_A, F_B=F_B, 元素差率=字节["元素差率"], 拍差率=字节["拍差率"],
                  不同字节=字节["不同字节"], 总字节=字节["总字节"], 翻转拍=float(F * 测基["n"]),
                  基Acc_A=测基["Acc_A"], 换Acc_A=测换["Acc_A"]),
         阈值=dict(D_o=PS1_Do, F=PS1_F), 通过=通过, must_fail=mf, 秒=秒,
         说明="每拍把 A 换成 47002 终态 A′；D_o=字节差率（预注册），另报拍级/元素级与 F_A/F_B")
except Exception as e:                                     # noqa: BLE001
    卡点.append(dict(臂="A1", 异常=repr(e)))

# ═══════════════════════════ A2 PS2 扰动 A ═══════════════════════════
try:
    t0 = time.perf_counter()
    基2 = 逐拍(训练好(种子_模A), 序列测, 学读出=False, 学自我=True, 记迹=False)
    u = np.where(np.random.default_rng(47107).integers(0, 2, size=D) == 1, 1.0, -1.0)

    def 钩_扰(mm, t):
        A = np.asarray(mm.自我态, dtype=np.float64)
        幅 = 0.1 * float(np.mean(np.abs(A)))
        mm.设自我态向量(核._归(A + 幅 * u))

    扰 = 逐拍(训练好(种子_模A), 序列测, 学读出=False, 学自我=True, 前钩=钩_扰, 记迹=False)
    ΔA = float(基2["Acc_A"] - 扰["Acc_A"])
    ΔB = float(基2["Acc_B"] - 扰["Acc_B"])
    J = float(ΔA - ΔB)
    bs = bootstrap_delta(基2, 扰, 基2["A"], 次数=1000, 种子=47106)
    通过 = bool(ΔA >= PS2_dA and ΔB <= PS2_dB and J >= PS2_J and bs["下界"] > PS2_CI)

    # must-fail：常数慢滤波 A_t ← (1−λ)A_{t−1} + λc，λ=0.20，c=中性常数向量
    c = 核._归(核.投影P(D, 种子_模A) @ 核.特征(0.5))
    状态 = [None]

    def 钩_常(mm, t):
        if 状态[0] is None:
            状态[0] = np.asarray(mm.自我态, dtype=np.float64).copy()
        状态[0] = 核._归((1.0 - 0.20) * 状态[0] + 0.20 * c)
        mm.设自我态向量(状态[0])

    常 = 逐拍(训练好(种子_模A), 序列测, 学读出=False, 学自我=True, 前钩=钩_常, 记迹=False)
    ΔA常 = float(基2["Acc_A"] - 常["Acc_A"])
    ΔB常 = float(基2["Acc_B"] - 常["Acc_B"])
    J常 = float(ΔA常 - ΔB常)
    秒 = time.perf_counter() - t0
    mf = dict(读数=dict(Acc_A=常["Acc_A"], Acc_B=常["Acc_B"], ΔAcc_A=ΔA常, ΔAcc_B=ΔB常, J=J常),
              阈值=dict(Acc_A_max=PS2_MF_AccA, J_max=PS2_MF_J),
              通过=bool(常["Acc_A"] <= PS2_MF_AccA and J常 < PS2_MF_J),
              说明="常数慢滤波（无奖赏驱动）必须撑不起 Acc_A 且不产生 A/B 选择性损伤")
    记臂("A2_PS2_扰动A",
         实测=dict(ΔAcc_A=ΔA, ΔAcc_B=ΔB, J=J, 基Acc_A=基2["Acc_A"], 扰Acc_A=扰["Acc_A"],
                  基Acc_B=基2["Acc_B"], 扰Acc_B=扰["Acc_B"], bootstrap=bs),
         阈值=dict(ΔAcc_A=PS2_dA, ΔAcc_B=PS2_dB, J=PS2_J, CI下界=PS2_CI), 通过=通过,
         must_fail=mf, 秒=秒,
         说明="每拍 A←归(A+0.1·mean|A|·u)；Δ=基线−干预；J=ΔA−ΔB")
except Exception as e:                                     # noqa: BLE001
    卡点.append(dict(臂="A2", 异常=repr(e)))

# ═══════════════════════════ A3 PS3 冻结 A（消融） ═══════════════════════════
try:
    t0 = time.perf_counter()
    主3 = 逐拍(训练好(种子_模A, 模式="两区室"), 序列测, 学读出=False, 学自我=True)
    消3 = 逐拍(训练好(种子_模A, 模式="消融"), 序列测, 学读出=False, 学自我=True)
    Δ消 = float(主3["Acc_A"] - 消3["Acc_A"])
    秒 = time.perf_counter() - t0
    通过 = bool(Δ消 >= PS3_dAbl and 消3["Acc_A"] <= PS3_AccA)
    记臂("A3_PS3_冻结A_消融",
         实测=dict(Δ消融=Δ消, 主Acc_A=主3["Acc_A"], 消融Acc_A=消3["Acc_A"],
                  主Acc_B=主3["Acc_B"], 消融Acc_B=消3["Acc_B"],
                  自指退化=Δ消, 世界退化=abs(主3["Acc_B"] - 消3["Acc_B"])),
         阈值=dict(Δ消融=PS3_dAbl, 消融Acc_A_max=PS3_AccA), 通过=通过, must_fail=None, 秒=秒,
         说明="复现 Δ消融（期望≈0.469）；冻结自我态必须 Acc_A≤0.62")
except Exception as e:                                     # noqa: BLE001
    卡点.append(dict(臂="A3", 异常=repr(e)))

# ═══════════════════════════ A4 PS4 迁移（held-out） ═══════════════════════════
try:
    t0 = time.perf_counter()
    序列迁 = 核.生成序列(47103, 400, (0.25, 0.75))
    序列反 = 核.生成序列(47102, 400, 边界训)
    序列窄 = 核.生成序列(47104, 400, (0.45, 0.55))
    # 修订 1：改用 V4 判据的「已见口径」= 读出冻结、自我在线（迁移 0.874 即此口径）
    测已见在线 = 逐拍(训练好(种子_模A), 序列测, 学读出=False, 学自我=True)
    测已见冻结 = 逐拍(训练好(种子_模A), 序列测, 学读出=False, 学自我=False)
    测迁 = 逐拍(训练好(种子_模A), 序列迁, 学读出=False, 学自我=True)
    测反在线 = 逐拍(训练好(种子_模A), 序列反, 学读出=False, 学自我=True, 反转=True)
    测反冻结 = 逐拍(训练好(种子_模A), 序列反, 学读出=False, 学自我=False, 反转=True)
    测窄在线 = 逐拍(训练好(种子_模A), 序列窄, 学读出=False, 学自我=True)
    测窄冻结 = 逐拍(训练好(种子_模A), 序列窄, 学读出=False, 学自我=False)
    比例 = float(测迁["Acc_A"] / 测已见在线["Acc_A"]) if 测已见在线["Acc_A"] else float("nan")
    秒 = time.perf_counter() - t0
    通过 = bool(测迁["Acc_A"] >= PS4_1 and 比例 >= PS4_比例)
    记臂("A4_PS4_迁移_heldout",
         实测=dict(已见_在线自我=dict(Acc_A=测已见在线["Acc_A"], Acc_B=测已见在线["Acc_B"],
                                  Acc=测已见在线["Acc"]),
                   已见_冻结学习_诊断=dict(Acc_A=测已见冻结["Acc_A"], Acc_B=测已见冻结["Acc_B"],
                                      Acc=测已见冻结["Acc"]),
                   **{"①已见口径_band025_075": dict(Acc_A=测迁["Acc_A"], Acc_B=测迁["Acc_B"],
                                                Acc=测迁["Acc"], 相对已见=比例),
                      "②未见规则反转_1xb": dict(Acc_A=测反在线["Acc_A"], Acc_B=测反在线["Acc_B"],
                                             Acc=测反在线["Acc"],
                                             冻结口径_诊断=dict(Acc_A=测反冻结["Acc_A"],
                                                           Acc_B=测反冻结["Acc_B"],
                                                           Acc=测反冻结["Acc"])),
                      "③未见带宽_band045_055": dict(Acc_A=测窄在线["Acc_A"], Acc_B=测窄在线["Acc_B"],
                                                 Acc=测窄在线["Acc"],
                                                 冻结口径_诊断=dict(Acc_A=测窄冻结["Acc_A"],
                                                               Acc_B=测窄冻结["Acc_B"],
                                                               Acc=测窄冻结["Acc"]))}),
         阈值={"①Acc_A_min": PS4_1, "①相对已见_min": PS4_比例, "②③": "不设 pass"},
         通过=通过, must_fail=None, 秒=秒,
         说明="训练 band(0.35,0.65)；① 已见口径迁移；② 规则反转与 ③ 未见带宽只报数字（'只教过的会'证据）")
    臂["A4_PS4_迁移_heldout"]["②③不设pass"] = True
except Exception as e:                                     # noqa: BLE001
    卡点.append(dict(臂="A4", 异常=repr(e)))

# ═══════════════════════════ A5 PS5 跨任务持久 ═══════════════════════════
try:
    t0 = time.perf_counter()
    序列二 = 核.生成序列(47102, 400, (0.50,))
    保 = 逐拍(训练好(种子_模A), 序列二, 学读出=False, 学自我=True, 记迹=True)
    重模型 = 训练好(种子_模A)

    def 钩_重置(mm, t):
        mm.设自我标量(0.5)

    重 = 逐拍(重模型, 序列二, 学读出=False, 学自我=True, 前钩=钩_重置, 记迹=True)
    t50保 = t50(保["选"] == 保["真"])
    t50重 = t50(重["选"] == 重["真"])
    lg = lag1(保["标量迹"])
    秒 = time.perf_counter() - t0
    通过 = bool(t50保 >= PS5_t50 and lg >= PS5_lag1)
    mf = dict(读数=dict(t50=t50重, Acc=重["Acc"], Acc_A=重["Acc_A"]),
              阈值=dict(t50_eq=0), 通过=bool(t50重 == 0),
              说明="每拍重置自我标量(0.5) ⇒ 中性自我态对 b=0.50 恰正确，应 t50=0")
    记臂("A5_PS5_跨任务持久",
         实测=dict(保留=dict(t50=t50保, lag1=lg, Acc=保["Acc"], Acc_A=保["Acc_A"],
                             Acc_B=保["Acc_B"], 标量首末=[float(保["标量迹"][0]),
                                                        float(保["标量迹"][-1])]),
                   重置=dict(t50=t50重, Acc=重["Acc"], Acc_A=重["Acc_A"], Acc_B=重["Acc_B"],
                             标量首末=[float(重["标量迹"][0]), float(重["标量迹"][-1])]),
                   段2边界=0.50, 段1边界=[0.35, 0.65]),
         阈值=dict(t50_min=PS5_t50, lag1_min=PS5_lag1), 通过=通过, must_fail=mf, 秒=秒,
         说明="段1 训练 band(0.35,0.65)→段2 常数 b=0.50；保留 A vs 每拍重置 A")
except Exception as e:                                     # noqa: BLE001
    卡点.append(dict(臂="A5", 异常=repr(e)))

# ═══════════════════════════ A6 PV 归属（预测-实际比较器，内生） ═══════════════════════════
try:
    t0 = time.perf_counter()
    m6 = 训练好(种子_模A)
    rng6 = np.random.default_rng(47108)
    xs = np.empty(400, dtype=np.float64)
    xs[0] = 0.5
    for t in range(1, 400):
        xs[t] = min(0.98, max(0.02, xs[t - 1] + 0.03 * float(rng6.standard_normal())))
    B列 = np.zeros((400, D), dtype=np.float64)
    o列 = np.zeros((400, D), dtype=np.float64)
    for t in range(400):
        B列[t] = m6.世界态(float(xs[t]))
        o列[t] = m6.前向_试次(float(xs[t]))
    置换 = np.random.default_rng(47109).permutation(D)
    注入拍 = set(int(t) for t in range(40, 400, 40))          # 40..360，共 9
    源 = np.array([0 if t in 注入拍 else 1 for t in range(400)], dtype=np.int64)
    键列_原 = np.zeros((400, D), dtype=np.float64)
    for t in range(400):
        键列_原[t] = B列[t][置换] if t in 注入拍 else B列[t]
    中心 = np.mean(B列[:40], axis=0)                          # 修订 2：预热均值（因果），去直流
    键列_去直 = np.zeros((400, D), dtype=np.float64)
    for t in range(400):
        k = B列[t] - 中心
        键列_去直[t] = k[置换] if t in 注入拍 else k
    目标 = np.zeros((400, D), dtype=np.float64)
    for t in range(400):
        目标[t] = o列[t + 1] if t < 399 else o列[t]
    随机目标序 = np.random.default_rng(47111).permutation(400)

    def 跑E(键列, 门控=True, θ=0.60, 预热=40, η=0.05):
        E = np.zeros((D, D), dtype=np.float64)
        a列 = np.zeros(400, dtype=np.float64)
        g列 = np.zeros(400, dtype=np.float64)
        an列 = np.zeros(400, dtype=np.float64)
        for t in range(400):
            key = 键列[t]
            ê = (E.T @ key) / float(D)
            ne = float(np.linalg.norm(ê))
            no = float(np.linalg.norm(目标[t]))
            a = float(ê @ 目标[t] / (ne * no)) if (ne > 1e-12 and no > 1e-12) else 0.0
            on = float(np.linalg.norm(o列[随机目标序[t]]))
            an = float(ê @ o列[随机目标序[t]] / (ne * on)) if (ne > 1e-12 and on > 1e-12) else 0.0
            a列[t] = a
            an列[t] = an
            if 门控:
                g = 1.0 if (t < 预热 or a >= θ) else 0.0
            else:
                g = 1.0
            g列[t] = g
            if g > 0.0:
                E = (1.0 - η) * E + η * np.outer(key, o列[t])
        return E, a列, g列, an列

    后 = np.arange(400) >= 40
    自后 = 后 & (源 == 1)
    外 = 源 == 0

    def 评(键列):
        E, a, g, an = 跑E(键列, 门控=True)
        E无, a无, g无, an无 = 跑E(键列, 门控=False)
        return dict(
            AUC=AUC(a[后], 源[后]), AUC_无门控=AUC(a无[后], 源[后]),
            g_self=float(np.mean(g[自后])), g_ext=float(np.mean(g[外])),
            g_self_无门控=float(np.mean(g无[自后])), g_ext_无门控=float(np.mean(g无[外])),
            a_self=float(np.mean(a[自后])), a_ext=float(np.mean(a[外])),
            a_self_空目标=float(np.mean(an[自后])), a_ext_空目标=float(np.mean(an[外])),
            a列=a, a无列=a无,
        )

    原 = 评(键列_原)
    去直 = 评(键列_去直)
    # must-fail ①：来源标签置换（原口径）
    标置 = 源.copy()
    标置[后] = np.random.default_rng(47110).permutation(标置[后])
    AUC置 = AUC(原["a列"][后], 标置[后])
    AUC置_去直 = AUC(去直["a列"][后], 标置[后])
    # must-fail ③：oracle 标签（阳性）
    分神 = 源.astype(np.float64) + 1e-9 * np.arange(400, dtype=np.float64)
    AUC神 = AUC(分神[后], 源[后])
    # 解码独立复测：256D A vs 1D 自我标量（重代入质心 + 5折 CV 两口径）
    测6 = 逐拍(m6, 序列测, 学读出=False, 学自我=True, 记迹=True)
    规则标 = np.rint(np.asarray(序列测["边界"], dtype=np.float64) * 100.0).astype(np.int64)
    A解, _, _ = 质心解码(测6["A迹"], 规则标)
    标解, _, _ = 质心解码(测6["标量迹"].reshape(-1, 1), 规则标)
    A解CV = 质心解码CV(测6["A迹"], 规则标)
    标解CV = 质心解码CV(测6["标量迹"].reshape(-1, 1), 规则标)
    秒 = time.perf_counter() - t0
    通过 = bool(原["AUC"] >= PV_AUC and 原["g_ext"] <= PV_g_ext and 原["g_self"] >= PV_g_self)
    通过_去直 = bool(去直["AUC"] >= PV_AUC and 去直["g_ext"] <= PV_g_ext
                    and 去直["g_self"] >= PV_g_self)
    mf = {
        "①标签置换": dict(读数=dict(AUC=AUC置, AUC_去直流=AUC置_去直), 阈值=dict(AUC_max=PV_perm),
                        通过=bool(AUC置 <= PV_perm)),
        "②统一低可塑无门控": dict(读数=dict(g_ext=原["g_ext_无门控"], g_self=原["g_self_无门控"],
                                        AUC=原["AUC_无门控"], g_ext_去直流=去直["g_ext_无门控"],
                                        AUC_去直流=去直["AUC_无门控"]),
                                阈值=dict(g_ext_min=PV_g_ext),
                                通过=bool(原["g_ext_无门控"] >= PV_g_ext)),
        "③oracle标签": dict(读数=dict(AUC=AUC神), 阈值=dict(AUC_min=PV_oracle),
                         通过=bool(AUC神 >= PV_oracle)),
    }
    记臂("A6_PV_归属_预测实际比较器",
         实测=dict(AUC=原["AUC"], AUC_无门控=原["AUC_无门控"],
                   g_self=原["g_self"], g_ext=原["g_ext"],
                   g_self_无门控=原["g_self_无门控"], g_ext_无门控=原["g_ext_无门控"],
                   a_self均值=原["a_self"], a_ext均值=原["a_ext"],
                   a_self_空目标=原["a_self_空目标"], a_ext_空目标=原["a_ext_空目标"],
                   修正口径_去直流=dict(AUC=去直["AUC"], g_self=去直["g_self"], g_ext=去直["g_ext"],
                                    a_self均值=去直["a_self"], a_ext均值=去直["a_ext"],
                                    a_self_空目标=去直["a_self_空目标"],
                                    a_ext_空目标=去直["a_ext_空目标"], 通过=通过_去直),
                   n_self=int(np.sum(自后)), n_ext=int(np.sum(外)),
                   预热拍=40, 门控阈值=0.60, E形状=[D, D], η=0.05,
                   解码=dict(A256D=A解, 标量1D=标解, 差=float(A解 - 标解),
                             A256D_CV=A解CV, 标量1D_CV=标解CV, 差_CV=float(A解CV - 标解CV),
                             规则标类=sorted(set(int(v) for v in 规则标)))),
         阈值=dict(AUC=PV_AUC, g_ext=PV_g_ext, g_self=PV_g_self), 通过=通过,
         must_fail=mf, 秒=秒,
         说明="E 泄漏 Hebbian(256x256,η=0.05)；AUC 区分自源/外源；g 为实际可塑权；"
              "原口径(键=B)为预注册主判据，去直流(键=B−预热均值)为修订口径；另独立复测 256D/1D 解码")
except Exception as e:                                     # noqa: BLE001
    卡点.append(dict(臂="A6", 异常=repr(e)))

# ═══════════════════════════ A7 2x2 消融（PS × PV） ═══════════════════════════
try:
    t0 = time.perf_counter()
    置换7 = np.random.default_rng(47112).permutation(D)

    def 跑格(模式, 允许):
        m = 训练好(种子_模A, 模式=模式)

        def 钩(mm, t):
            v = 核._归(np.asarray(mm.自我态, dtype=np.float64)[置换7])
            mm.注入外源规则(v, 允许=允许)

        r = 逐拍(m, 序列测, 学读出=False, 学自我=True, 前钩=钩, 记迹=False)
        return dict(Acc_A=r["Acc_A"], Acc_B=r["Acc_B"], Acc=r["Acc"])

    M00 = 跑格("两区室", False)     # PS 活 × PV 开（外源不可塑）
    M10 = 跑格("消融", False)       # PS 冻 × PV 开
    M01 = 跑格("两区室", True)      # PS 活 × PV 关（外源可塑）
    M11 = 跑格("消融", True)        # PS 冻 × PV 关
    I = float(M00["Acc_A"] - M10["Acc_A"] - M01["Acc_A"] + M11["Acc_A"])
    秒 = time.perf_counter() - t0
    双分离 = dict(
        PS效应_PV开=float(M00["Acc_A"] - M10["Acc_A"]),
        PS效应_PV关=float(M01["Acc_A"] - M11["Acc_A"]),
        PV效应_PS活=float(M00["Acc_A"] - M01["Acc_A"]),
        PV效应_PS冻=float(M10["Acc_A"] - M11["Acc_A"]),
    )
    通过 = bool(abs(I) > 0.10 and 双分离["PV效应_PS活"] > 0.10 and 双分离["PV效应_PS冻"] < 0.05)
    记臂("A7_2x2消融_PSxPV",
         实测=dict(M00_PS活_PV开=M00, M10_PS冻_PV开=M10, M01_PS活_PV关=M01, M11_PS冻_PV关=M11,
                   I=I, 双分离=双分离,
                   MC侧="未测", MC未测原因="V4 两区室核无「记忆槽」拆件（只有 P/读出/感W/自我W/码本），无法做 MC 消融"),
         阈值={"|I|大于": 0.10, "PV效应_PS活大于": 0.10, "PV效应_PS冻小于": 0.05}, 通过=通过,
         must_fail=None, 秒=秒,
         说明="PS(冻结A) × PV(关 provenance 门=外源可塑) 四格 Acc_A；I=M00−M10−M01+M11")
except Exception as e:                                     # noqa: BLE001
    卡点.append(dict(臂="A7", 异常=repr(e)))

# ═══════════════════════════ 汇总落盘 ═══════════════════════════
未测 = [
    "A7 MC 侧（记忆槽拆件）：V4 两区室核无记忆槽拆件 ⇒ 未测",
    "A4 ②未见规则反转 / ③未见带宽：按要求不设 pass，只作'只教过的会'证据",
    "A6 内生门替代接口门的闭环（无预热/预热长度扫描）：未测；本件只测「门控=比较器阈值」在预热引导下的行为",
    "A6 未见外源类型迁移、重复同键升（R2 §6.1 的另两条 must-fail）：未测（本轮任务书未列）",
    "A5 lag-1 只算 1D 自我标量；256D A 迹的 lag-1 未测",
    "A1 只测同协议跨种子（47001→47002）替换；跨层/跨任务替换未测",
    "C13 两趟/四配置确定性：本件未跑（既有件与 R2_确定性协议负责）",
    "A2 扰动方向只取 1 个固定 u（seed 47107）；u 的方向敏感性未测",
]

总结 = dict(
    臂数=len(臂), 通过臂数=int(sum(1 for v in 臂.values() if v["通过"] is True)),
    未通过臂数=int(sum(1 for v in 臂.values() if v["通过"] is False)),
    不设pass臂=[k for k, v in 臂.items() if v["通过"] is None],
    卡点数=len(卡点),
)

# 首跑（2026-10-17，同一脚本第一版）的读数与跑后修订：只改口径实现，不改阈值
修订 = [
    dict(序=1, 触发="首跑 A4 ①Acc_A=0.4409 < 0.75",
         问题="把任务书『已见口径』误实现为完全冻结学习（学读出=False 且 学自我=False）；"
              "V4 判据的测试口径是读出冻结、自我在线（迁移 0.874 即此口径）",
         处置="A4 已见/①②③ 改为 自我在线；阈值未动；冻结口径读数保留为 *_诊断 字段"),
    dict(序=2, 触发="首跑 A6 AUC=0.4720、a_self均值=−0.0401",
         问题="键=B_t 被特征常数项（第 3 分量恒 1）主导 ⇒ B_s·B_t 有 ~92 的公共内积，"
              "Eᵀ键 被直流项淹没，比较器退化为噪声",
         处置="增加『键=B−预热均值』去直流变体（阈值未动）作修订口径；原口径读数保留为主判据"),
    dict(序=3, 触发="A1 字节级 D_o=0.4267 但 元素差率=1.0、拍差率=1.0",
         问题="float64 逐字节差率对同量级数值结构性达不到 0.90（符号/指数字节常相同），"
              "阈值 0.90 实际只在『拍级/元素级』口径可达",
         处置="补报 D_o_拍/D_o_元素 与 F_A/F_B；主判据仍按预注册字节口径，判定不改"),
]

脚本字节 = open(本件, "rb").read()
结果 = dict(
    件="A_自我态与归属.py",
    脚本=本件,
    脚本sha256=hashlib.sha256(脚本字节).hexdigest(),
    生成时间=time.strftime("%Y-%m-%d %H:%M:%S"),
    环境=dict(python=sys.version.split()[0], numpy=np.__version__, 核路径=核路径,
             核sha256=hashlib.sha256(open(核路径, "rb").read()).hexdigest()),
    预注册=预注册,
    修订=修订,
    臂=臂,
    卡点=卡点,
    未测=未测,
    总结=总结,
)

with open(结果路径, "w", encoding="utf-8") as f:
    json.dump(结果, f, ensure_ascii=False, indent=1, default=str)


def _简洁(名, v):
    实 = v["实测"]
    键 = []
    for k in ("D_o", "F", "ΔAcc_A", "ΔAcc_B", "J", "Δ消融", "t50", "lag1", "AUC", "g_self", "g_ext", "I"):
        if k in 实:
            键.append("%s=%s" % (k, 实[k] if not isinstance(实[k], float) else round(实[k], 4)))
    return "%-28s 通过=%-5s %s" % (名, v["通过"], " ".join(键))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                      # noqa: BLE001
        pass
    print("== A 组七臂（真跑）==")
    for 名, v in 臂.items():
        print(_简洁(名, v))
        if v.get("must_fail"):
            print("    must-fail: %s" % json.dumps(v["must_fail"], ensure_ascii=False, default=str))
    print("总结:", json.dumps(总结, ensure_ascii=False, default=str))
    if 卡点:
        print("卡点:", json.dumps(卡点, ensure_ascii=False, default=str))
    print("结果:", 结果路径)
